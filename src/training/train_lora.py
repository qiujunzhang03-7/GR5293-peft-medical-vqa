"""
LoRA fine-tuning pipeline for Qwen2-VL-2B on VQA-RAD.

This script is the **canonical training pipeline** for the project. Members
2 (QLoRA) and 3 (DoRA) extend it by changing only the ``LoRATrainingConfig``:

* QLoRA → set ``load_in_4bit=True`` and the bnb quantization config
* DoRA  → set ``use_dora=True`` in the LoRA config

Everything else (data, prompt template, optimizer, evaluation, profiling)
is shared, so the comparison across methods is apples-to-apples.

Outputs
-------
``output_dir`` (default ``checkpoints/lora_quick/``) will contain:

* ``adapter_model.safetensors`` — the LoRA weights (small, ~10 MB)
* ``adapter_config.json``       — PEFT config
* ``training_metrics.json``     — params, peak GPU memory, time/epoch,
                                  trainable %, and post-training eval metrics
* ``lora_predictions.jsonl``    — per-example predictions on the test set
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.vqarad_dataset import VQARadDataset
from src.evaluation.evaluate_baseline import _per_example_open_scores, generate_answer
from src.evaluation.metrics import (
    compute_all_metrics,
    per_example_correct,
)
from src.evaluation.statistical_tests import bootstrap_ci
from src.training.data_collator import QwenVLSFTCollator
from src.training._enable_input_grads import enable_input_grads
from src.utils.profiling import (
    cleanup_gpu,
    count_parameters,
    get_gpu_memory,
    print_parameter_summary,
    reset_peak_gpu_memory,
)
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LoRATrainingConfig:
    """All knobs in one place. Loadable from YAML.

    The defaults here are tuned for **a Colab T4 (16 GB) quick run** —
    a 200-example subset, 1 epoch, rank=8. Members 2/3's full runs will
    override these (see ``configs/lora_rank8.yaml``).
    """

    # --- Model ---
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    dtype: str = "float16"          # "float16" | "bfloat16"
    load_in_4bit: bool = False      # Member 2 flips this to True for QLoRA

    # --- Data ---
    train_split: str = "train"
    train_max_examples: Optional[int] = 200   # quick run; set None for full
    eval_split: str = "test"
    eval_max_examples: Optional[int] = None   # always evaluate on full test

    # --- LoRA hyperparameters ---
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    use_dora: bool = False          # Member 3 flips this to True for DoRA

    # --- Optimization ---
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    # --- Logging / IO ---
    logging_steps: int = 5
    output_dir: str = "checkpoints/lora_quick"
    seed: int = 42
    cache_dir: Optional[str] = None

    # --- Eval after training ---
    run_eval_after_training: bool = True
    eval_max_new_tokens: int = 64

    @classmethod
    def from_yaml(cls, path: str) -> "LoRATrainingConfig":
        """Load config from a YAML file (only known keys; extras ignored)."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Tolerate nested dicts (e.g. ``lora: {r: 8, alpha: 16}``)
        flat = {}
        for k, v in data.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}_{kk}" if not hasattr(cls, kk) else kk] = vv
            else:
                flat[k] = v
        # Filter to fields we actually have
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in flat.items() if k in known})


# ---------------------------------------------------------------------------
# Apply LoRA to the base model
# ---------------------------------------------------------------------------
def apply_lora_to_qwen(model, cfg: LoRATrainingConfig):
    """Wrap the Qwen2-VL model with PEFT LoRA adapters.

    This is the single line of code that QLoRA / DoRA experiments will
    differ on — they pass different ``LoraConfig`` arguments. By
    centralizing the call here, we guarantee comparable adapter geometry
    across methods.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=cfg.use_dora,
    )
    model = get_peft_model(model, lora_config)
    return model


# ---------------------------------------------------------------------------
# Model + processor loading (with optional 4-bit quantization for QLoRA)
# ---------------------------------------------------------------------------
def _load_base_model(cfg: LoRATrainingConfig):
    """Load Qwen2-VL with the right precision / quantization for ``cfg``."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(cfg.dtype, torch.float16)

    quant_config = None
    if cfg.load_in_4bit:
        # Member 2's QLoRA branch — kept here so Member 1's pipeline already
        # supports it without further changes.
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quant_config,
        cache_dir=cfg.cache_dir,
    )

    # Prepare for k-bit training if quantized (handles dtype casts +
    # input/output gradient hooks)
    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing,
        )

    if cfg.gradient_checkpointing and not cfg.load_in_4bit:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required when gradient checkpointing

    # Fix image resolution to avoid variable-resolution token mismatch.
    # min_pixels = 256*28*28; max_pixels = 768*28*28 — Qwen2-VL recommended
    # range that keeps spatial-merge math integral and consistent across batches.
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        cache_dir=cfg.cache_dir,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return model, processor


# ---------------------------------------------------------------------------
# One full training run
# ---------------------------------------------------------------------------
def train_lora(cfg: LoRATrainingConfig) -> dict:
    """Train a LoRA adapter on Qwen2-VL-2B for VQA-RAD.

    Returns the merged metrics dict (also written to disk).
    """
    set_global_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Data -----------------------------------------------------------
    train_ds = VQARadDataset(
        split=cfg.train_split, cache_dir=cfg.cache_dir,
        max_examples=cfg.train_max_examples,
    )
    print(f"Train set: {len(train_ds)} examples "
          f"({train_ds.question_type_counts()})")

    # -- 2. Model + processor ---------------------------------------------
    print(f"\nLoading base model {cfg.model_id} "
          f"(4bit={cfg.load_in_4bit}, dora={cfg.use_dora}) ...")
    model, processor = _load_base_model(cfg)
    print_parameter_summary(model, "Base (frozen)")

    # -- 3. Apply LoRA -----------------------------------------------------
    model = apply_lora_to_qwen(model, cfg)
    enable_input_grads(model)  # required for gradient checkpointing + LoRA
    base_param_summary = print_parameter_summary(model, "After LoRA")

    # -- 4. DataLoader -----------------------------------------------------
    collator = QwenVLSFTCollator(processor=processor)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,    # Colab is happiest with 0 workers
    )

    # -- 5. Optimizer + scheduler -----------------------------------------
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params, lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = (
        len(train_loader) * cfg.num_epochs
    ) // cfg.gradient_accumulation_steps
    total_steps = max(total_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    # -- 6. Train ----------------------------------------------------------
    reset_peak_gpu_memory()
    model.train()

    train_start = time.time()
    epoch_times: List[float] = []
    step_losses: List[float] = []

    global_step = 0
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % cfg.logging_steps == 0:
                avg = running_loss / (step + 1)
                pbar.set_postfix(loss=f"{avg:.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")
                step_losses.append({"step": global_step, "loss": avg})

        # Final flush of accumulator at end of epoch
        if (step + 1) % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dt = time.time() - epoch_start
        epoch_times.append(dt)
        print(f"  → epoch {epoch+1} done in {dt:.1f}s, "
              f"avg loss = {running_loss / len(train_loader):.4f}")

    train_time = time.time() - train_start
    peak_mem = get_gpu_memory()

    # -- 7. Save adapter ---------------------------------------------------
    print(f"\nSaving adapter to {out_dir}")
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    # -- 8. Build metrics dict --------------------------------------------
    method_label = (
        "DoRA" if cfg.use_dora else
        ("QLoRA" if cfg.load_in_4bit else "LoRA")
    )
    metrics: dict = {
        "method": method_label,
        "config": asdict(cfg),
        "params": base_param_summary,
        "training": {
            "total_seconds": round(train_time, 1),
            "epoch_seconds": [round(t, 1) for t in epoch_times],
            "mean_epoch_seconds": round(sum(epoch_times) / len(epoch_times), 1),
            "n_epochs": cfg.num_epochs,
            "n_train_examples": len(train_ds),
            "global_steps": global_step,
            "loss_curve": step_losses,
        },
        "peak_gpu_memory_gb": round(peak_mem["peak_allocated_gb"], 3),
        "peak_gpu_reserved_gb": round(peak_mem["peak_reserved_gb"], 3),
    }

    # -- 9. Run evaluation on the test split ------------------------------
    if cfg.run_eval_after_training:
        print(f"\nEvaluating fine-tuned model on {cfg.eval_split} split ...")
        model.eval()
        eval_metrics = _evaluate_model(
            model, processor, cfg.eval_split,
            max_examples=cfg.eval_max_examples,
            cache_dir=cfg.cache_dir,
            max_new_tokens=cfg.eval_max_new_tokens,
            output_dir=out_dir,
            seed=cfg.seed,
        )
        metrics["evaluation"] = eval_metrics

    # -- 10. Save metrics --------------------------------------------------
    metrics_file = out_dir / "training_metrics.json"
    with metrics_file.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics → {metrics_file}")

    cleanup_gpu()
    return metrics


# ---------------------------------------------------------------------------
# Post-training evaluation (re-uses baseline metric machinery)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_model(
    model, processor, split: str,
    max_examples: Optional[int],
    cache_dir: Optional[str],
    max_new_tokens: int,
    output_dir: Path,
    seed: int,
) -> dict:
    """Evaluate on a VQA-RAD split using the same metrics as the baseline.

    This replicates the baseline's evaluation logic so that LoRA results
    are directly comparable. Predictions are saved alongside the adapter.
    """
    eval_ds = VQARadDataset(split=split, cache_dir=cache_dir, max_examples=max_examples)
    print(f"  Eval set: {len(eval_ds)} examples")

    predictions, references, qtypes, ids, questions = [], [], [], [], []
    for i in tqdm(range(len(eval_ds)), desc="LoRA evaluation"):
        ex = eval_ds[i]
        try:
            pred = generate_answer(
                model, processor, ex["image"], ex["question"],
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Eval failed on {ex['id']}: {e}")
            pred = ""
        predictions.append(pred)
        references.append(ex["answer"])
        qtypes.append(ex["qtype"])
        ids.append(ex["id"])
        questions.append(ex["question"])

    metrics = compute_all_metrics(predictions, references, qtypes)

    # CIs and per-example score export (same shape as baseline)
    correct = per_example_correct(predictions, references)
    f1 = _per_example_open_scores(predictions, references, qtypes, "f1")
    bleu = _per_example_open_scores(predictions, references, qtypes, "bleu1")
    rouge = _per_example_open_scores(predictions, references, qtypes, "rougeL")
    metrics["overall"]["ci95"] = bootstrap_ci(correct, seed=seed)
    closed_correct = [c for c, q in zip(correct, qtypes) if q == "closed"]
    if closed_correct:
        metrics["closed"]["ci95"] = bootstrap_ci(closed_correct, seed=seed)
    open_f1 = [s for s, q in zip(f1, qtypes) if q == "open"]
    if open_f1:
        metrics["open"]["ci95_f1"] = bootstrap_ci(open_f1, seed=seed)

    # Save predictions
    pred_file = output_dir / "lora_predictions.jsonl"
    with pred_file.open("w") as f:
        for i, p, r, q, qt in zip(ids, predictions, references, questions, qtypes):
            f.write(json.dumps({
                "id": i, "question": q, "qtype": qt,
                "reference": r, "prediction": p,
            }) + "\n")
    print(f"  Predictions → {pred_file}")

    scores_file = output_dir / "per_example_scores.json"
    with scores_file.open("w") as f:
        json.dump({
            "ids": ids, "qtypes": qtypes, "correct": correct,
            "bleu1": bleu, "rougeL": rouge, "f1": f1,
        }, f)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="LoRA fine-tuning on VQA-RAD")
    ap.add_argument("--config", default=None,
                    help="Path to a YAML config (default: built-in defaults)")
    ap.add_argument("--max_train", type=int, default=None,
                    help="Override train_max_examples (e.g. 50 for smoke test)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override num_epochs")
    ap.add_argument("--rank", type=int, default=None,
                    help="Override lora_r")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    if args.config:
        cfg = LoRATrainingConfig.from_yaml(args.config)
    else:
        cfg = LoRATrainingConfig()
    if args.max_train is not None: cfg.train_max_examples = args.max_train
    if args.epochs is not None:    cfg.num_epochs = args.epochs
    if args.rank is not None:      cfg.lora_r = args.rank
    if args.output_dir is not None: cfg.output_dir = args.output_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_lora(cfg)


if __name__ == "__main__":
    main()
