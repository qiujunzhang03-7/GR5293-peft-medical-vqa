"""End-to-end zero-shot baseline evaluation: Qwen2-VL-2B on VQA-RAD.

This module is callable both as a script and as a function. The notebook
(``notebooks/02_baseline_zeroshot.ipynb``) imports ``run_baseline``.

Outputs (in ``output_dir``)
---------------------------
* ``predictions.jsonl`` — per-example predictions for paired error analysis.
* ``metrics.json`` — aggregated metrics + 95% bootstrap CIs.
* ``per_example_scores.json`` — per-example correctness/F1, used by
  Members 2 and 3 for paired comparisons against PEFT methods.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.vqarad_dataset import VQARadDataset, build_qwen_prompt
from src.evaluation.metrics import (
    _bleu1_pair,
    _f1_pair,
    _rouge_l_pair,
    compute_all_metrics,
    normalize_text,
    per_example_correct,
)
from src.evaluation.statistical_tests import bootstrap_ci

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_OUTPUT_DIR = "results/baseline"


# ---------------------------------------------------------------------------
# Model loading — kept separate so PEFT experiments can reuse load/cleanup
# ---------------------------------------------------------------------------
def load_qwen_vl(
    model_id: str = DEFAULT_MODEL_ID,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
):
    """Load Qwen2-VL-2B-Instruct in inference mode.

    Returns
    -------
    (model, processor)
    """
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    logger.info(f"Loading {model_id} (dtype={dtype}, device_map={device_map})")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )
    return model, processor


# ---------------------------------------------------------------------------
# Single-example inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_answer(
    model,
    processor,
    image,
    question: str,
    max_new_tokens: int = 64,
) -> str:
    """Run a single zero-shot inference call.

    ``max_new_tokens=64`` is enough: 99% of VQA-RAD answers are under
    10 tokens. Larger values just waste compute.
    """
    messages = build_qwen_prompt(image, question)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        generated_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return answer.strip()


# ---------------------------------------------------------------------------
# Per-example score helpers (for statistical tests)
# ---------------------------------------------------------------------------
def _per_example_open_scores(
    predictions, references, qtypes, metric_name: str
):
    """Per-example open-ended scores; returns 0 for closed examples.

    This way the per-example score lists across methods are aligned by
    index, regardless of question type.
    """
    out = []
    for p, r, q in zip(predictions, references, qtypes):
        if q != "open":
            out.append(0.0)
            continue
        if metric_name == "bleu1":
            out.append(_bleu1_pair(p, r))
        elif metric_name == "rougeL":
            out.append(_rouge_l_pair(p, r))
        elif metric_name == "f1":
            out.append(_f1_pair(p, r))
        else:
            raise ValueError(f"Unknown metric {metric_name}")
    return out


# ---------------------------------------------------------------------------
# Full-split evaluation
# ---------------------------------------------------------------------------
def run_baseline(
    model_id: str = DEFAULT_MODEL_ID,
    split: str = "test",
    max_examples: Optional[int] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cache_dir: Optional[str] = None,
    save_predictions: bool = True,
    bootstrap_resamples: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run zero-shot evaluation on VQA-RAD and save results.

    See module docstring for outputs.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data ----
    dataset = VQARadDataset(
        split=split, cache_dir=cache_dir, max_examples=max_examples
    )
    if verbose:
        print(f"Loaded VQA-RAD {split}: {len(dataset)} examples")
        print(f"Question types: {dataset.question_type_counts()}")

    # ---- 2. Load model ----
    if verbose:
        print(f"\nLoading model {model_id} ...")
    t0 = time.time()
    model, processor = load_qwen_vl(model_id)
    if verbose:
        print(f"Model loaded in {time.time() - t0:.1f}s")
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory after load: {mem_gb:.2f} GB")

    # Reset peak-memory counter so we can report it cleanly later
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- 3. Inference loop ----
    predictions, references, qtypes, ids, questions = [], [], [], [], []
    inference_start = time.time()
    iterator = range(len(dataset))
    if verbose:
        iterator = tqdm(iterator, desc="Zero-shot inference")
    for i in iterator:
        ex = dataset[i]
        try:
            pred = generate_answer(model, processor, ex["image"], ex["question"])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Inference failed on example {ex['id']}: {e}")
            pred = ""
        predictions.append(pred)
        references.append(ex["answer"])
        qtypes.append(ex["qtype"])
        ids.append(ex["id"])
        questions.append(ex["question"])
    inference_time = time.time() - inference_start
    if verbose:
        print(f"\nInference complete in {inference_time:.1f}s "
              f"({inference_time / max(len(dataset), 1):.2f}s/example)")

    # ---- 4. Compute metrics ----
    metrics = compute_all_metrics(predictions, references, qtypes)

    # ---- 5. Per-example scores (for paired statistical tests later) ----
    correct_overall = per_example_correct(predictions, references)
    correct_closed = [
        c for c, q in zip(correct_overall, qtypes) if q == "closed"
    ]
    f1_per_example = _per_example_open_scores(
        predictions, references, qtypes, "f1"
    )
    bleu_per_example = _per_example_open_scores(
        predictions, references, qtypes, "bleu1"
    )
    rouge_per_example = _per_example_open_scores(
        predictions, references, qtypes, "rougeL"
    )

    # ---- 6. Bootstrap 95% CIs ----
    metrics["overall"]["ci95"] = bootstrap_ci(
        correct_overall, n_resamples=bootstrap_resamples, seed=seed
    )
    if correct_closed:
        metrics["closed"]["ci95"] = bootstrap_ci(
            correct_closed, n_resamples=bootstrap_resamples, seed=seed
        )
    open_only_f1 = [
        s for s, q in zip(f1_per_example, qtypes) if q == "open"
    ]
    if open_only_f1:
        metrics["open"]["ci95_f1"] = bootstrap_ci(
            open_only_f1, n_resamples=bootstrap_resamples, seed=seed
        )

    # ---- 7. Meta ----
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1e9
        if torch.cuda.is_available() else 0.0
    )
    metrics["meta"] = {
        "model_id": model_id,
        "split": split,
        "n_examples": len(dataset),
        "inference_seconds": round(inference_time, 1),
        "device": str(model.device) if hasattr(model, "device") else "unknown",
        "peak_gpu_memory_gb": round(peak_gb, 3),
        "trainable_params": 0,
        "seed": seed,
    }

    # ---- 8. Save outputs ----
    metrics_file = out_path / "metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if save_predictions:
        pred_file = out_path / "predictions.jsonl"
        with pred_file.open("w", encoding="utf-8") as f:
            for i, p, r, q, qt in zip(
                ids, predictions, references, questions, qtypes
            ):
                f.write(json.dumps({
                    "id": i, "question": q, "qtype": qt,
                    "reference": r, "prediction": p,
                }, ensure_ascii=False) + "\n")
        if verbose:
            print(f"Saved per-example predictions to {pred_file}")

    # Per-example scores (for paired tests by Members 2/3)
    scores_file = out_path / "per_example_scores.json"
    with scores_file.open("w", encoding="utf-8") as f:
        json.dump({
            "ids": ids,
            "qtypes": qtypes,
            "correct": correct_overall,
            "bleu1": bleu_per_example,
            "rougeL": rouge_per_example,
            "f1": f1_per_example,
        }, f, ensure_ascii=False)

    if verbose:
        print(f"Saved aggregated metrics to {metrics_file}")
        print(f"Saved per-example scores to {scores_file}")
        _print_metrics_table(metrics)

    # ---- 9. Cleanup ----
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _print_metrics_table(metrics: dict) -> None:
    """Pretty-print metrics with CIs as a Markdown-style block."""
    c = metrics["closed"]
    o = metrics["open"]
    ov = metrics["overall"]
    print("\n" + "=" * 64)
    print("BASELINE RESULTS — Qwen2-VL-2B-Instruct (zero-shot)")
    print("=" * 64)
    print(f"\nClosed-ended (n={c['n']}):")
    print(f"  Exact Match : {c['exact_match']:.4f}", end="")
    if "ci95" in c:
        print(f"   [95% CI: {c['ci95']['lower']:.4f} – {c['ci95']['upper']:.4f}]")
    else:
        print()
    print(f"\nOpen-ended (n={o['n']}):")
    print(f"  BLEU-1      : {o['bleu1']:.4f}")
    print(f"  ROUGE-L     : {o['rougeL']:.4f}")
    print(f"  Token-F1    : {o['f1']:.4f}", end="")
    if "ci95_f1" in o:
        ci = o["ci95_f1"]
        print(f"   [95% CI: {ci['lower']:.4f} – {ci['upper']:.4f}]")
    else:
        print()
    print(f"\nOverall (n={ov['n']}):")
    print(f"  Exact Match : {ov['exact_match']:.4f}", end="")
    if "ci95" in ov:
        print(f"   [95% CI: {ov['ci95']['lower']:.4f} – {ov['ci95']['upper']:.4f}]")
    else:
        print()
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Zero-shot baseline on VQA-RAD"
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Truncate the split (for smoke tests)",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_baseline(
        model_id=args.model_id, split=args.split,
        max_examples=args.max_examples, output_dir=args.output_dir,
        cache_dir=args.cache_dir, seed=args.seed,
    )
