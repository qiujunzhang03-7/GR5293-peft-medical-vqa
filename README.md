# PEFT for Medical Multimodal VQA

Comparing three Parameter-Efficient Fine-Tuning methods (LoRA, QLoRA, DoRA) for medical visual question answering on VQA-RAD, using Qwen2-VL-2B-Instruct.

Course: STAT GR5293, Columbia University, Spring 2026.

## Overview

We fine-tune Qwen2-VL-2B-Instruct on VQA-RAD (1,797 train / 451 test) and compare three PEFT methods — **LoRA**, **QLoRA**, and **DoRA** — at three ranks each, plus a Q-DoRA combination and a target-module ablation (11 PEFT configurations in total).

**Headline result:** the best configuration (DoRA r=4) lifts Closed EM from 0.5657 (zero-shot) to 0.7610 and Overall EM from 0.3792 to 0.5455, training only 0.24% of parameters. LoRA r=4 is statistically tied with DoRA r=4 (CIs cross zero); QLoRA trades ~6 pp Closed EM for a 35% reduction in peak GPU memory. All gains over baseline are statistically significant (paired bootstrap CIs do not cross zero; McNemar p < 10⁻⁹ for LoRA / DoRA, p < 10⁻³ for QLoRA).

The full report (with cross-method paired tests, Pareto analysis, and a Venn breakdown of which examples each method fixes) is in `docs/REPORT.md`. Dataset details are in `docs/DATA_CARD.md`.

## Repository Structure

```
.
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
├── .github/workflows/tests.yml
├── configs/
│   ├── baseline_config.yaml
│   ├── lora_quick.yaml
│   ├── lora_rank4.yaml
│   ├── lora_rank8.yaml
│   └── lora_rank16.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_zeroshot.ipynb
│   └── 03_lora_experiments.ipynb
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── run_baseline.sh
│   └── run_lora_quick.sh
├── checkpoints/
│   ├── lora_quick/
│   ├── lora_rank4/
│   ├── lora_rank8/
│   └── lora_rank16/
├── results/
│   ├── baseline/
│   ├── statistical_tests/
│   ├── tables/
│   ├── figures/
│   └── error_analysis/
├── docs/
│   ├── REPORT.md
│   ├── DATA_CARD.md
│   └── HANDOFF.md
└── tests/
```

## Setup

We use Google Colab with a T4 GPU. To run:

```
!git clone https://github.com/qiujunzhang03-7/GR5293-peft-medical-vqa.git
%cd GR5293-peft-medical-vqa
%pip install -q -r requirements.txt
```

VQA-RAD is downloaded automatically on first use; no manual data setup is required.

## How to Reproduce

The recommended way is to run the three notebooks in order. Each notebook ends by writing its outputs to disk so the next stage can pick them up.

### Step 1 — Data exploration (optional, ~2 min)

Open `notebooks/01_data_exploration.ipynb` and run all cells. This produces split sizes, length distributions, and image statistics. Useful for sanity checking the dataset but not required for the main pipeline.

### Step 2 — Zero-shot baseline (~12 min on T4)

Open `notebooks/02_baseline_zeroshot.ipynb` and run all cells (or run the script `bash scripts/run_baseline.sh`).

Inputs: `configs/baseline_config.yaml`

Outputs:
- `results/baseline/metrics.json` — headline Closed EM / Open Token-F1 / Overall EM with 95% bootstrap CIs
- `results/baseline/predictions.jsonl` — per-example predictions on the 451-example test split
- `results/baseline/per_example_scores.json` — per-example correctness used for paired statistical tests

### Step 3 — LoRA fine-tuning (~3 hours per rank on T4)

Open `notebooks/03_lora_experiments.ipynb`. The notebook runs four configurations:

| Config file | Train size | Rank | Target modules | Wall-clock |
|---|---|---|---|---|
| `configs/lora_quick.yaml` | 200 | 8 | attention only | ~6 min |
| `configs/lora_rank4.yaml` | 1,797 | 4 | attention + FFN | ~3 hours |
| `configs/lora_rank8.yaml` | 1,797 | 8 | attention + FFN | ~3 hours |
| `configs/lora_rank16.yaml` | 1,797 | 16 | attention + FFN | ~3 hours |

For each run, outputs go to `checkpoints/lora_{quick,rank4,rank8,rank16}/`:
- `training_metrics.json` — trainable params, peak GPU memory, epoch time, eval metrics, loss curve
- `lora_predictions.jsonl` — per-example predictions on the 451-example test split
- `per_example_scores.json` — per-example correctness used for paired statistical tests
- `adapter_model.safetensors` — LoRA adapter weights (gitignored due to file size)

The quick run is a 6-minute pilot that validates the pipeline before launching the 3-hour ablations.

### Step 4 — Statistical tests

The third notebook also calls `src/evaluation/statistical_tests.py` to compare each LoRA run against the baseline using paired bootstrap CIs and McNemar's exact test.

Outputs:
- `results/statistical_tests/baseline_vs_lora_{quick,rank4,rank8,rank16}.json`

These are the numbers reported in `docs/REPORT.md` Table 2.

### Step 5 — Tables and figures

The same notebook regenerates summary artifacts:
- `results/tables/main_results.md` — Table 1 of the report
- `results/figures/rank_scaling.{pdf,png}` — accuracy vs. rank
- `results/figures/loss_curves.{pdf,png}` — training loss curves
- `results/figures/improvements_bar.{pdf,png}` — improvements over baseline
- `results/error_analysis/baseline_vs_lora_rank4.json` — per-example wins and losses for qualitative analysis

## Where to find each result

If you just want to read the numbers without running anything:

| What | Where |
|---|---|
| Baseline Closed EM / Open F1 / Overall EM | `results/baseline/metrics.json` |
| LoRA results per rank | `checkpoints/lora_*/training_metrics.json` |
| Paired bootstrap CIs and McNemar p-values | `results/statistical_tests/baseline_vs_lora_*.json` |
| Main results table | `results/tables/main_results.md` |
| Figures used in the report | `results/figures/` |
| Qualitative wins/losses | `results/error_analysis/baseline_vs_lora_rank4.json` |
| Full discussion | `docs/REPORT.md` |

## Performance and memory optimization

To fit Qwen2-VL-2B fine-tuning into a single T4 GPU (16 GB), the training pipeline uses several memory-saving and throughput optimizations:

- **fp16 mixed precision** for both forward and backward passes, halving activation memory versus fp32.
- **Gradient checkpointing** during training, recomputing activations on the backward pass to trade compute for memory. This single switch is what brings peak GPU usage from 14+ GB down to ~6.7–6.9 GB on the LoRA runs.
- **Pinned image preprocessing** (`min_pixels=256*28*28`, `max_pixels=768*28*28`) so Qwen2-VL's dynamic resolution does not silently inflate token counts and memory between training and evaluation.
- **Per-device batch size 1 + gradient accumulation 4–8** for an effective batch size of 8, the largest batch that fits alongside the 2B base model and LoRA adapters on T4.
- **4-bit NF4 quantization** of the base model in the QLoRA / Q-DoRA configurations, dropping peak GPU usage to ~4.3–4.5 GB at the cost of ~6 pp Closed EM.
- **Greedy decoding** (`do_sample=False`) and `max_new_tokens=64` for fast, fully reproducible inference; one full evaluation of the 451-example test split runs in ~12 minutes.

Training and inference profiling utilities live in `src/utils/profiling.py` and emit peak GPU memory, wall-clock time, and trainable-parameter count into every `training_metrics.json` automatically.

## Tests

```
pytest tests/ -v
```

85 tests covering metrics, dataset preprocessing, statistical tests, and config parsing. The suite mixes pure unit tests (e.g. `exact_match`, `token_f1`, `bootstrap_ci`) with integration-style tests (`LoRATrainingConfig` end-to-end YAML parsing and CLI override paths). All tests are CPU-only and finish in under 30 seconds. They also run automatically via GitHub Actions on every push (`.github/workflows/tests.yml`, Python 3.10 and 3.11), and the suite is green on the current `main` branch.

## QLoRA, DoRA, Q-DoRA, and Demo

The full set of PEFT experiments beyond Member 1's LoRA ablation — QLoRA, DoRA, Q-DoRA, the target-module ablation, the cross-method analysis notebook, and the Gradio demo — together with **all** trained adapter weights and per-experiment outputs, are too large to commit to GitHub. We provide the complete bundle on Google Drive.

### Google Drive

> **[Google Drive folder](https://drive.google.com/drive/folders/1Y_zvc7aqt1fogymPh5SxwpCKkREsGoTB)**

Drive contents checked from the shared folder: **190 files** are visible recursively in the current `5293 Final project` folder, organized under **23 subfolders**. The Drive connector used for verification does not expose an exact byte-size field, so this README avoids claiming a precise total size. In Colab, mount Google Drive with `drive.mount('/content/drive')`, then open or copy the project folder directly from Drive.

### What's in the bundle

```
5293 final project/
├── configs/                                    <- target-module ablation YAML configs
│   ├── lora_rank8_attn_only.yaml
│   ├── lora_rank8_ffn_only.yaml
│   └── lora_rank8_qv_only.yaml
├── checkpoints 3/                              <- LoRA / QLoRA / DoRA / Q-DoRA adapter folders
│   ├── lora_quick/
│   ├── lora_rank{4,8,16}/
│   ├── qlora_rank{4,8,16}/
│   ├── dora_rank{4,8,16}/
│   └── qdora_rank8/
├── target_module_ablation_checkpoints/         <- attn-only / qv-only / FFN-only r=8 ablation outputs
│   ├── lora_rank8_attn_only/
│   ├── lora_rank8_ffn_only/
│   └── lora_rank8_qv_only/
├── results 5/                                  <- cross-method tables, figures, baseline, and tests
│   ├── baseline/
│   ├── error_analysis/
│   ├── figures/
│   ├── statistical_tests/
│   └── tables/
├── 04_qlora_dora_experiments.ipynb             <- Wanrong Dang: QLoRA / DoRA / Q-DoRA training
├── 05_cross_method_analysis_drive_paths.ipynb  <- LingKun Xu: tables, paired tests, figures
├── 06_gradio_demo_drive_paths.ipynb            <- Wanrong Dang: interactive web demo
└── 07_target_module_ablation_runpod.ipynb      <- Wanrong Dang: target-module ablation RunPod notebook
```

### Running the demo

After opening or copying the Google Drive folder in Colab, mount or `cd` into the project folder, open `06_gradio_demo_drive_paths.ipynb` in Colab (T4 runtime), and run all cells. The notebook:

1. Loads Qwen2-VL-2B-Instruct in fp16
2. Attaches the **DoRA r=4** adapter (best Overall EM in Table 1) by default — switch via the `ADAPTER_PATH` variable to demo any other checkpoint
3. Wires `generate_answer()` into a Gradio interface with 6 curated radiology examples
4. Launches with `share=True` for a public URL suitable for the final-presentation video

### Reproducing the QLoRA / DoRA training runs

Each PEFT method maps to a YAML config in the bundle. To reproduce, e.g., QLoRA r=8:

```
python -m src.training.train_lora \
    --config configs/qlora_rank8.yaml
```

The same entry point handles all methods; flipping `load_in_4bit: true` and/or `use_dora: true` in the YAML is the only difference between LoRA / QLoRA / DoRA / Q-DoRA. See `docs/REPORT.md` § 5.1 for the design rationale and § 5.2 for the full run matrix.

## Troubleshooting

Common issues encountered while running this project, and how to fix them. Most of these came up during our own runs.

### `pip install -r requirements.txt` hangs or fails on Colab

Colab pre-installs older versions of `torch`, `transformers`, and `peft`. When the pinned versions in `requirements.txt` conflict with what Colab already has, the install can stall or raise resolver errors.

**Fix:** add `-q` (quiet) and restart the runtime after install:

```
%pip install -q -r requirements.txt
# Runtime → Restart session, then re-import
```

### `ModuleNotFoundError: No module named 'src'`

Notebooks expect the project root to be on `sys.path`. If you opened a notebook directly without cloning, the imports fail.

**Fix:** make sure you ran the clone cell at the top of the notebook:

```
!git clone https://github.com/qiujunzhang03-7/GR5293-peft-medical-vqa.git
%cd GR5293-peft-medical-vqa
```

### `CUDA out of memory` during training

The full LoRA pipeline peaks around 6.7–6.9 GB on a Colab T4 (16 GB). If you see OOM at the same configuration, something else is using GPU memory.

**Fix:**

1. Restart the Colab runtime (Runtime → Restart session) to clear residual memory
2. Confirm the runtime is T4 (Runtime → Change runtime type → T4 GPU); some shared CPU runtimes silently fall back to no GPU
3. If you customized `per_device_train_batch_size` above 1, lower it back to the default 1 with `gradient_accumulation_steps=4`
4. For QLoRA / Q-DoRA configurations, peak usage is ~4.3–4.5 GB and should never OOM on T4

### `CUDA out of memory` during inference

Inference uses less memory than training, but still needs the full base model loaded. The most common cause is leftover memory from a previous training cell.

**Fix:** restart the runtime before running the baseline.

### `Image token count mismatch` error during training or inference

Qwen2-VL uses dynamic image resolution, and the processor expands images into a variable number of image tokens. If `min_pixels` and `max_pixels` are not pinned, the same image can produce different token counts at training versus inference, which breaks the loss / generation pipeline.

**Fix:** the project always uses `min_pixels=256*28*28, max_pixels=768*28*28`, set in `src/training/train_lora.py` and `src/evaluation/evaluate_baseline.py`. Don't override these — every checkpoint we ship was trained with these exact values.

### HuggingFace dataset download fails or hangs

VQA-RAD is loaded from `flaviagiammarino/vqa-rad`. If the download hangs, it's almost always a network or cache issue.

**Fix:**

1. Try again — transient network errors are common
2. Set the cache directory explicitly: `export HF_HOME=/content/hf_cache` before running
3. If you are behind a strict firewall, the dataset can be downloaded directly from https://huggingface.co/datasets/flaviagiammarino/vqa-rad and passed as a local path

### `pytest` collects fewer than 85 tests

The full suite is 85 tests on Python 3.10 / 3.11. If you see fewer, you are most likely missing `pyyaml` or `scipy` (the config-parsing and statistical tests skip without them).

**Fix:** install the full test dependencies:

```
pip install -r requirements.txt
pip install pytest scipy pyyaml
pytest tests/ -v
```

### Numbers slightly different from the report

Even with the same seed, CUDA non-determinism can shift Closed EM and Overall EM by ~1–2 percentage points across reruns. The headline pattern (DoRA ≈ LoRA > QLoRA, rank 4 > rank 16) is stable, but exact decimals are not.

**Fix:** none needed — this is expected. For an exact match to the numbers in `docs/REPORT.md`, use the `training_metrics.json` and `per_example_scores.json` files committed in `checkpoints/lora_*/`.

## Authors

- Qiujun Zhang (qz2579)
- Wanrong Dang (wd2423)
- Longkun Xu (lx2358)
