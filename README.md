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

## Tests

```
pytest tests/ -v
```

85 unit tests covering metrics, dataset preprocessing, statistical tests, and config parsing. CPU-only, runs in under 30 seconds. Also runs automatically via GitHub Actions on every push.

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
├── 04_qlora_dora_experiments.ipynb             <- Member 2: QLoRA / DoRA / Q-DoRA training
├── 05_cross_method_analysis_drive_paths.ipynb  <- Member 3: tables, paired tests, figures
├── 06_gradio_demo_drive_paths.ipynb            <- Member 3: interactive web demo
└── 07_target_module_ablation_runpod.ipynb      <- target-module ablation RunPod notebook
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

## Authors

- Qiujun Zhang (qz2579)
- Wanrong Dang (wd2423)
- Longkun Xu (lx2358)
