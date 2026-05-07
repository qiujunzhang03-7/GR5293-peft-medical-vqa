# PEFT for Medical Multimodal VQA

Comparing three Parameter-Efficient Fine-Tuning methods (LoRA, QLoRA, DoRA) for medical visual question answering on VQA-RAD, using Qwen2-VL-2B-Instruct.

Course: STAT GR5293, Columbia University, Spring 2026.

## Overview

We fine-tune Qwen2-VL-2B-Instruct on VQA-RAD (1,797 train / 451 test) and compare three PEFT methods. Headline result: LoRA r=4 lifts Closed-EM from 0.5657 (zero-shot) to 0.7570 and Overall-EM from 0.3792 to 0.5432, training only 0.21% of parameters. All gains are statistically significant (paired bootstrap CIs do not cross zero; McNemar p < 1e-6).

The full report is in `docs/REPORT.md`. Dataset details are in `docs/DATA_CARD.md`.

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
│   └── DATA_CARD.md
└── tests/
```

## Setup

We use Google Colab with a T4 GPU. To run:

```
!git clone https://github.com/qiujunzhang03-7/GR5293-peft-medical-vqa.git
%cd GR5293-peft-medical-vqa
%pip install -q -r requirements.txt
```

VQA-RAD is downloaded automatically from HuggingFace on first use; no manual data setup is required.

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

## QLoRA, DoRA, and Demo

The QLoRA and DoRA experiments, cross-method analysis notebook, Gradio demo notebook, and trained checkpoint weights are stored on Google Drive due to file size:

**[Google Drive folder](https://drive.google.com/drive/folders/1Y_zvc7aqt1fogymPh5SxwpCKkREsGoTB)**

The Drive folder contains:
- `checkpoints/` — QLoRA and DoRA adapter weights
- `target_module_ablation_checkpoints/` — additional target-module ablation experiments
- `results/` — cross-method comparison metrics and figures
- `05_cross_method_analysis.ipynb` — comparison across LoRA / QLoRA / DoRA
- `06_gradio_demo.ipynb` — interactive Gradio demo using the best adapter

To run the demo: mount the Drive folder in Colab, open `06_gradio_demo.ipynb`, and run all cells. The notebook launches a Gradio app with `share=True` for a public link.

## Authors

- Qiujun Zhang (qz2579)
- Wanrong Dang (wd2423)
- Longkun Xu (lx2358)
