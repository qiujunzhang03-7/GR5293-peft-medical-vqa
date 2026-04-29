# PEFT for Medical Multimodal VQA

[![tests](https://github.com/qiujunzhang03-7/GR5293-peft-medical-vqa/actions/workflows/tests.yml/badge.svg)](https://github.com/qiujunzhang03-7/GR5293-peft-medical-vqa/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/code-MIT-green)](LICENSE)

A systematic comparison of three Parameter-Efficient Fine-Tuning (PEFT) methods — **LoRA**, **QLoRA**, and **DoRA** — for medical visual question answering on **VQA-RAD**, using **Qwen2-VL-2B-Instruct** as the base vision-language model.

> **Course:** STAT GR5293 — Generative AI (Spring 2026), Columbia University
> **Project type:** Final project, three-person team

---

## Table of contents

- [Headline results](#headline-results)
- [Project status](#project-status)
- [Research questions](#research-questions)
- [Repository structure](#repository-structure)
- [Quick start (Google Colab)](#quick-start-google-colab)
- [Local setup](#local-setup)
- [Running the tests](#running-the-tests)
- [Reproducibility](#reproducibility)
- [Team and responsibilities](#team-and-responsibilities)
- [Citation](#citation)

---

## Headline results

LoRA fine-tuning of Qwen2-VL-2B on VQA-RAD's 1,797 training examples improves zero-shot accuracy by **+16.4 percentage points** (Overall EM, McNemar p < 10⁻⁶) using only 0.21% of the model's parameters. Detailed results across rank ablations (r ∈ {4, 8, 16}):

| Method | Trainable | Closed EM (n=251) | Open Token-F1 (n=200) | Overall EM (n=451) |
|---|---:|---:|---:|---:|
| Zero-shot baseline | 0 | 0.5657 [0.51, 0.63] | 0.2008 [0.15, 0.25] | 0.3792 [0.33, 0.43] |
| **LoRA r=4** (best) | **4.62M (0.21%)** | **0.7570** [0.70, 0.81] | **0.3561** [0.30, 0.42] | **0.5432** [0.50, 0.59] |
| LoRA r=8 | 9.23M (0.42%) | 0.7371 [0.68, 0.79] | 0.3223 [0.26, 0.38] | 0.5211 [0.48, 0.57] |
| LoRA r=16 | 18.46M (0.83%) | 0.7371 [0.68, 0.79] | 0.3561 [0.30, 0.42] | 0.5344 [0.49, 0.58] |

All gains over baseline are statistically significant (paired bootstrap CIs do not cross zero; McNemar p < 10⁻⁶). Full results with figures and qualitative error analysis are in [`docs/MEMBER1_REPORT.md`](docs/MEMBER1_REPORT.md) and [`results/`](results/).

**Key finding:** rank=4 outperforms rank=8 and rank=16 across all metrics despite using fewer parameters — strong evidence that this dataset/model combination overfits with higher capacity. See `results/figures/rank_scaling.pdf`.

---

## Project status

| Phase | Owner | Status |
|---|---|---|
| Weeks 1–2: Data pipeline, environment, repo scaffolding | Member 1 | ✅ Done |
| Weeks 3–4: Zero-shot baseline + statistical-test infrastructure | Member 1 | ✅ Done |
| Weeks 3–4: LoRA quick-run pipeline (200 examples) | Member 1 | ✅ Done |
| Weeks 3–4: LoRA full-data ablation (rank ∈ {4, 8, 16}) | Member 1 | ✅ Done |
| Weeks 5–8: QLoRA + DoRA experiments + hyperparameter tuning | Member 2 | ⏳ Upcoming |
| Weeks 9–10: Cross-method comparative analysis & visualization | Member 3 | ⏳ Upcoming |
| Weeks 11–12: Gradio demo + presentation + final report | Member 3 + all | ⏳ Upcoming |

---

## Research questions

- **RQ1.** How much accuracy improvement do LoRA, QLoRA, and DoRA provide over zero-shot Qwen2-VL-2B-Instruct on VQA-RAD?
- **RQ2.** How do the three methods differ in trainable parameter count, GPU memory, and training time?
- **RQ3.** How do hyperparameters (LoRA rank, quantization precision, target modules) trade off performance against efficiency?

For full motivation see [`docs/literature_review/`](docs/literature_review/) and the original proposal.

---

## Repository structure

```
peft-medical-vqa/
├── README.md                        <- this file
├── requirements.txt                 <- pinned dependencies (Colab-compatible)
├── LICENSE                          <- MIT
├── CONTRIBUTING.md                  <- branch / commit conventions for the team
├── .github/workflows/tests.yml      <- CI: pytest on every push/PR
├── .gitignore
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    <- VQA-RAD statistics & visualization
│   ├── 02_baseline_zeroshot.ipynb   <- Zero-shot baseline
│   └── 03_lora_experiments.ipynb    <- All 4 LoRA experiments (parameterized)
│
├── src/
│   ├── data/
│   │   ├── load_vqarad.py           <- HuggingFace loader + qtype classifier
│   │   └── vqarad_dataset.py        <- PyTorch Dataset + Qwen2-VL prompt builder
│   ├── evaluation/
│   │   ├── metrics.py               <- Exact Match, BLEU-1, ROUGE-L, Token-F1
│   │   ├── statistical_tests.py     <- Bootstrap CI, paired bootstrap, McNemar
│   │   └── evaluate_baseline.py     <- End-to-end zero-shot evaluation
│   ├── training/
│   │   ├── train_lora.py            <- LoRA / QLoRA / DoRA training pipeline
│   │   ├── data_collator.py         <- Qwen2-VL SFT collator (prompt-mask labels)
│   │   └── _enable_input_grads.py   <- Gradient-checkpointing + LoRA helper
│   └── utils/
│       ├── seed.py                  <- Reproducibility helpers
│       └── profiling.py             <- GPU memory / timing / param count utilities
│
├── tests/                           <- pytest unit tests (CPU-only, no model)
│   ├── test_metrics.py              <- Metric correctness tests
│   ├── test_statistical_tests.py    <- Statistical-test correctness tests
│   ├── test_data.py                 <- Question-type classifier tests
│   ├── test_profiling.py            <- Profiling utility tests
│   └── test_training_config.py      <- LoRA / QLoRA / DoRA config switching tests
│
├── scripts/
│   ├── run_baseline.sh              <- bash wrapper for the baseline CLI
│   └── run_lora_quick.sh            <- bash wrapper for the LoRA quick run
│
├── configs/
│   ├── baseline_config.yaml         <- baseline hyperparameters
│   ├── lora_quick.yaml              <- LoRA quick run (200 examples, attention-only pilot)
│   ├── lora_rank4.yaml              <- LoRA full-data ablation, rank=4 (best)
│   ├── lora_rank8.yaml              <- LoRA full-data ablation, rank=8
│   └── lora_rank16.yaml             <- LoRA full-data ablation, rank=16
│
├── results/
│   ├── baseline/                    <- Zero-shot baseline outputs
│   │   ├── metrics.json             <- aggregated metrics + 95% CIs
│   │   ├── predictions.jsonl        <- per-example predictions
│   │   └── per_example_scores.json  <- per-example scores for paired tests
│   ├── statistical_tests/           <- Paired bootstrap + McNemar (vs baseline)
│   │   ├── baseline_vs_lora_quick.json
│   │   ├── baseline_vs_lora_rank4.json
│   │   ├── baseline_vs_lora_rank8.json
│   │   └── baseline_vs_lora_rank16.json
│   ├── tables/main_results.md       <- Tables 1 & 2 (paper-ready)
│   ├── figures/                     <- Plots (PNG + PDF)
│   │   ├── loss_curves.{png,pdf}
│   │   ├── rank_scaling.{png,pdf}
│   │   └── improvements_bar.{png,pdf}
│   └── error_analysis/              <- Qualitative wins (closed + open)
│       └── baseline_vs_lora_rank4.json
│
├── checkpoints/                     <- LoRA adapter outputs (weights gitignored)
│   ├── lora_quick/                  <- training_metrics.json + predictions tracked
│   ├── lora_rank4/
│   ├── lora_rank8/
│   └── lora_rank16/
│
└── docs/
    ├── DATA_CARD.md                 <- VQA-RAD provenance, license, statistics
    ├── MEMBER1_REPORT.md            <- Member 1 weeks 1-4 detailed writeup
    ├── HANDOFF.md                  <- Member 2 / 3 onboarding manual (setup, schemas, gotchas, 16 sections)
    └── literature_review/           <- Reading notes on 6 foundational papers
        ├── lora_hu2021.md
        ├── qlora_dettmers2023.md
        ├── dora_liu2024.md
        ├── vqarad_lau2018.md
        ├── llavamed_li2023.md
        └── qwen2vl_wang2024.md
```

---

## Quick start (Google Colab)

The fastest way to reproduce all results.

### 1. Open Colab and select GPU

Go to <https://colab.research.google.com>, create a new notebook, then:

> **Runtime → Change runtime type → T4 GPU → Save**

The free tier suffices for the baseline. The full LoRA ablations (3 hours each) consume ~80 Colab Pro compute units total.

### 2. Get the project into Colab

```python
!git clone https://github.com/USER/peft-medical-vqa.git
%cd peft-medical-vqa
```

### 3. Reproduce the baseline (~25 min)

Open `notebooks/02_baseline_zeroshot.ipynb` and `Runtime → Run all`. Outputs go to `results/baseline/`.

### 4. Reproduce a LoRA experiment (~10 min for quick, ~3 hours for full)

Open `notebooks/03_lora_experiments.ipynb`. In Section 2, set:

```python
EXPERIMENT = 'rank4'   # one of: 'quick' / 'rank4' / 'rank8' / 'rank16'
```

then `Runtime → Run all`. Outputs go to `checkpoints/lora_{EXPERIMENT}/`.

| EXPERIMENT | Train size | Epochs | Rank | Wall time (T4) |
|---|---:|---:|---:|---:|
| `quick`  | 200  | 1 | 8  | ~10 min |
| `rank4`  | 1797 | 3 | 4  | ~3 hours |
| `rank8`  | 1797 | 3 | 8  | ~3 hours |
| `rank16` | 1797 | 3 | 16 | ~3 hours |

---

## Local setup

If you have your own GPU (≥10 GB VRAM recommended):

```bash
# 1. Clone
git clone https://github.com/USER/peft-medical-vqa.git
cd peft-medical-vqa

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate                # Linux/Mac
# .\venv\Scripts\activate              # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the baseline (full test split, default settings)
python -m src.evaluation.evaluate_baseline
# Or via the convenience wrapper:
bash scripts/run_baseline.sh

# 5. Run a LoRA experiment
python -m src.training.train_lora --config configs/lora_rank4.yaml
# Or with overrides:
python -m src.training.train_lora --config configs/lora_quick.yaml --rank 16

# 6. Smoke tests (cheap end-to-end check)
python -m src.evaluation.evaluate_baseline --max_examples 5 \
    --output_dir results/baseline_smoke
python -m src.training.train_lora --config configs/lora_quick.yaml \
    --max_train 20 --output_dir checkpoints/lora_smoke
```

CLI arguments for `train_lora`:
- `--config FILE`: YAML config file (default: built-in defaults)
- `--max_train N`: override `train_max_examples` (smoke testing)
- `--epochs N`: override `num_epochs`
- `--rank N`: override `lora_r`
- `--output_dir DIR`: override checkpoint directory

---

## Running the tests

The test suite verifies the metric and statistical-test implementations. **It does not require a GPU or network** — all tests are pure-Python:

```bash
pip install pytest numpy rouge-score
pytest tests/ -v
```

Expected output: `85 passed in <30s`. Tests also run automatically on every push via [GitHub Actions](.github/workflows/tests.yml) on Python 3.10 and 3.11.

---

## Reproducibility

We take reproducibility seriously because the project's RQ1/RQ2/RQ3 comparisons require apples-to-apples measurement across methods.

| Mechanism | Where |
|---|---|
| Pinned dependency versions | [`requirements.txt`](requirements.txt) |
| Global seed (Python / NumPy / PyTorch) | [`src/utils/seed.py`](src/utils/seed.py); applied in every notebook |
| Greedy decoding (no sampling at inference) | [`src/evaluation/evaluate_baseline.py`](src/evaluation/evaluate_baseline.py) |
| Identical preprocessor settings (min/max image pixels) for baseline + LoRA | [`src/training/train_lora.py`](src/training/train_lora.py) |
| Per-example outputs saved | `results/baseline/`, `checkpoints/lora_*/` |
| Bootstrap CIs and McNemar tests | [`src/evaluation/statistical_tests.py`](src/evaluation/statistical_tests.py) |
| Continuous integration | [`.github/workflows/tests.yml`](.github/workflows/tests.yml) |
| Configuration as code | [`configs/`](configs/) |

Note on hardware non-determinism: **inference results are deterministic** under greedy decoding with a fixed seed. **Training results** have low residual variance from CUDA non-determinism; we document this as a single-seed limitation in the final report.

---

## Team and responsibilities

| Member | Name | UNI | Weeks | Owns |
|---|---|---|---|---|
| 1 | Qiujun Zhang | qz2579 | 1–4 | Data pipeline, baseline (zero-shot) eval, statistical-test infrastructure, **complete LoRA experiments (quick + 3-rank ablation)**, repo scaffolding, literature review |
| 2 | Wanrong Dang | wd2423 | 5–8 | QLoRA experiments, DoRA experiments, hyperparameter tuning, weight-decomposition analysis |
| 3 | Longkun Xu | lx2358 | 9–12 | Cross-method comparative analysis, Gradio demo, presentation, final report |

All three contribute to the final report and presentation.

---

## Citation

If you use this code or build on this work, please cite the underlying papers:

```bibtex
@article{lau2018vqarad,
  title={A dataset of clinically generated visual questions and answers
         about radiology images},
  author={Lau, Jason J. and Gayen, Soumya and Ben Abacha, Asma and
          Demner-Fushman, Dina},
  journal={Scientific Data}, volume={5}, year={2018}
}
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and others}, booktitle={ICLR}, year={2022}
}
@inproceedings{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and others}, booktitle={NeurIPS}, year={2023}
}
@inproceedings{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and others}, booktitle={ICML}, year={2024}
}
@article{wang2024qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the
         World at Any Resolution},
  author={Wang, Peng and others},
  journal={arXiv:2409.12191}, year={2024}
}
```

## License

- Code: [MIT](LICENSE)
- VQA-RAD data: CC0 1.0 Universal (Lau et al., 2018)
- Qwen2-VL-2B-Instruct weights: Apache 2.0 (Alibaba)
