# PEFT for Medical Multimodal VQA

[![tests](https://github.com/USER/peft-medical-vqa/actions/workflows/tests.yml/badge.svg)](https://github.com/USER/peft-medical-vqa/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/code-MIT-green)](LICENSE)

A systematic comparison of three Parameter-Efficient Fine-Tuning (PEFT)
methods — **LoRA**, **QLoRA**, and **DoRA** — for medical visual question
answering on **VQA-RAD**, using **Qwen2-VL-2B-Instruct** as the base
vision-language model.

> **Course:** STAT GR5293 — Generative AI (Spring 2026), Columbia University
> **Project type:** Final project, three-person team

---

## Table of contents

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

## Project status

| Phase | Owner | Status |
|-------|-------|--------|
| Weeks 1–2: Data pipeline, environment, repo scaffolding | Member 1 | ✅ Done |
| Weeks 3–4: Zero-shot baseline + LoRA quick-run + statistical-test infrastructure | Member 1 | ✅ Done |
| Weeks 5–6: LoRA full-data + QLoRA experiments + hp tuning | Member 2 | ⏳ Upcoming |
| Weeks 5–8: DoRA experiments + Gradio demo | Member 3 | ⏳ Upcoming |
| Weeks 7–8: Hyperparameter / rank ablation | Member 2 + 3 | ⏳ Upcoming |
| Weeks 9–10: Result analysis & visualization | All | ⏳ Upcoming |
| Weeks 11–12: Demo + final report | All | ⏳ Upcoming |

---

## Research questions

- **RQ1.** How much accuracy improvement do LoRA, QLoRA, and DoRA provide
  over zero-shot Qwen2-VL-2B-Instruct on VQA-RAD?
- **RQ2.** How do the three methods differ in trainable parameter count,
  GPU memory, and training time?
- **RQ3.** How do hyperparameters (LoRA rank, quantization precision,
  target modules) trade off performance against efficiency?

For full motivation see [`docs/literature_review/`](docs/literature_review/)
and the original [proposal](docs/proposal.pdf).

---

## Repository structure

```
peft-medical-vqa/
├── README.md                        ← this file
├── requirements.txt                 ← pinned dependencies (Colab-compatible)
├── LICENSE                          ← MIT
├── CONTRIBUTING.md                  ← branch / commit conventions for the team
├── .github/workflows/tests.yml      ← CI: pytest on every push/PR
├── .gitignore
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    ← VQA-RAD statistics & visualization
│   ├── 02_baseline_zeroshot.ipynb   ← Zero-shot baseline ★
│   └── 03_lora_quick_run.ipynb      ← LoRA fine-tuning quick run ★
│
├── src/
│   ├── data/
│   │   ├── load_vqarad.py           ← HuggingFace loader + qtype classifier
│   │   └── vqarad_dataset.py        ← PyTorch Dataset + Qwen2-VL prompt builder
│   ├── evaluation/
│   │   ├── metrics.py               ← Exact Match, BLEU-1, ROUGE-L, Token-F1
│   │   ├── statistical_tests.py     ← Bootstrap CI, paired bootstrap, McNemar
│   │   └── evaluate_baseline.py     ← End-to-end zero-shot evaluation
│   ├── training/
│   │   ├── train_lora.py            ← LoRA / QLoRA / DoRA training pipeline
│   │   └── data_collator.py         ← Qwen2-VL SFT collator (prompt-mask labels)
│   └── utils/
│       ├── seed.py                  ← Reproducibility helpers
│       └── profiling.py             ← GPU memory / timing / param count utilities
│
├── tests/                           ← pytest unit tests (CPU-only, no model)
│   ├── test_metrics.py              ← 30+ tests on metric correctness
│   ├── test_statistical_tests.py    ← 16 tests on stat-test correctness
│   ├── test_data.py                 ← question-type classifier tests
│   ├── test_profiling.py            ← profiling utility tests
│   └── test_training_config.py      ← LoRA / QLoRA / DoRA config switching tests
│
├── scripts/
│   ├── run_baseline.sh              ← bash wrapper for the baseline CLI
│   └── run_lora_quick.sh            ← bash wrapper for the LoRA quick run
│
├── configs/
│   ├── baseline_config.yaml         ← all baseline hyperparameters
│   ├── lora_quick.yaml              ← LoRA quick run (Member 1's Week-4 deliverable)
│   └── lora_full.yaml               ← LoRA full-data config (Member 2 starting point)
│
├── results/baseline/                ← populated by running the baseline
│   ├── metrics.json                 ← aggregated metrics + 95% CIs
│   ├── predictions.jsonl            ← per-example predictions
│   └── per_example_scores.json      ← per-example scores for paired tests
│
├── checkpoints/lora_quick/          ← populated by the LoRA quick run
│   ├── adapter_model.safetensors    ← LoRA weights (~10 MB)
│   ├── adapter_config.json
│   ├── training_metrics.json        ← params, GPU mem, epoch time, eval results
│   ├── lora_predictions.jsonl
│   └── per_example_scores.json
│
└── docs/
    ├── DATA_CARD.md                 ← VQA-RAD provenance, license, statistics
    ├── MEMBER1_REPORT.md            ← Member 1 weeks 1-4 work writeup
    ├── EXTENDING_TO_QLORA_DORA.md   ← hand-off doc for Members 2 & 3
    └── literature_review/           ← reading notes on 6 foundational papers
        ├── lora_hu2021.md
        ├── qlora_dettmers2023.md
        ├── dora_liu2024.md
        ├── vqarad_lau2018.md
        ├── llavamed_li2023.md
        └── qwen2vl_wang2024.md
```

---

## Quick start (Google Colab)

The fastest way to reproduce the zero-shot baseline.

### 1. Open Colab and select GPU

Go to <https://colab.research.google.com>, create a new notebook, then:

> **Runtime → Change runtime type → T4 GPU → Save**

Free tier is sufficient. You don't need Colab Pro.

### 2. Get the project into Colab

Two options:

**Option A — clone from GitHub** (recommended once the repo is public):

```python
!git clone https://github.com/USER/peft-medical-vqa.git
%cd peft-medical-vqa
```

**Option B — upload zip**: Use the file panel (folder icon, left sidebar)
to upload `peft-medical-vqa.zip`, then unzip in a code cell:

```python
!unzip -q peft-medical-vqa.zip -d /content/
%cd /content/peft-medical-vqa
```

### 3. Open the baseline notebook

In the Colab file panel, navigate to
`peft-medical-vqa/notebooks/02_baseline_zeroshot.ipynb` and double-click.

### 4. Run all cells

> **Runtime → Run all**

Expected timeline:
- Dependency install: ~2 minutes
- Model download (Qwen2-VL-2B, ~4 GB): ~3 minutes
- Smoke test (5 examples): ~2 minutes
- **Full evaluation (451 examples): ~15–25 minutes**
- Bootstrap CIs + statistical tests: ~10 seconds

Total: ~25 minutes end-to-end.

### 5. Outputs

The baseline notebook writes three files to `results/baseline/`:

| File | What it contains |
|------|------------------|
| `metrics.json` | Aggregated metrics with 95% bootstrap CIs |
| `predictions.jsonl` | Per-example predictions (one JSON object per line) |
| `per_example_scores.json` | Per-example correctness/F1 vectors for paired statistical tests later |

### 6. Run the LoRA quick run (Week 4 deliverable)

After the baseline finishes, open `notebooks/03_lora_quick_run.ipynb` and `Run all`. This:

* Trains a LoRA adapter on a 200-example subset of VQA-RAD (1 epoch, rank 8) — ~10–15 min on T4
* Evaluates the adapted model on the full 451-example test split — ~15–20 min
* Computes a paired bootstrap CI and a McNemar exact test against the baseline
* Saves the adapter and a comparison table to `checkpoints/lora_quick/`

Total ~30–40 minutes including model load. This validates the entire training pipeline end-to-end before Members 2 & 3 commit to multi-hour full runs.

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

# 5. Run the LoRA quick run
bash scripts/run_lora_quick.sh
# Or with overrides:
python -m src.training.train_lora --config configs/lora_quick.yaml --rank 16

# 6. Smoke tests
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

The test suite verifies the metric and statistical-test implementations.
**It does not require a GPU or network** — all tests are pure-Python:

```bash
pip install pytest numpy rouge-score
pytest tests/ -v
```

Expected output: `63 passed in <60s`. Tests also run automatically on
every push via [GitHub Actions](.github/workflows/tests.yml) on Python
3.10 and 3.11.

---

## Reproducibility

We take reproducibility seriously because the project's RQ1/RQ2/RQ3
comparisons require apples-to-apples measurement across methods.

| Mechanism | Where |
|-----------|-------|
| Pinned dependency versions | [`requirements.txt`](requirements.txt) |
| Global seed (Python / NumPy / PyTorch) | [`src/utils/seed.py`](src/utils/seed.py); applied in every notebook |
| Greedy decoding (no sampling at inference) | [`src/evaluation/evaluate_baseline.py`](src/evaluation/evaluate_baseline.py) |
| Per-example outputs saved | `results/baseline/predictions.jsonl`, `per_example_scores.json` |
| Bootstrap CIs and McNemar tests | [`src/evaluation/statistical_tests.py`](src/evaluation/statistical_tests.py) |
| Continuous integration | [`.github/workflows/tests.yml`](.github/workflows/tests.yml) |
| Configuration as code | [`configs/baseline_config.yaml`](configs/baseline_config.yaml) |

Note on hardware non-determinism: **inference results are deterministic** under
greedy decoding with a fixed seed. **Training results** (Members 2 and 3)
will have low residual variance from CUDA non-determinism; we document this
in the final report and use multiple seeds where it matters.

---

## Team and responsibilities

| Member | UNI | Weeks | Owns |
|--------|-----|-------|------|
| Longkun Xu | lx2358 | 1–4 | Data pipeline, baseline (zero-shot) eval, statistical-test infrastructure, **LoRA quick-run pipeline**, repo scaffolding, literature review |
| Qiujun Zhang | qz2579 | 5–6 | LoRA full-data run, QLoRA experiments, hyperparameter tuning |
| Wanrong Dang | wd2423 | 5–8 | DoRA experiments, ablation studies, Gradio demo |

All three contribute to the final report and presentation.

---

## Citation

If you use this code or build on this work, please cite the underlying
papers:

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
