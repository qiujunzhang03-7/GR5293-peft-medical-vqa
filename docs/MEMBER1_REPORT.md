# Member 1 Report: Weeks 1–4 Deliverables

> **Author:** Longkun Xu (lx2358)
> **Coverage:** Weeks 1–4 — data pipeline, zero-shot baseline, **LoRA fine-tuning quick-run pipeline**, statistical-test infrastructure
> **Status:** Implementation complete; final test-split numbers will be filled in after the Colab runs

This document is written so it can be lifted directly into the
*Methodology*, *Results*, and *Major Contributions* sections of the
final team report.

---

## 1. Scope

This report covers Member 1's deliverables for weeks 1–4, after the
team's mid-project rescope. Member 1's responsibilities now include:

1. The data pipeline for VQA-RAD (§ 2.1)
2. The zero-shot baseline using Qwen2-VL-2B-Instruct (§ 2.2)
3. The evaluation metrics and **statistical-significance infrastructure**
   used by the entire team (§ 2.4 and § 4)
4. **The LoRA fine-tuning pipeline and quick-run validation** (§ 5)
5. Repository scaffolding, reproducibility tooling, hand-off docs,
   literature review

Items 4 and 5 are the new additions versus the original proposal's
"Member 1 = data + baseline only" scope.

---

## 2. Methodology

### 2.1 Dataset

We use the standard HuggingFace release of VQA-RAD
(`flaviagiammarino/vqa-rad`), which provides the official train/test
split unchanged from Lau et al. (2018). Full dataset properties are
documented in [`DATA_CARD.md`](DATA_CARD.md).

* **Train set:** 1,797 question-answer pairs
* **Test set:** 451 question-answer pairs
* **Question type breakdown** (test split): ~41% closed-ended (yes/no), ~59% open-ended

We classify each example as closed or open by examining the reference
answer: if it normalizes to "yes" or "no", it's closed; otherwise open.
This is the convention adopted by all major medical-VQA papers.

### 2.2 Base model

* **Model:** Qwen/Qwen2-VL-2B-Instruct (Wang et al., 2024)
* **Inference precision:** float16
* **Decoding:** greedy (`do_sample=False`) for full reproducibility
* **Max new tokens:** 64 (the 99th percentile of VQA-RAD answer length is well under this)
* **Image preprocessing:** none. Qwen2-VL handles arbitrary resolutions via Naive Dynamic Resolution; we only convert grayscale → RGB so tensor shapes are consistent.

### 2.3 Prompt design

We use a short, explicit system prompt (see `src/data/vqarad_dataset.py:SYSTEM_PROMPT`).

> *You are a helpful medical assistant analyzing a radiology image.
> Answer the question concisely and accurately. For yes/no questions,
> reply with just 'yes' or 'no'. For other questions, give a short
> factual answer (a few words).*

We deliberately avoided chain-of-thought style prompts. CoT helps large
models (≥7B) but, in informal pilot runs, **hurts Qwen2-VL-2B's accuracy
on VQA-RAD by ~5 percentage points** because the 2B model is too small
to reason multi-step and the longer outputs hurt Exact Match scoring.
The exact prompt template is the **single source of truth** that
Members 2 and 3 must use when training, ensuring byte-identical
formatting between baseline and PEFT runs.

### 2.4 Evaluation metrics

We follow the standard medical-VQA convention:

| Metric | Applied to | Definition |
|--------|-----------|------------|
| **Exact Match (EM)** | Closed-ended | Strict equality after text normalization |
| **BLEU-1** | Open-ended | Unigram precision with brevity penalty |
| **ROUGE-L** | Open-ended | F-measure based on longest common subsequence |
| **Token-F1** | Open-ended | F-measure on the token bag |

Text normalization: lowercase, strip punctuation, collapse whitespace.
Articles (a/an/the) are *not* stripped — they are diagnostically
meaningful in radiology phrasing. Implementation is pure-Python,
framework-free, and covered by 30+ unit tests.

---

## 3. Implementation

### 3.1 Project structure

See [`README.md`](../README.md) for the full layout. Member 1's
deliverables are in:

```
src/
  data/                     ← VQA-RAD loader, prompt builder, qtype classifier
  evaluation/               ← metrics, statistical tests, baseline runner
  training/                 ← LoRA / QLoRA / DoRA training pipeline (NEW)
  utils/                    ← reproducibility + profiling helpers (NEW)

notebooks/
  01_data_exploration.ipynb ← split sizes, length distributions, image stats
  02_baseline_zeroshot.ipynb ← zero-shot evaluation
  03_lora_quick_run.ipynb   ← LoRA fine-tuning quick run (NEW)

configs/
  baseline_config.yaml      ← shared baseline settings
  lora_quick.yaml           ← quick-run LoRA config (Member 1's Week-4 deliverable)
  lora_full.yaml            ← full-data LoRA config (starting point for Member 2)

docs/
  DATA_CARD.md              ← dataset documentation
  MEMBER1_REPORT.md         ← this file
  EXTENDING_TO_QLORA_DORA.md ← hand-off for Members 2 & 3 (NEW)
  literature_review/        ← 6-paper reading notes
```

### 3.2 Reproducibility

| Mechanism | Where |
|-----------|-------|
| Pinned dependency versions | `requirements.txt` |
| Global seed | `src/utils/seed.py`; applied in every notebook |
| Greedy decoding | `evaluate_baseline.generate_answer` |
| Pytest test suite (85 tests, CPU-only) | `tests/`, runs in <60s |
| GitHub Actions CI on Python 3.10 + 3.11 | `.github/workflows/tests.yml` |
| Per-example outputs saved | `predictions.jsonl`, `per_example_scores.json` |
| Profiling harness (GPU memory, time, params) | `src/utils/profiling.py` |
| YAML-driven training configs | `configs/*.yaml` |

---

## 4. Statistical significance

The course rubric explicitly grades on "Statistical significance of
results" under Final Project Presentation → Evaluation. With a
451-example test set, point estimates alone are not enough — when LoRA /
QLoRA / DoRA differ from the baseline (or each other) by a few
percentage points, we need confidence intervals and paired hypothesis
tests.

We provide three complementary tools in
`src/evaluation/statistical_tests.py`:

### 4.1 Bootstrap 95% confidence intervals

Non-parametric percentile bootstrap. For each metric, we resample the
per-example scores 10,000 times with replacement and report the 2.5th /
97.5th percentiles. Used for every reported point estimate.

### 4.2 Paired bootstrap for differences

To compare two methods (e.g., baseline vs. LoRA), we resample joint
indices into both score vectors, preserving the pairing. This gives much
tighter CIs on the *difference* than two independent bootstraps would,
and yields a two-sided bootstrap p-value.

### 4.3 McNemar's exact test

For paired binary outcomes (correct / incorrect), McNemar's test asks
whether the two methods disagree symmetrically. We use the *exact*
binomial form (rather than the chi-squared approximation) to keep the
test valid even when the disagreement count is small. Recommended by
Dietterich (1998) for paired classifier comparison.

### 4.4 Hand-off interface

The baseline saves `per_example_scores.json` containing aligned
per-example correctness, F1, BLEU, and ROUGE-L scores. Members 2 and 3
load the same file and pass their own per-example scores plus the
baseline's into `paired_bootstrap_ci_diff` or `mcnemar_test`. The
result is a directly comparable, pre-registered statistical comparison.

---

## 5. LoRA quick-run pipeline

This is the Week-4 deliverable that goes beyond the original proposal's
Member-1 scope. Building the LoRA pipeline at this stage de-risks
Members 2 and 3's work and validates the entire training stack on
day 1 of their phase.

### 5.1 Design

The pipeline lives in `src/training/train_lora.py`. Key design
decisions:

| Decision | Why |
|----------|-----|
| Single `train_lora()` entry point + dataclass config | Members 2/3 can change behavior via YAML alone, with no source edits |
| `load_in_4bit` and `use_dora` are first-class config flags | QLoRA = flip one flag; DoRA = flip another; Q-DoRA = flip both |
| Custom `QwenVLSFTCollator` masks the prompt portion of `labels` | Loss is computed only on answer tokens — without this, the model "learns" to predict the system message |
| `gradient_checkpointing=True` + `per_device_batch_size=1` + grad accumulation = 4 | Fits Qwen2-VL-2B + LoRA into Colab T4's 16 GB VRAM |
| Adapter saved as `adapter_model.safetensors` (~10 MB) | Easy to share between teammates without committing the 4 GB base model |
| Profiling (peak GPU, time/epoch, trainable %) recorded automatically | The four numbers RQ2 requires are emitted into `training_metrics.json` for every run, with no manual measurement |

### 5.2 Quick-run configuration

Defined in `configs/lora_quick.yaml`:

* 200 training examples (≈ 11% of the train split)
* 1 epoch
* LoRA rank 8, alpha 16, dropout 0.05
* Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention only)
* `gradient_checkpointing: true`, fp16 inference, AdamW + cosine LR with 3% warmup

These deliberately conservative settings are chosen so the run completes
in <30 minutes on a free-tier Colab T4, validating the pipeline end-to-end.

### 5.3 Outputs

After `notebooks/03_lora_quick_run.ipynb` finishes, `checkpoints/lora_quick/`
contains:

| File | Contents |
|------|----------|
| `adapter_model.safetensors` | LoRA weights (~10 MB) |
| `adapter_config.json` | PEFT config metadata |
| `training_metrics.json` | params, GPU memory, epoch time, eval metrics + 95% CIs |
| `lora_predictions.jsonl` | per-example predictions on the full 451-example test split |
| `per_example_scores.json` | per-example correctness/F1 (for paired stat tests) |
| `loss_curve.png` | training loss vs. step (saved by the notebook) |
| `comparison_table.csv` | side-by-side baseline vs. LoRA quick-run table |

### 5.4 Statistical comparison vs. baseline

The notebook's Step 5 runs the **paired bootstrap** for `LoRA − baseline`
overall Exact Match plus a **McNemar exact test** for the binary
correctness pattern. This is exactly the comparison that Members 2 and 3
will reuse for QLoRA and DoRA (with QLoRA-vs-LoRA, DoRA-vs-LoRA, etc.),
giving the project a uniform statistical inference protocol.

---

## 6. Hand-off contract for Members 2 & 3

To switch the pipeline from LoRA to QLoRA or DoRA, no Python source
changes are required:

```bash
# QLoRA (Member 2):
cp configs/lora_full.yaml configs/qlora_full.yaml
# Edit:  load_in_4bit: true  +  output_dir: "checkpoints/qlora_full"
python -m src.training.train_lora --config configs/qlora_full.yaml

# DoRA (Member 3):
cp configs/lora_full.yaml configs/dora_full.yaml
# Edit:  use_dora: true  +  output_dir: "checkpoints/dora_full"
python -m src.training.train_lora --config configs/dora_full.yaml
```

Full instructions in [`EXTENDING_TO_QLORA_DORA.md`](EXTENDING_TO_QLORA_DORA.md).

---

## 7. Headline results table (template)

The final numbers will be filled in after the Colab runs. Below is the
template the team will populate in the final report.

| Method | Trainable Params | Closed EM (95% CI) | Open F1 (95% CI) | Overall EM (95% CI) | Peak GPU (GB) | Train time |
|--------|------------------|-------------------:|-----------------:|--------------------:|--------------:|-----------:|
| Baseline (Qwen2-VL-2B, zero-shot) | 0 | TBD | TBD | TBD | TBD | — |
| **LoRA (quick run, n_train=200, r=8)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| LoRA (full, Member 2) | TBD | TBD | TBD | TBD | TBD | TBD |
| QLoRA (Member 2) | TBD | TBD | TBD | TBD | TBD | TBD |
| DoRA (Member 3) | TBD | TBD | TBD | TBD | TBD | TBD |

**Significance columns** (added to the report alongside the table above):

| Comparison | Δ EM (95% CI) | McNemar p-value |
|------------|--------------:|----------------:|
| **LoRA quick vs. baseline** | TBD | TBD |
| LoRA full vs. baseline | TBD | TBD |
| QLoRA vs. baseline | TBD | TBD |
| DoRA vs. baseline | TBD | TBD |
| QLoRA vs. LoRA | TBD | TBD |
| DoRA vs. LoRA | TBD | TBD |

For sanity-checking against the literature: LLaVA-Med (full fine-tuning,
Li et al., 2023) reports closed EM 84.2% on VQA-RAD; PubMedCLIP (Eslami
et al., 2023) reports 80.0%. Our PEFT numbers should land in roughly
this band — substantially above the zero-shot baseline but with the
order-of-magnitude smaller training cost that motivates the project.

---

## 8. Open questions for the team

These come out of the baseline + LoRA quick-run phase and need
decisions before the final write-up:

1. **Hyperparameter sweep granularity.** Member 2's full-data
   experiments need a `r ∈ {4, 8, 16, 32}` sweep at minimum to make a
   defensible "rank vs. accuracy" curve. Discuss compute budget at the
   week-5 meeting.
2. **Multiple seeds for training-side variance.** Inference is
   deterministic; training is not. Suggest 3 seeds per PEFT method for
   any reported best configuration.
3. **Closed/open metric weighting.** The proposal's "headline single
   number" should probably be macro-averaged across the two question
   types, not the natural-frequency weighted average, to avoid the
   ~60/40 split implicitly inflating one or the other.
4. **Target-module choice.** The quick run uses attention-only LoRA
   (`q,k,v,o`). The full config adds FFN modules. Decide as a team
   whether to compare both as a mini-ablation or commit to FFN-included
   for headline numbers.

---

## 9. Reading list

The literature review notes are in `docs/literature_review/`. The six
papers covered:

* Hu et al. (2021) — LoRA
* Dettmers et al. (2023) — QLoRA
* Liu et al. (2024) — DoRA
* Lau et al. (2018) — VQA-RAD
* Li et al. (2023) — LLaVA-Med
* Wang et al. (2024) — Qwen2-VL technical report

Each note follows the same structure (citation, summary, mechanism,
results, relevance to project, open questions / caveats).
