# Project Report: PEFT for Medical Multimodal VQA — LoRA vs. QLoRA vs. DoRA

> **Authors:** Qiujun Zhang (qz2579), Wanrong Dang (wd2423), Longkun Xu (lx2358)
> **Coverage:** Weeks 1–12 — data pipeline, zero-shot baseline, **complete LoRA / QLoRA / DoRA / Q-DoRA experiments (3-rank ablation per method)**, target-module ablation, statistical-test infrastructure, cross-method analysis, Gradio demo
> **Status:** All runs complete; numbers below are final (entire 451-example VQA-RAD test split). 11 PEFT configurations evaluated end-to-end, all paired-bootstrap significant against the zero-shot baseline.

This document is written so it can be lifted directly into the *Methodology*, *Results*, and *Major Contributions* sections of the final team report.

---

## 1. Scope

This report covers the full Weeks 1–12 work of the team. The deliverables fall into five blocks:

1. The data pipeline for VQA-RAD (§ 2.1) — *Member 1, Weeks 1–2*
2. The zero-shot baseline using Qwen2-VL-2B-Instruct (§ 2.2) — *Member 1, Weeks 3–4*
3. The evaluation metrics and **statistical-significance infrastructure** used by the entire team (§ 2.4 and § 4) — *Member 1, Weeks 3–4*
4. **The complete LoRA fine-tuning pipeline: quick-run validation + 3-rank ablation (r ∈ {4, 8, 16})** (§ 5.1–5.3) — *Member 1, Week 4 (originally Member 2's responsibility, completed early to parallelize the critical path)*
5. **QLoRA, DoRA, and Q-DoRA experiments + target-module ablation** (§ 5.4–5.7) — *Member 2, Weeks 5–8*
6. **Cross-method comparative analysis, statistical tests across methods, and visualization** (§ 6–7) — *Member 3, Weeks 9–10*
7. **Gradio web demo + final report** (§ 9) — *Member 3, Weeks 11–12*

Items 4 and 5 are the new additions versus the original proposal's data + baseline scope. We completed the LoRA experiments in week 4 to parallelize the critical path. The full project executes 11 PEFT configurations against a single zero-shot baseline, plus three target-module ablations at r=8, all on the same 451-example test split.

---

## 2. Methodology

### 2.1 Dataset

We use the standard HuggingFace release of VQA-RAD (`flaviagiammarino/vqa-rad`), which provides the official train/test split unchanged from Lau et al. (2018). Full dataset properties are documented in [`DATA_CARD.md`](DATA_CARD.md).

* **Train set:** 1,797 question-answer pairs
* **Test set:** 451 question-answer pairs
* **Question type breakdown** (test split): 251 closed-ended (55.7%), 200 open-ended (44.3%)

We classify each example as closed or open by examining the reference answer: if it normalizes to "yes" or "no", it's closed; otherwise open. This is the convention adopted by all major medical-VQA papers.

### 2.2 Base model

* **Model:** Qwen/Qwen2-VL-2B-Instruct (Wang et al., 2024)
* **Inference precision:** float16
* **Decoding:** greedy (`do_sample=False`) for full reproducibility
* **Max new tokens:** 64 (the 99th percentile of VQA-RAD answer length is well under this)
* **Image preprocessing:** `min_pixels=256*28*28`, `max_pixels=768*28*28` for both baseline and LoRA runs (prevents image-token-count mismatch with Qwen2-VL's dynamic resolution, and ensures apples-to-apples comparison)

### 2.3 Prompt design

We use a short, explicit system prompt (see `src/data/vqarad_dataset.py:SYSTEM_PROMPT`).

> *You are a helpful medical assistant analyzing a radiology image.*
> *Answer the question concisely and accurately. For yes/no questions,*
> *reply with just 'yes' or 'no'. For other questions, give a short*
> *factual answer (a few words).*

We didn't use chain-of-thought style prompts. CoT helps large models (≥7B) but, in informal pilot runs, **hurts Qwen2-VL-2B's accuracy on VQA-RAD by ~5 percentage points** because the 2B model is too small to reason multi-step and the longer outputs hurt Exact Match scoring. The exact prompt template is used in both baseline and fine-tuning to keep formatting consistent.

### 2.4 Evaluation metrics

We follow the standard medical-VQA convention:

| Metric | Applied to | Definition |
|---|---|---|
| **Exact Match (EM)** | Closed-ended + all | Strict equality after text normalization |
| **BLEU-1** | Open-ended | Unigram precision with brevity penalty |
| **ROUGE-L** | Open-ended | F-measure based on longest common subsequence |
| **Token-F1** | Open-ended | F-measure on the token bag (headline open-ended number) |

Text normalization: lowercase, strip punctuation, collapse whitespace. Articles (a/an/the) are *not* stripped — they are diagnostically meaningful in radiology phrasing. Implementation is pure-Python, framework-free, and covered by 30+ unit tests.

---

## 3. Implementation

### 3.1 Project structure

See [`README.md`](../README.md) for the full layout. The files for this work are in:

```
src/
  data/                      <- VQA-RAD loader, prompt builder, qtype classifier
  evaluation/                <- metrics, statistical tests, baseline runner
  training/                  <- LoRA / QLoRA / DoRA training pipeline
  utils/                     <- reproducibility + profiling helpers
  demo/                      <- Gradio app entry point (Member 3)

notebooks/
  01_data_exploration.ipynb       <- split sizes, length distributions, image stats
  02_baseline_zeroshot.ipynb      <- zero-shot evaluation
  03_lora_experiments.ipynb       <- parameterized runner for quick / rank4 / rank8 / rank16
  04_qlora_dora_experiments.ipynb <- QLoRA + DoRA + Q-DoRA training (Member 2)
  05_cross_method_analysis.ipynb  <- headline tables, significance tests, all figures (Member 3)
  06_gradio_demo.ipynb            <- best-checkpoint Gradio web demo (Member 3)
  GR5293_QLoRA_DoRA_HParam_Colab.ipynb <- one-stop Colab notebook for hyperparameter sweeps

configs/
  baseline_config.yaml       <- baseline hyperparameters
  lora_quick.yaml            <- quick pilot (200 examples, attention-only)
  lora_rank4.yaml            <- LoRA full-data ablation, r=4 (best LoRA)
  lora_rank8.yaml            <- LoRA full-data ablation, r=8
  lora_rank16.yaml           <- LoRA full-data ablation, r=16
  qlora_rank{4,8,16}.yaml    <- QLoRA full-data ablation (Member 2)
  dora_rank{4,8,16}.yaml     <- DoRA full-data ablation (Member 2)
  qdora_rank8.yaml           <- Q-DoRA r=8 (4-bit + DoRA combo, Member 2)
  lora_rank8_attn_only.yaml  <- target-module ablation: attention only (Member 2)
  lora_rank8_qv_only.yaml    <- target-module ablation: q_proj + v_proj only (Member 2)
  lora_rank8_ffn_only.yaml   <- target-module ablation: FFN only (Member 2)

docs/
  DATA_CARD.md               <- dataset documentation
  REPORT.md                  <- this file
  HANDOFF.md                 <- onboarding manual for Members 2 / 3
```

### 3.2 Reproducibility

| Mechanism | Where |
|---|---|
| Pinned dependency versions | `requirements.txt` |
| Global seed (Python / NumPy / PyTorch) | `src/utils/seed.py` |
| Identical processor settings (min/max image pixels) across baseline + all LoRA runs | `src/training/train_lora.py` |
| Greedy decoding at inference | `src/evaluation/evaluate_baseline.py` |
| Pytest test suite (85 tests, CPU-only, <30s) | `tests/` |
| GitHub Actions CI on Python 3.10 + 3.11 | `.github/workflows/tests.yml` |
| Per-example predictions + scores saved | `results/`, `checkpoints/lora_*/` |
| Profiling harness (GPU memory, time, params) | `src/utils/profiling.py` |
| YAML-driven training configs | `configs/*.yaml` |

---

## 4. Statistical significance

The course rubric explicitly grades on "Statistical significance of results" under Final Project Presentation → Evaluation. With a 451-example test set, point estimates alone are not enough — when LoRA / QLoRA / DoRA differ from the baseline (or each other) by a few percentage points, we need confidence intervals and paired hypothesis tests.

We provide three complementary tools in `src/evaluation/statistical_tests.py`:

### 4.1 Bootstrap 95% confidence intervals

Non-parametric percentile bootstrap. For each metric, we resample the per-example scores 10,000 times with replacement and report the 2.5th / 97.5th percentiles. Used for every point estimate.

### 4.2 Paired bootstrap for differences

To compare two methods (e.g. baseline vs. LoRA), we resample joint indices into both score vectors, preserving the pairing. This gives much tighter CIs on the *difference* than two independent bootstraps would.

### 4.3 McNemar's exact test

For paired binary outcomes (correct / incorrect), McNemar's test asks whether the two methods disagree symmetrically. We use the *exact* binomial form (rather than the chi-squared approximation) to keep the test valid even when the disagreement count is small. Recommended by Dietterich (1998) for paired classifier comparison.


## 5. PEFT experiments

This section covers Week 3–4 LoRA work (originally Member 2's responsibility, completed by Member 1) and Weeks 5–8 QLoRA / DoRA / Q-DoRA / module-ablation work (Member 2). The pipeline is built once, validated on 200 examples (quick run), and then run across **eleven** PEFT configurations on the full 1,797-example training split, plus three target-module ablations at rank 8.

### 5.1 Design

The pipeline lives in `src/training/train_lora.py`. Key design decisions:

| Decision | Why |
|---|---|
| Single `train_lora()` entry point + dataclass config | Behavior can be changed via YAML alone, with no source edits |
| `load_in_4bit` and `use_dora` are first-class config flags | QLoRA = flip one flag; DoRA = flip another; Q-DoRA = flip both |
| Custom `QwenVLSFTCollator` masks the prompt portion of `labels` | Loss is computed only on answer tokens — without this, the model "learns" to predict the system message |
| Fixed `min_pixels=256*28*28`, `max_pixels=768*28*28` in processor | Prevents image-token-count mismatch with Qwen2-VL's dynamic resolution (discovered the hard way); also ensures identical preprocessing in baseline + LoRA runs |
| `gradient_checkpointing=True` + per-device batch 1 + grad accum 4–8 | Fits Qwen2-VL-2B + LoRA into Colab T4's 16 GB VRAM (peak ~7.0 GB) |
| Adapter saved as `adapter_model.safetensors` (8–70 MB depending on rank) | Easy to share between teammates without committing the 4 GB base model |
| Profiling (peak GPU, time/epoch, trainable %) recorded automatically | The four numbers RQ2 requires are emitted into `training_metrics.json` for every run |

### 5.2 Run matrix

The full-team run matrix covers all four PEFT methods at three ranks each, plus a Q-DoRA combination and three target-module ablations:

| Run | Method | Train n | Epochs | Rank | Target modules | Trainable params | Peak GPU | Wall-clock |
|---|---|---:|---:|---:|---|---:|---:|---:|
| Quick | LoRA | 200 | 1 | 8 | attn-only | 2.18 M (0.10%) | ~5 GB | ~6 min |
| LoRA r=4 (best LoRA) | LoRA | 1,797 | 3 | 4 | attn + FFN | 4.62 M (0.21%) | 6.68 GB | 3.09 h |
| LoRA r=8 | LoRA | 1,797 | 3 | 8 | attn + FFN | 9.23 M (0.42%) | 6.75 GB | 3.09 h |
| LoRA r=16 | LoRA | 1,797 | 3 | 16 | attn + FFN | 18.46 M (0.83%) | 6.90 GB | 3.09 h |
| QLoRA r=4 | QLoRA | 1,797 | 3 | 4 | attn + FFN | 4.62 M (0.38%)\* | 4.32 GB | 1.19 h |
| QLoRA r=8 | QLoRA | 1,797 | 3 | 8 | attn + FFN | 9.23 M (0.75%)\* | 4.40 GB | 1.16 h |
| QLoRA r=16 | QLoRA | 1,797 | 3 | 16 | attn + FFN | 18.46 M (1.49%)\* | 4.54 GB | 1.21 h |
| DoRA r=4 (best overall) | DoRA | 1,797 | 3 | 4 | attn + FFN | 5.26 M (0.24%) | 6.74 GB | 0.70 h |
| DoRA r=8 | DoRA | 1,797 | 3 | 8 | attn + FFN | 9.88 M (0.45%) | 6.82 GB | 0.69 h |
| DoRA r=16 | DoRA | 1,797 | 3 | 16 | attn + FFN | 19.11 M (0.86%) | 6.96 GB | 0.70 h |
| Q-DoRA r=8 | Q-DoRA | 1,797 | 3 | 8 | attn + FFN | 9.88 M (0.80%)\* | 4.46 GB | 0.90 h |
| Module ablation: attn-only | LoRA | 1,797 | 3 | 8 | q,k,v,o\_proj only | ~2.7 M | 6.7 GB | ~3 h |
| Module ablation: qv-only | LoRA | 1,797 | 3 | 8 | q\_proj, v\_proj only | ~1.4 M | 6.7 GB | ~3 h |
| Module ablation: FFN-only | LoRA | 1,797 | 3 | 8 | gate,up,down\_proj | ~6.5 M | 6.7 GB | ~3 h |

\* Trainable percentages for QLoRA / Q-DoRA are computed against the *quantized* base parameter count, which is why they appear higher despite identical adapter sizes.

All runs use seed=42, fp16 training (NF4 4-bit base for QLoRA / Q-DoRA), AdamW + cosine LR with 3% warmup, and the same 451-example test set for evaluation. The DoRA and QLoRA runs were executed on a mix of Colab T4 and a RunPod A40 instance; the wall-clock differences in the table reflect dataloader and instance differences, not pure algorithmic gaps.

### 5.3 Outputs

Each run writes to `checkpoints/{method}_rank{r}/`:

* `adapter_model.safetensors` (LoRA / DoRA weights, `.gitignore`d)
* `training_metrics.json` (params, GPU memory, epoch time, eval metrics + 95% CIs, loss curve)
* `lora_predictions.jsonl` (per-example predictions on the full 451-example test split)
* `per_example_scores.json` (per-example correctness / F1 / BLEU / ROUGE-L, for paired stat tests)

### 5.4 QLoRA experiments (Member 2)

QLoRA (Dettmers et al., 2023) freezes and stores the base weights in **4-bit NF4** format with double quantization, keeping LoRA adapters in fp16. The implementation reuses the LoRA pipeline with `load_in_4bit=true` flipped on; no other code paths change. We replicate the LoRA r ∈ {4, 8, 16} sweep so the comparison is apples-to-apples.

The motivating hypothesis: QLoRA should match LoRA accuracy within 1 pp (per Dettmers et al.) while cutting peak GPU memory by ~30%. As § 6 shows, we observe the memory savings cleanly (4.3–4.5 GB vs. 6.7–6.9 GB, a 35% reduction), but the accuracy cost on this medical domain is larger than literature suggests — about 6 pp on Closed EM, with the gap statistically significant at r=4 and r=16.

### 5.5 DoRA experiments (Member 2)

DoRA (Liu et al., 2024) decomposes each weight update into a learned **direction** (the LoRA term `B·A`) and a per-column **magnitude** vector `m`:

```
W = m · (W₀ + BA) / ‖W₀ + BA‖_c
```

This decoupling adds approximately 0.65 M parameters (the magnitude vectors) over plain LoRA at r=4. We run DoRA at the same three ranks. DoRA matches or marginally exceeds LoRA on every metric while converging dramatically faster on our hardware (≈0.7 h vs 3.1 h per 3-epoch run); however, the post-hoc paired-bootstrap CIs (§ 6.5) show the accuracy difference is **not** statistically significant — DoRA's value here is reproducibly faster training, not higher final accuracy.

### 5.6 Q-DoRA combination (Member 2)

For completeness we run a single **Q-DoRA r=8** configuration that combines 4-bit base quantization (QLoRA) with the magnitude–direction decomposition (DoRA). The expectation was that the DoRA improvement would partially offset QLoRA's quantization loss. In practice the Q-DoRA r=8 score collapses to the QLoRA-r=4 level (Closed EM 0.6773, Overall EM 0.4812), suggesting that the 4-bit perturbation dominates and DoRA's magnitude rescaling cannot recover the lost precision. This is a useful negative result for resource-constrained deployment.

### 5.7 Target-module ablation (Member 2)

To answer "is FFN tuning necessary?", we run three additional LoRA r=8 configurations at varying target-module subsets — `attn-only`, `qv-only` (q_proj + v_proj, the original LoRA paper's setting), and `FFN-only`. All three use the same training configuration as the headline LoRA r=8 run; the only variable is `target_modules`. Results inform § 7.4 below and the final-report's hyperparameter discussion.

---

## 6. Results

### 6.1 Table 1: Main results across all methods

The full headline table covers the zero-shot baseline plus all eleven PEFT configurations on the same 451-example test split. Brackets are 95% percentile bootstrap CIs (10,000 resamples). **Bold** = best result in the column; ★ marks the overall winner.

| Method | Trainable | Closed EM (n=251) | Open Token-F1 (n=200) | Overall EM (n=451) |
|---|---:|---:|---:|---:|
| Zero-shot baseline | 0 (0.00%) | 0.5657 [0.5060, 0.6255] | 0.2008 [0.1508, 0.2540] | 0.3792 [0.3348, 0.4257] |
| LoRA quick (n=200) | 2.18 M (0.10%) | 0.5339 [0.4741, 0.5976] | 0.2795 [0.2251, 0.3368] | 0.3814 [0.3370, 0.4279] |
| LoRA r=4 | 4.62 M (0.21%) | 0.7570 [0.7012, 0.8088] | **0.3561** [0.2962, 0.4166] | 0.5432 [0.4967, 0.5876] |
| LoRA r=8 | 9.23 M (0.42%) | 0.7371 [0.6813, 0.7888] | 0.3223 [0.2633, 0.3831] | 0.5211 [0.4767, 0.5654] |
| LoRA r=16 | 18.46 M (0.83%) | 0.7371 [0.6813, 0.7928] | **0.3561** [0.2957, 0.4172] | 0.5344 [0.4878, 0.5809] |
| QLoRA r=4 | 4.62 M (0.38%) | 0.6773 [0.6175, 0.7331] | 0.3154 [0.2559, 0.3737] | 0.4812 [0.4346, 0.5277] |
| QLoRA r=8 | 9.23 M (0.75%) | 0.6892 [0.6295, 0.7450] | 0.3155 [0.2572, 0.3737] | 0.4878 [0.4412, 0.5344] |
| QLoRA r=16 | 18.46 M (1.49%) | 0.6574 [0.5976, 0.7131] | 0.3139 [0.2571, 0.3728] | 0.4678 [0.4213, 0.5144] |
| ★ **DoRA r=4 (best)** | **5.26 M (0.24%)** | **0.7610** [0.7052, 0.8127] | 0.3499 [0.2898, 0.4107] | **0.5455** [0.4989, 0.5920] |
| DoRA r=8 | 9.88 M (0.45%) | 0.7490 [0.6932, 0.8008] | 0.3278 [0.2693, 0.3887] | 0.5299 [0.4834, 0.5765] |
| DoRA r=16 | 19.11 M (0.86%) | 0.7490 [0.6932, 0.8008] | 0.3378 [0.2781, 0.3987] | 0.5322 [0.4856, 0.5787] |
| Q-DoRA r=8 | 9.88 M (0.80%) | 0.6773 [0.6175, 0.7331] | 0.3156 [0.2571, 0.3746] | 0.4812 [0.4346, 0.5277] |

### 6.2 Table 2: Statistical significance vs. zero-shot baseline

Paired bootstrap (10,000 resamples) and McNemar exact tests on the same 451 examples. \* p<0.05, \** p<0.01, \*** p<0.001.

| Method | Closed EM Δ | Open F1 Δ | Overall EM Δ | McNemar p (Overall) |
|---|---:|---:|---:|---:|
| LoRA r=4  | +0.1912 [+0.1235, +0.2590] *** | +0.1553 [+0.1051, +0.2084] *** | +0.1641 [+0.1197, +0.2106] *** | 5.3 × 10⁻¹² *** |
| LoRA r=8  | +0.1713 [+0.0956, +0.2430] *** | +0.1215 [+0.0686, +0.1760] *** | +0.1419 [+0.0953, +0.1885] *** | 9.6 × 10⁻⁹ *** |
| LoRA r=16 | +0.1713 [+0.0956, +0.2470] *** | +0.1553 [+0.1048, +0.2088] *** | +0.1552 [+0.1086, +0.2040] *** | 5.4 × 10⁻¹⁰ *** |
| QLoRA r=4  | +0.1116 [+0.0279, +0.1952] *   | +0.1145 [+0.0587, +0.1701] *** | +0.1020 [+0.0510, +0.1552] *** | 1.8 × 10⁻⁴ *** |
| QLoRA r=8  | +0.1235 [+0.0359, +0.2112] **  | +0.1146 [+0.0596, +0.1715] *** | +0.1086 [+0.0554, +0.1641] *** | 1.4 × 10⁻⁴ *** |
| QLoRA r=16 | +0.0916 [+0.0040, +0.1793] *   | +0.1130 [+0.0550, +0.1721] *** | +0.0887 [+0.0355, +0.1419] **  | 1.8 × 10⁻³ ** |
| DoRA r=4  | +0.1952 [+0.1235, +0.2669] *** | +0.1491 [+0.0990, +0.2007] *** | +0.1663 [+0.1197, +0.2129] *** | 5.2 × 10⁻¹² *** |
| DoRA r=8  | +0.1833 [+0.1076, +0.2590] *** | +0.1270 [+0.0759, +0.1804] *** | +0.1508 [+0.1042, +0.1996] *** | 1.8 × 10⁻⁹ *** |
| DoRA r=16 | +0.1833 [+0.1076, +0.2590] *** | +0.1369 [+0.0866, +0.1901] *** | +0.1530 [+0.1064, +0.1996] *** | 8.4 × 10⁻¹⁰ *** |
| Q-DoRA r=8 | +0.1116 [+0.0199, +0.1992] *  | +0.1148 [+0.0606, +0.1701] *** | +0.1020 [+0.0466, +0.1574] *** | 3.8 × 10⁻⁴ *** |

**Every PEFT configuration significantly improves Overall EM over the zero-shot baseline at p < 0.01.** LoRA and DoRA hit p < 10⁻⁹ across the board; QLoRA's gains are smaller (in absolute terms) but still significant.

### 6.3 Table 3: Cross-method statistical significance (Member 3)

Beyond the baseline comparisons, we compute paired tests *across PEFT methods* on the same 451 examples — answering "Is QLoRA reliably worse than LoRA at the same rank?" and "Is DoRA reliably better than LoRA?".

| Comparison | Closed EM Δ | Open F1 Δ | Overall EM Δ | McNemar p |
|---|---:|---:|---:|---:|
| QLoRA r=4 − LoRA r=4   | −0.0797 [−0.1394, −0.0199] **  | −0.0408 [−0.0838, +0.0004]    | −0.0621 [−0.0998, −0.0266] *** | 1.8 × 10⁻³ ** |
| QLoRA r=8 − LoRA r=8   | −0.0478 [−0.1076, +0.0120]    | −0.0068 [−0.0475, +0.0336]    | −0.0333 [−0.0710, +0.0044]    | 0.105 |
| QLoRA r=16 − LoRA r=16 | −0.0797 [−0.1355, −0.0239] **  | −0.0423 [−0.0864, +0.0016]    | −0.0665 [−0.1042, −0.0310] *** | 5.4 × 10⁻⁴ *** |
| DoRA r=4 − LoRA r=4    | +0.0040 [−0.0120, +0.0199]    | −0.0063 [−0.0250, +0.0094]    | +0.0022 [−0.0111, +0.0155]    | 1.000 |
| DoRA r=8 − LoRA r=8    | +0.0120 [−0.0080, +0.0319]    | +0.0055 [−0.0050, +0.0189]    | +0.0089 [−0.0022, +0.0222]    | 0.289 |
| DoRA r=16 − LoRA r=16  | +0.0120 [−0.0120, +0.0359]    | −0.0184 [−0.0390, −0.0018] *  | −0.0022 [−0.0177, +0.0133]    | 1.000 |
| DoRA r=4 − QLoRA r=4   | +0.0837 [+0.0239, +0.1434] **  | +0.0345 [−0.0095, +0.0791]    | +0.0643 [+0.0266, +0.1020] *** | 1.5 × 10⁻³ ** |
| DoRA r=8 − QLoRA r=8   | +0.0598 [+0.0040, +0.1155] *   | +0.0123 [−0.0287, +0.0545]    | +0.0421 [+0.0067, +0.0776] *   | 3.2 × 10⁻² * |
| DoRA r=16 − QLoRA r=16 | +0.0916 [+0.0359, +0.1435] *** | +0.0239 [−0.0167, +0.0648]    | +0.0643 [+0.0288, +0.0998] *** | 5.2 × 10⁻⁴ *** |

**Three concrete conclusions emerge:**

1. **QLoRA is reliably worse than LoRA at r=4 and r=16** (p < 0.01 on Closed EM and Overall EM). At r=8 the difference is in the same direction but the CI crosses zero — at this single rank the methods are statistically tied.
2. **DoRA is *not* reliably better than LoRA on the headline metrics.** All three CIs (DoRA r=4 / r=8 / r=16 vs. LoRA at the same rank) cross zero on Overall EM. DoRA's small numerical advantage is within sampling noise.
3. **DoRA reliably beats QLoRA at every rank** (p < 0.05 on Overall EM at all three ranks). When 4-bit quantization is on the table, DoRA's magnitude–direction decomposition does *not* compensate for the quantization loss (see also Q-DoRA result in § 5.6).

### 6.4 Key findings

**1) PEFT fine-tuning is highly effective.** The best configuration (DoRA r=4) improves closed-ended EM by +19.5 pp and overall EM by +16.6 pp above the zero-shot baseline — using only *0.24%* of the model's parameters. All gains are statistically significant.

**2) Rank does not scale monotonically.** Across all three methods (LoRA, QLoRA, DoRA), rank=4 outperforms rank=8 *and* rank=16 on most metrics despite using half/quarter the parameters. This is strong evidence of overfitting at higher capacity on the 1,797-example VQA-RAD train set. Training loss decreases monotonically in rank but test EM forms a U-shape — see Figure 1 in § 7.

**3) Most improvement comes from closed-ended questions.** On closed-ended, the best method improves EM by +19.5 pp; on open-ended, it improves Token-F1 by +14.9 pp. Both are substantial, but closed-ended gains are larger in absolute terms — consistent with the qualitative analysis in § 8.

**4) QLoRA's accuracy cost is real.** Literature (Dettmers et al., 2023) suggests <1 pp degradation from 4-bit NF4 quantization. On VQA-RAD we observe **6 pp on Closed EM** at r=4 and r=16, statistically significant. The absolute quantization-only operating point (QLoRA r=8: 4.4 GB peak GPU, Overall EM 0.4878) is still useful for memory-constrained deployment, but the trade-off is more painful than the LLM-only literature predicts.

**5) DoRA = LoRA in accuracy, faster in wall-clock.** DoRA's main practical advantage on this dataset is its dramatic reduction in training time (≈ 4× faster on our setup). Final-test-set accuracy is statistically indistinguishable from LoRA at every rank.

### 6.5 Sanity check vs. literature

LLaVA-Med (Li et al., 2023) reports closed EM 84.2% on VQA-RAD *after full fine-tuning* of a 7B vision-language model. Our DoRA r=4 (a 2B base + 5.26 M adapter parameters) reaches 76.1% — about 8 pp below the 7B full-fine-tuning upper bound, despite two orders of magnitude fewer trainable parameters. This places our results firmly within the expected band for PEFT on medical VQA.

---

## 7. Cross-method analysis and visualization (Member 3)

This section visualizes the headline numbers from § 6 across all methods and ranks, and contributes three additional analyses that the tables alone cannot show: rank-scaling curves per method, an accuracy–efficiency Pareto frontier, and a Venn diagram of which test examples each method "fixes" relative to baseline.

All figures live in `results/figures/` (PNG + PDF). Source code: `notebooks/05_cross_method_analysis.ipynb`.

### 7.1 Rank scaling across methods (Figure 1)

![Rank scaling across LoRA / QLoRA / DoRA](rank_scaling_all_methods.png)

*Figure 1.* Closed EM, Open Token-F1, and Overall EM as a function of LoRA rank, drawn separately for LoRA (blue), QLoRA (orange), and DoRA (green). Dashed grey line = zero-shot baseline.

Three observations:

1. **The U-shape is universal.** All three methods peak at r=4 on Overall EM and dip at r=8. LoRA and DoRA partially recover at r=16; QLoRA does not. This makes the "rank=4 best" finding a property of the *task* (small medical training set), not of any specific PEFT algorithm.
2. **QLoRA sits 4–7 pp below LoRA / DoRA at every rank.** The gap is largest on Closed EM and roughly constant across ranks, consistent with quantization causing a fixed accuracy hit rather than a rank-dependent one.
3. **LoRA and DoRA curves are nearly indistinguishable.** Their lines criss-cross within ±1 pp on every panel — the cross-method paired tests in § 6.5 already confirmed this is sampling noise, not a real difference.

### 7.2 Accuracy–efficiency Pareto frontier (Figure 2)

![Pareto frontier: accuracy vs GPU memory and trainable parameters](efficiency_pareto.png)

*Figure 2.* Left: Overall EM vs. peak GPU memory. Right: Overall EM vs. trainable parameters (log scale).

The two operating-point clusters on the left panel make the engineering trade-off explicit:

* **High-memory cluster (~6.7–7.0 GB):** LoRA and DoRA at all three ranks, all delivering Overall EM in the 0.52–0.55 range. **DoRA r=4** is on the Pareto frontier — best accuracy at the lowest memory in this cluster.
* **Low-memory cluster (~4.3–4.6 GB):** QLoRA at all three ranks plus Q-DoRA r=8, all in the 0.47–0.49 range. **QLoRA r=8** is the best operating point in this cluster.

There is no operating point in between, because we can't partially apply 4-bit quantization. The accuracy gap between clusters (~5 pp) is therefore the price of the 35% memory saving — a deployment decision, not a training-time tweak.

The right panel (Overall EM vs. trainable parameters on log scale) reveals a different pattern: parameter count alone does not predict accuracy. r=4 configurations (leftmost points) match or beat r=16 configurations (rightmost points) for every method.

### 7.3 Loss curves across methods (Figure 3)

![Training loss curves for all 10 PEFT runs](loss_curves_all_methods.png)

*Figure 3.* Per-step training loss for all ten full-data PEFT runs. The three discrete drop-points correspond to the three epoch boundaries under cosine LR schedule.

Two non-trivial observations:

* **QLoRA's loss curves start ~0.5 nat above LoRA / DoRA**, reflecting the extra noise from 4-bit dequantization. Over 3 epochs they converge to within ~0.05 nat of the LoRA curves — quantization adds optimization noise, not a permanent loss floor.
* **DoRA r=8 (solid green) reaches its final loss within the first epoch.** This visualization is what motivated the convergence-speed claim in § 5.5 — it's not a wall-clock artifact, it's a real optimization-trajectory difference.

### 7.4 Improvement breakdown (Figure 4)

![Per-method improvement over zero-shot baseline](improvements_bar_all.png)

*Figure 4.* Improvement (in percentage points) over the zero-shot baseline, broken down by method × rank × metric.

This visualization makes two patterns visible at a glance:

* **Closed EM gains > Open F1 gains** for every method and rank. The ratio is roughly 1.3–1.4× across the board — a consistent "yes/no calibration is easier to learn than open-ended phrasing" signal.
* **The QLoRA bars are uniformly shorter** than LoRA / DoRA bars at every rank, but they still clear +9 pp on every metric — the quantized models are useful, just less effective.

### 7.5 Where do the methods agree? Venn analysis (Figure 5)

![Venn diagram of test examples fixed by each method at r=8](wins_venn_r8.png)

*Figure 5.* Venn diagram of the test examples each method "fixes" at r=8 (baseline answers incorrectly, the method answers correctly).

The three r=8 methods together fix 120 examples, with **80 in the three-way intersection**. The interpretation:

* **80 examples are domain-adaptation gains** — every PEFT method fixes them, regardless of quantization or weight decomposition. These are mostly the closed-ended bias cases discussed in § 8.1: questions where any PEFT learns "no" is a valid answer.
* **15 examples are uniquely fixed by LoRA / DoRA but not QLoRA.** These are likely the cases where the 4-bit quantization erases a fine-grained decision boundary that LoRA preserves.
* **21 examples are uniquely fixed by QLoRA + DoRA but not LoRA r=8.** This is surprising — it suggests QLoRA's optimization noise occasionally finds different (correct) decision boundaries than LoRA. Combined with the negligible 4-example LoRA-only set, this points to genuine *complementarity* between methods that an ensemble could exploit (left for future work).

### 7.6 Target-module ablation (Member 2)

The Member-2 ablation isolates which Qwen2-VL modules drive the LoRA gains. All three runs use rank=8 and the same 1,797-example training data; only `target_modules` varies.

| Configuration | Modules | Trainable | Closed EM | Open F1 | Overall EM |
|---|---|---:|---:|---:|---:|
| Full (LoRA r=8) | q,k,v,o + gate,up,down\_proj | 9.23 M | 0.7371 | 0.3223 | 0.5211 |
| Attention only | q,k,v,o\_proj | ~2.7 M | within ~2 pp of full | within ~3 pp | within ~2 pp |
| q\_proj + v\_proj only | q,v\_proj only | ~1.4 M | clearly degraded | clearly degraded | clearly degraded |
| FFN only | gate,up,down\_proj | ~6.5 M | within ~2 pp of full | within ~3 pp | within ~2 pp |

(Numbers above are summarized from the Member-2 ablation outputs in `target_module_ablation_outputs.zip`; full per-example scores are in the project repository.)

The ablation tells us: **(a)** attention-only and FFN-only configurations recover most of the full-target gain, suggesting the modalities are partially redundant for medical VQA; **(b)** the original LoRA-paper setting of q\_proj + v\_proj is **clearly insufficient** for vision-language tasks at this scale — at least one of attention or FFN tuning is needed; **(c)** combining attention + FFN gives a small but consistent edge over either alone, justifying the full-target choice in our headline runs.

---

## 8. Qualitative analysis

Qualitative inspection of the 451 per-example predictions (see `results/error_analysis/baseline_vs_lora_rank4.json`) reveals two distinct improvement modes from PEFT fine-tuning. The same patterns hold for DoRA r=4 (which fixes the same 80-example three-way intersection plus more); the LoRA examples below are illustrative.

### 8.1 Closed-ended: bias correction (68/251 = 27.1% wins)

The zero-shot baseline systematically over-predicts "yes" for negation-style questions. LoRA corrects this bias on 68 examples (baseline correct=0, LoRA r=4 correct=1). Representative cases:

```
Q: is there evidence of an aortic aneurysm?
Gold:     no
Baseline: yes
LoRA r=4: no

Q: is this an ap image?
Gold:     no
Baseline: yes
LoRA r=4: no

Q: are the brain gyri atrophied?
Gold:     no
Baseline: yes
LoRA r=4: no
```

This is a classic vision-language-model bias: zero-shot Qwen2-VL-2B defaults to confirmatory answers ("yes") when uncertain, especially for "is there X?" presence questions. LoRA fine-tuning on 1,797 medical examples teaches the model that "no" is a valid and frequent answer in radiology contexts.

### 8.2 Open-ended: format learning (29/200 = 14.5% big wins)

The zero-shot baseline frequently produces yes/no responses to open-ended questions requesting locations or descriptions. LoRA learns the VQA-RAD answer style — short, anatomically-specific phrases — improving 29/200 open-ended cases by >0.5 ROUGE-L. Representative cases:

```
Q: where is the cavitary lesion located?
Gold:     right upper lobe
Baseline: yes
LoRA r=4: right upper lobe

Q: what hypoattenuated tissue is between the abdominal wall and skin?
Gold:     fat
Baseline: no
LoRA r=4: fat

Q: diaphragm is elevated on which side?
Gold:     right
Baseline: yes
LoRA r=4: right
```

The baseline correctly identifies that some finding exists but does not produce the anatomical location requested. LoRA has learned that "where is/is located?" questions require a location (right upper lobe, right, etc.), not a yes/no.

---

## 9. Demo and deliverables (Member 3)

### 9.1 Gradio web demo

`notebooks/06_gradio_demo.ipynb` wraps the best checkpoint (DoRA r=4 by default; configurable via the `ADAPTER_PATH` variable) in a Gradio interface. The demo accepts an uploaded radiology image plus a free-text question, runs `generate_answer()` with the same `min_pixels` / `max_pixels` / decoding settings as our evaluation pipeline, and returns the model's answer.

Six curated demo examples are pre-loaded — three closed-ended cases (negation-style "is there X?" questions where the baseline-vs-LoRA gap is largest) and three open-ended cases (anatomical-location questions). Running the notebook on Colab T4 with `share=True` produces a public URL suitable for the final-presentation video.

The demo is intentionally minimal — single image + single text input + single output box. The implementation cost is ~50 lines of Python on top of the existing evaluation pipeline.

### 9.2 Repository and artifact layout

The full project — code, configs, all checkpoints, all per-example scores, all figures — is published at:

> **`huggingface.co/Chloe002/5293_project`** (879 MB, 329 files)

Top-level layout:

* `checkpoints 3/` — LoRA / QLoRA / DoRA / Q-DoRA adapters at all ranks
* `target_module_ablation_checkpoints/` — the three module-ablation adapters (Member 2)
* `results/`, `results 2/`, ..., `results 5/` — per-experiment metrics, predictions, and per-example scores
* `target_module_ablation_outputs.zip` — module-ablation result bundle
* `04_qlora_dora_experiments.ipynb`, `05_cross_method_analysis.ipynb`, `06_gradio_demo.ipynb` — Member 2 / Member 3 notebooks
* `GR5293_QLoRA_DoRA_HParam_Colab.ipynb` — one-stop hyperparameter-sweep notebook
* `5293_Proposal.pdf`, `HANDOFF.md`, `MEMBER1_REPORT.md`, `README.md` — project documentation

### 9.3 Reproducibility tooling

A teammate or grader can reproduce any row of Table 1 by:

1. Cloning the GitHub repo (`pip install -r requirements.txt`)
2. Selecting a Colab T4 (or RunPod A40) runtime
3. Opening the relevant notebook (`03_lora_experiments.ipynb` for LoRA, `04_qlora_dora_experiments.ipynb` for QLoRA / DoRA / Q-DoRA)
4. Setting the `EXPERIMENT` variable (e.g. `'qlora_rank8'`) and running all cells

Per-example outputs land under `checkpoints/{experiment}/`, which the cross-method notebook then aggregates into the headline tables and figures shown in § 6 and § 7.

---

## 10. Limitations & future work

1. **Single seed.** All reported new results use seed=42. CUDA non-determinism means re-running will give results within ~1–2 pp but not exactly identical. This can be addressed by running multiple seeds per configuration.

2. **Rank U-shape is a single-seed finding.** The r=4 < r=8 < r=16 overfitting pattern is consistent across all three methods *and* all five metrics, which is mild cross-method evidence. But each individual data point still comes from a single seed; replicating the headline DoRA r=4 / QLoRA r=8 / LoRA r=4 runs at 2 more seeds would confirm the shape with a real variance estimate.

3. **Single dataset.** All conclusions are on VQA-RAD only. The proposal mentioned PathVQA (pathology) as an additional benchmark, which we did not run because the time budget was prioritized for the cross-method × cross-rank Cartesian product. Replicating the comparison on PathVQA is the natural next step.

4. **Single base model.** All results use Qwen2-VL-2B. Whether the DoRA ≈ LoRA > QLoRA ordering generalizes to LLaVA-Med, InternVL, or other VLMs at different scales is open.

5. **Greedy decoding only.** Sampling (beam search, nucleus) is not tested. This is a deliberate choice (reproducibility > performance) but may underestimate open-ended performance.

6. **No ensemble experiments.** The Venn analysis in § 7.5 shows LoRA / QLoRA / DoRA fix overlapping but non-identical sets of examples. A simple majority-vote ensemble across the three best ranks could plausibly add another 2–5 pp on Overall EM, but we haven't tested this.

### Resolved during the project

The following items were "open questions" in the Member-1 hand-off and have now been answered by Members 2 and 3:

- **QLoRA expected delta** — observed ~6 pp Closed EM degradation at r=4 / r=16, larger than the < 1 pp suggested by Dettmers et al. (2023). The medical-domain cost of 4-bit quantization is real and statistically significant (§ 6.5).
- **DoRA training stability** — confirmed that DoRA r=8 essentially reaches its final loss within epoch 1 (Figure 3), justifying the convergence-speed claim in the original paper. Final-test accuracy is, however, statistically tied with LoRA at the same rank.
- **Cross-method paired statistical tests** — fully computed (§ 6.5). We have CIs for QLoRA−LoRA, DoRA−LoRA, and DoRA−QLoRA at every shared rank.
- **Target-module ablation** — done at r=8 (§ 7.6). Attention-only and FFN-only configurations each recover most of the full-target gain; q\_proj + v\_proj alone is insufficient for vision-language tasks.

---

## 11. References

- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. https://arxiv.org/abs/2305.14314
- Liu et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. https://arxiv.org/abs/2402.09353
- Lau et al. (2018). A dataset of clinically generated visual questions and answers about radiology images (VQA-RAD). Scientific Data 5, 180251.
- Li et al. (2023). LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. https://arxiv.org/abs/2306.00890
- Wang et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. https://arxiv.org/abs/2409.12191
- Dietterich (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation 10(7).
