# Member 1 → Members 2 & 3 Hand-off

> This is the **single onboarding document** for Members 2 and 3. It tells you (a) what Member 1 finished, (b) what files you need from Member 1's work, (c) exactly how to run QLoRA / DoRA / analysis / demo, (d) what numbers to expect, and (e) the gotchas Member 1 hit so you don't re-discover them.

---

## Table of contents

- [1. What Member 1 finished](#1-what-member-1-finished)
- [2. Files you'll need from Member 1's work](#2-files-youll-need-from-member-1s-work)
- [3. Environment setup (do this first)](#3-environment-setup-do-this-first)
- [4. The single source of truth](#4-the-single-source-of-truth)
- [5. How to run QLoRA (Member 2)](#5-how-to-run-qlora-member-2)
- [6. How to run DoRA (Member 2)](#6-how-to-run-dora-member-2)
- [7. The four metrics every method must report](#7-the-four-metrics-every-method-must-report)
- [8. Per-example scores schema (paired tests)](#8-per-example-scores-schema-paired-tests)
- [9. Sanity-check ranges](#9-sanity-check-ranges)
- [10. Cross-method analysis (Member 3)](#10-cross-method-analysis-member-3)
- [11. Building the Gradio demo (Member 3)](#11-building-the-gradio-demo-member-3)
- [12. Known gotchas Member 1 hit](#12-known-gotchas-member-1-hit)
- [13. Common errors and fixes](#13-common-errors-and-fixes)
- [14. What you must NOT change](#14-what-you-must-not-change)
- [15. Commit conventions](#15-commit-conventions)
- [16. Final-results checklist](#16-final-results-checklist)

---

## 1. What Member 1 finished

Weeks 1–4 deliverables (see `docs/MEMBER1_REPORT.md` for the full writeup):

1. **Data pipeline**: `flaviagiammarino/vqa-rad` HuggingFace loader, question-type classifier, custom Dataset + collator
2. **Zero-shot baseline**: Qwen2-VL-2B-Instruct on the 451-example test split — Closed EM 0.5657, Open Token-F1 0.2008, Overall EM 0.3792 (with 95% bootstrap CIs)
3. **Statistical-test infrastructure**: bootstrap CI, paired bootstrap, McNemar exact test (used by all three of us)
4. **LoRA quick run**: 200 examples, 1 epoch, rank 8, attention-only (pilot study, validates pipeline end-to-end in ~6 min)
5. **LoRA full-data 3-rank ablation**: r ∈ {4, 8, 16} on full 1,797 train examples, 3 epochs
   * Best result: **r=4** with Closed EM 0.7570, Open Token-F1 0.3561, Overall EM 0.5432
   * Counter-intuitive U-shape (r=4 beats r=8 *and* r=16) — overfitting evidence
6. **Repo scaffolding + reproducibility**: pinned deps, 85 pytest tests, GitHub Actions CI, profiling harness
7. **Documentation**: this hand-off, `MEMBER1_REPORT.md`, `DATA_CARD.md`, 6-paper literature review

All numbers above are paired-bootstrap-significant vs. baseline (CIs don't cross zero, McNemar p < 10⁻⁶). Full Tables 1 + 2 in `results/tables/main_results.md`.

---

## 2. Files you'll need from Member 1's work

### For Member 2 (QLoRA / DoRA experiments)

You will **call** (i.e. import or run) Member 1's code:

| File | What it does | You'll touch? |
|---|---|---|
| `src/training/train_lora.py` | The training entry point. Reads YAML, builds model + LoRA, trains, evaluates, saves results. | Run it (no code edits) |
| `src/training/data_collator.py` | Masks the prompt portion of `labels` so loss is computed only on answer tokens. | **Don't change** |
| `src/training/_enable_input_grads.py` | Helper that re-enables input gradients after PEFT wraps the model (required for gradient_checkpointing). | **Don't change** |
| `src/data/vqarad_dataset.py` | Dataset + prompt builder. Contains `SYSTEM_PROMPT`. | **Don't change** |
| `src/data/load_vqarad.py` | HuggingFace loader + question-type classifier. | **Don't change** |
| `src/evaluation/evaluate_baseline.py` | `generate_answer()` function — used at end of training to evaluate the LoRA adapter. | **Don't change** |
| `src/evaluation/metrics.py` | Exact Match, BLEU-1, ROUGE-L, Token-F1. | **Don't change** |
| `src/evaluation/statistical_tests.py` | Bootstrap CI, paired bootstrap, McNemar. You'll call these for QLoRA-vs-LoRA comparisons. | Import (no edits) |
| `configs/lora_rank8.yaml` | Template you'll copy + modify for QLoRA / DoRA. | Copy, then edit copy |
| `configs/baseline_config.yaml` | Reference for baseline settings. | Read-only |

You will **read** (not modify) Member 1's results to do paired statistical tests:

| File | Use |
|---|---|
| `results/baseline/per_example_scores.json` | Per-example correctness for paired tests vs your QLoRA/DoRA |
| `checkpoints/lora_rank4/per_example_scores.json` | Best LoRA — compare your QLoRA / DoRA against this |
| `checkpoints/lora_rank8/per_example_scores.json` | Same-rank LoRA reference for QLoRA vs LoRA at r=8 |
| `results/baseline/predictions.jsonl` | Per-example predictions if you want to inspect specific cases |
| `docs/MEMBER1_REPORT.md` § 6 | Tables 1 + 2 (already paper-ready) |

### For Member 3 (analysis + demo)

| File | Use |
|---|---|
| All of Member 1's `per_example_scores.json` files | Build cross-method comparison tables (LoRA r=4 vs QLoRA vs DoRA) |
| All `training_metrics.json` files | Pull trainable params, peak GPU memory, training time for the efficiency table |
| `results/figures/loss_curves.{png,pdf}` | Already plotted — extend to include QLoRA/DoRA |
| `results/figures/rank_scaling.{png,pdf}` | Reference plot style |
| `results/error_analysis/baseline_vs_lora_rank4.json` | Qualitative wins example, extend to QLoRA/DoRA |
| `notebooks/03_lora_experiments.ipynb` | Template — copy for your demo notebook |
| `checkpoints/lora_rank4/` | Best-performing adapter — load it for the demo |

### Files you absolutely DO NOT need to touch

- `tests/` — Member 1's test suite, runs automatically on CI
- `scripts/` — convenience wrappers, optional
- `.github/workflows/tests.yml` — CI config

---

## 3. Environment setup (do this first)

### Option A: Google Colab (recommended — what Member 1 used)

```python
# Cell 1: select runtime
# Runtime → Change runtime type → T4 GPU → Save

# Cell 2: clone
!git clone https://github.com/<USER>/peft-medical-vqa.git
%cd peft-medical-vqa

# Cell 3: install dependencies (~2-3 min)
# Colab pre-installs torch with the right CUDA version; we don't reinstall it.
%pip install -q -r requirements.txt

# Cell 4: make project importable
import os, sys
PROJECT_ROOT = '/content/peft-medical-vqa'
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"CWD: {os.getcwd()}")

# Cell 5: smoke test (verifies environment)
!python -m pytest tests/ -q
# Expected: 85 passed in <30s
```

### Option B: Local with GPU (≥10 GB VRAM)

```bash
git clone https://github.com/<USER>/peft-medical-vqa.git
cd peft-medical-vqa
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

### How to know setup is correct

`pytest tests/` prints `85 passed`. If anything fails, **stop** and check:
1. Python version — must be 3.10 or 3.11
2. Pinned versions — `pip list | grep -E "torch|transformers|peft|bitsandbytes"`
3. Network access to HuggingFace (for downloading Qwen2-VL-2B model and VQA-RAD dataset)

---

## 4. The single source of truth

Everything goes through one entry point:

```bash
python -m src.training.train_lora --config configs/<your_config>.yaml
```

`src/training/train_lora.py:train_lora()` reads the YAML, builds the model + LoRA adapters, runs training, evaluates, and saves `training_metrics.json` + `per_example_scores.json` + `lora_predictions.jsonl` + `adapter_model.safetensors`.

**You should not need to modify `train_lora.py` itself.** If you do, propose the change in a PR — diverging implementations break the cross-method comparison. Member 1 will then re-run baseline + 3-rank ablation so we're all on the same code.

---

## 5. How to run QLoRA (Member 2)

### Step 1 — Copy the config

```bash
cp configs/lora_rank8.yaml configs/qlora_rank8.yaml
```

### Step 2 — Edit two lines

Open `configs/qlora_rank8.yaml`:

```yaml
load_in_4bit: true                       # was: false
output_dir: "checkpoints/qlora_rank8"    # was: "checkpoints/lora_rank8"
```

Everything else (rank, target_modules, learning rate, etc.) stays the same so the comparison is apples-to-apples.

### Step 3 — Train

```bash
python -m src.training.train_lora --config configs/qlora_rank8.yaml
```

The pipeline automatically:
- Loads Qwen2-VL-2B in 4-bit NF4 quantization
- Wraps it with `prepare_model_for_kbit_training`
- Applies LoRA adapters in fp16 on top of the quantized base
- Records peak GPU memory (will be ~40–50% lower than fp16 LoRA)

Wall time on Colab T4: ~3 hours.

### Step 4 — Compare to LoRA statistically

```python
from src.evaluation.statistical_tests import paired_bootstrap_ci_diff, mcnemar_test
import json

with open('checkpoints/lora_rank8/per_example_scores.json') as f:
    lora = json.load(f)
with open('checkpoints/qlora_rank8/per_example_scores.json') as f:
    qlora = json.load(f)
assert lora['ids'] == qlora['ids'], "Per-example IDs must match"

diff = paired_bootstrap_ci_diff(lora['correct'], qlora['correct'], seed=42)
mc = mcnemar_test(lora['correct'], qlora['correct'])
print(f"QLoRA - LoRA Overall EM: {diff['point']:+.4f}  95% CI [{diff['lower']:+.4f}, {diff['upper']:+.4f}]")
print(f"McNemar p-value: {mc['p_value']:.4g}")
```

Also compare against `lora_rank4` (best LoRA) and against `baseline`. See § 8 for the score schema.

### Step 5 — QLoRA rank ablation

Member 1 has done this for plain LoRA (r ∈ {4, 8, 16}). Replicate for QLoRA, optionally extending to r=32:

```bash
for r in 4 8 16; do
    python -m src.training.train_lora \
        --config configs/qlora_rank8.yaml \
        --rank $r \
        --output_dir checkpoints/qlora_rank${r}
done
```

Plot Closed EM / Token-F1 / Overall EM vs. rank. **Open question**: does QLoRA show the same r=4 < r=8 < r=16 U-shape that LoRA showed?

---

## 6. How to run DoRA (Member 2)

### Step 1 — Copy the config

```bash
cp configs/lora_rank8.yaml configs/dora_rank8.yaml
```

### Step 2 — Edit two lines

```yaml
use_dora: true                          # was: false
output_dir: "checkpoints/dora_rank8"    # was: "checkpoints/lora_rank8"
```

### Step 3 — Train

```bash
python -m src.training.train_lora --config configs/dora_rank8.yaml
```

The PEFT library handles DoRA natively when `use_dora=True` is set on `LoraConfig` — no additional library or code changes needed.

### Step 4 — Compare

Same paired-bootstrap pattern as QLoRA. Interesting comparisons:
- DoRA vs. zero-shot baseline (RQ1: does decomposition help over zero-shot?)
- DoRA vs. LoRA at matched rank (the headline DoRA paper claim — does decomposition beat plain LoRA?)
- DoRA vs. QLoRA (efficiency-quality trade-off)

### Step 5 — DoRA rank ablation

Same pattern as QLoRA above.

### Optional: Q-DoRA (combined)

If time permits — flip both `load_in_4bit: true` AND `use_dora: true`. Member 1's pipeline supports this with no code changes.

---

## 7. The four metrics every method must report

For RQ2 (efficiency), all of us extract the same four numbers from each `training_metrics.json`. The pipeline writes them automatically — **do not compute by hand**:

| Field | Path in `training_metrics.json` | Member 1 example (LoRA r=4) |
|---|---|---|
| Trainable parameters | `params.trainable` | 4,616,192 |
| Trainable % of total | `params.trainable_pct` | 0.2085 |
| Peak GPU memory (GB) | `training.peak_gpu_gb` | 6.68 |
| Total training seconds | `training.total_seconds` | ~10800 |

These plug directly into the headline efficiency table in the final report.

---

## 8. Per-example scores schema (paired tests)

Every method (baseline, LoRA, QLoRA, DoRA) emits `per_example_scores.json` with **identical structure**. This is the single interface for paired statistical tests.

### Schema

```json
{
  "ids":     [0, 1, 2, ..., 450],          // 451 unique example IDs (must match across methods)
  "qtypes":  ["closed", "closed", "open", ...], // length 451
  "correct": [0, 1, 0, 1, ...],            // length 451, exact match (1 if prediction matches reference)
  "bleu1":   [0.0, 0.0, 0.85, ...],        // length 451, BLEU-1 score
  "rougeL":  [0.0, 0.0, 1.0, ...],         // length 451, ROUGE-L F-measure
  "f1":      [0.0, 0.0, 1.0, ...]          // length 451, Token-F1
}
```

Notes:
- `correct` is **always** 0 or 1 (integer), not bool. For closed-ended it's strict EM; for open-ended it's also EM (note: open-ended EM is usually low, that's why we report Token-F1 as the headline open metric).
- `bleu1`, `rougeL`, `f1` are continuous in [0, 1].
- For closed-ended examples, `bleu1`/`rougeL`/`f1` are still computed but typically equal `correct` (since answers are 1-token "yes"/"no").
- `ids` is the index into the test split — **must be identical across methods** to do paired tests. The pipeline guarantees this.

### Sample paired-bootstrap usage

```python
from src.evaluation.statistical_tests import paired_bootstrap_ci_diff
import json

with open('results/baseline/per_example_scores.json') as f:
    base = json.load(f)
with open('checkpoints/qlora_rank8/per_example_scores.json') as f:
    qlora = json.load(f)

# Filter to closed-ended only:
idx = [i for i, q in enumerate(base['qtypes']) if q == 'closed']
a = [base['correct'][i] for i in idx]
b = [qlora['correct'][i] for i in idx]
diff = paired_bootstrap_ci_diff(a, b, n_resamples=10000, seed=42)
print(f"QLoRA - baseline closed EM: {diff['point']:+.4f} [{diff['lower']:+.4f}, {diff['upper']:+.4f}]")
```

---

## 9. Sanity-check ranges

If your QLoRA / DoRA numbers are way off, you have a bug. Use these as a guide:

### QLoRA expected (compare to LoRA r=8 reference numbers in parens)

| Metric | Expected range | LoRA r=8 reference |
|---|---|---|
| Closed EM | 0.65–0.78 | 0.7371 |
| Open Token-F1 | 0.25–0.38 | 0.3223 |
| Overall EM | 0.45–0.58 | 0.5211 |
| Trainable params | ~9.2M | 9,232,384 |
| Peak GPU memory | 3.5–4.5 GB | 6.75 GB (LoRA fp16) |
| Wall time | similar to LoRA | ~3 hours |

QLoRA should be **slightly worse or equal** to LoRA on accuracy (Dettmers et al., 2023 reports < 1pp degradation), but **40–50% lower peak GPU memory**.

If QLoRA Closed EM < 0.55: bug. If QLoRA peak GPU > 6 GB: bug (4-bit not actually applied).

### DoRA expected (compare to LoRA r=8)

| Metric | Expected range | LoRA r=8 reference |
|---|---|---|
| Closed EM | 0.70–0.80 | 0.7371 |
| Open Token-F1 | 0.30–0.40 | 0.3223 |
| Overall EM | 0.50–0.60 | 0.5211 |
| Trainable params | ~9.3M (slightly more than LoRA — DoRA adds magnitude vector) | 9,232,384 |
| Peak GPU memory | 7.0–8.0 GB | 6.75 GB |
| Wall time | 1.1–1.3× LoRA | ~3 hours |

DoRA paper claims it should match-or-beat LoRA at the same rank. If your DoRA is **worse** than LoRA, double-check `use_dora: true` is actually in the saved config.

### Red flags (something is wrong)

- Closed EM < 0.55 (worse than baseline)
- Open Token-F1 < 0.18 (worse than baseline)
- Train loss diverges or hits NaN
- Peak GPU memory > 14 GB (won't fit on T4)
- Wall time > 5 hours (something is non-quantized despite `load_in_4bit=true`)

---

## 10. Cross-method analysis (Member 3)

Member 3's main job (Weeks 9–10) is the cross-method comparison. The data is already in place:

### Inputs

```
results/baseline/per_example_scores.json
checkpoints/lora_quick/per_example_scores.json     # Member 1 — pilot
checkpoints/lora_rank4/per_example_scores.json     # Member 1 — best LoRA
checkpoints/lora_rank8/per_example_scores.json     # Member 1
checkpoints/lora_rank16/per_example_scores.json    # Member 1
checkpoints/qlora_rank4/per_example_scores.json    # Member 2
checkpoints/qlora_rank8/per_example_scores.json    # Member 2
checkpoints/qlora_rank16/per_example_scores.json   # Member 2
checkpoints/dora_rank4/per_example_scores.json     # Member 2
checkpoints/dora_rank8/per_example_scores.json     # Member 2
checkpoints/dora_rank16/per_example_scores.json    # Member 2
```

### Required outputs

1. **Cross-method headline table** — extend Table 1 in `results/tables/main_results.md` to include all 11 rows (baseline + 10 PEFT runs)
2. **Cross-method significance table** — extend Table 2 with paired bootstrap CIs for:
   - QLoRA vs. baseline, QLoRA vs. LoRA at matched rank
   - DoRA vs. baseline, DoRA vs. LoRA at matched rank, DoRA vs. QLoRA at matched rank
3. **Updated figures** in `results/figures/`:
   - `rank_scaling.pdf` — extend with QLoRA / DoRA curves
   - `improvements_bar.pdf` — extend with all PEFT methods
   - NEW: `efficiency_pareto.pdf` — accuracy vs. peak GPU memory scatter
4. **Qualitative analysis update** — extend `results/error_analysis/baseline_vs_lora_rank4.json` to include QLoRA / DoRA wins

### How to wire it together

Use `notebooks/03_lora_experiments.ipynb` as a template. Each cell already does the per-method analysis; you just need to add a final cell that aggregates all methods into one table.

---

## 11. Building the Gradio demo (Member 3)

The trained adapter (whichever performs best — likely `lora_rank4`) is the demo's centerpiece.

### Skeleton

Create `src/demo/app.py`:

```python
import gradio as gr
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Use the best-performing adapter — likely lora_rank4 from Member 1
ADAPTER_PATH = "checkpoints/lora_rank4"

base = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="float16",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    min_pixels=256*28*28, max_pixels=768*28*28,  # IMPORTANT — see § 12
)

def predict(image, question):
    from src.evaluation.evaluate_baseline import generate_answer
    return generate_answer(model, processor, image, question)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Radiology image"),
        gr.Textbox(label="Question"),
    ],
    outputs=gr.Textbox(label="Model answer"),
    title="LoRA-tuned Qwen2-VL on radiology VQA",
    description="Type a question about the uploaded radiology image.",
    examples=[  # pre-loaded examples for the demo video
        # add 3-5 example (image, question) pairs from the test set
    ],
)
demo.launch(share=True)
```

### Important notes

- `min_pixels=256*28*28, max_pixels=768*28*28` **must match** training-time settings (see § 12 below for why).
- The demo reads `lora_predictions.jsonl` from Member 1's results to pre-populate examples. Pick 3-5 cases where LoRA dramatically beats the baseline (look in `results/error_analysis/baseline_vs_lora_rank4.json` for the best demos).

---

## 12. Known gotchas Member 1 hit

These bugs cost Member 1 hours. **Don't re-discover them.**

### Gotcha 1: Image-token-count mismatch with Qwen2-VL

**Symptom:** Training crashes mid-step with a tensor shape mismatch error mentioning `image_pad` tokens or `grid_thw`.

**Root cause:** Qwen2-VL uses Naive Dynamic Resolution — it computes how many image tokens to insert based on the image's resolution. If the processor isn't pinned to specific min/max pixel counts, the count won't be a multiple of merge_size=2, and the placeholder count won't match the visual feature count.

**Fix (already in `src/training/train_lora.py`):**

```python
processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256*28*28,
    max_pixels=768*28*28,
)
```

**Member 1's lesson:** Same settings are used in `src/evaluation/evaluate_baseline.py` for the baseline. **You must use the same settings in your demo, otherwise inference results won't match training distribution.**

### Gotcha 2: gradient_checkpointing + LoRA = no input grads

**Symptom:** Training proceeds but loss never decreases. Or, you see a warning "use_reentrant: ... but no input has requires_grad=True".

**Root cause:** When PEFT wraps the model, it freezes the base parameters. `gradient_checkpointing` then has no input gradients to flow through.

**Fix (already in `src/training/_enable_input_grads.py`):** After `get_peft_model()`, call `enable_input_grads(model)` which registers a forward hook to set `requires_grad=True` on the input embeddings. The pipeline does this automatically.

**Member 1's lesson:** If you write a custom training loop, you'll need to call this helper too.

### Gotcha 3: Processor settings drift between training and eval

**Symptom:** LoRA training succeeds with high accuracy, but at inference time predictions are garbage.

**Root cause:** If you `AutoProcessor.from_pretrained()` separately for training and inference and forget to set `min_pixels`/`max_pixels` in one place, the image will be tokenized differently and the model will see out-of-distribution inputs.

**Fix:** Always set `min_pixels=256*28*28, max_pixels=768*28*28` whenever you load a processor (training, eval, demo, anywhere).

### Gotcha 4: Token-level loss masking

**Symptom:** Model "memorizes" the system prompt instead of learning the task.

**Root cause:** Default supervised-finetuning collators compute loss on all tokens. We want loss only on the answer tokens.

**Fix (already in `src/training/data_collator.py`):** `QwenVLSFTCollator` masks the prompt portion of `labels` with -100 (ignored by `CrossEntropyLoss`).

**Member 1's lesson:** Don't change `data_collator.py`. If you do, all methods need re-running.

---

## 13. Common errors and fixes

### Out-of-memory (OOM) on Colab T4

**Symptom:** `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Fixes (try in order):**

1. **Reduce `eval_max_new_tokens`** in your config from 64 to 48 (eval pass uses memory too)
2. **Set `per_device_batch_size: 1` and increase `gradient_accumulation_steps`** to 8 or 16 (effective batch size is unchanged, peak memory drops)
3. **Use QLoRA instead of LoRA** — `load_in_4bit: true` cuts memory ~50%
4. **Restart runtime** — sometimes OOM happens because of leaked memory from previous runs. Runtime → Restart.

### NaN loss

**Symptom:** Loss becomes NaN partway through training.

**Causes & fixes:**
- Learning rate too high → drop from 1e-4 to 5e-5
- bf16 instability with Qwen2-VL — stick with fp16 (Member 1's default)
- Bad data → check `train_max_examples: 5` first to verify data flow

### Training is extremely slow

**Symptom:** Each epoch takes >2 hours instead of ~1 hour.

**Causes & fixes:**
- `gradient_checkpointing: false` accidentally — turn it back on (saves memory at small speed cost — actually goes faster on T4 because we avoid OOM-induced swapping)
- Logging too frequently — set `logging_steps: 25` instead of `5`
- Eval running every epoch — set `eval_strategy: no` if you only need post-training eval

### Model loads but predictions are garbage

**Symptom:** Closed EM ~0.30 (worse than baseline).

**Cause:** Adapter not actually loaded. Verify:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base, "checkpoints/qlora_rank8")
model.print_trainable_parameters()  # should show ~9M trainable
```

If it shows 0 trainable parameters, the adapter directory is empty or `safetensors` file is missing.

### "ImportError: cannot import name 'paired_bootstrap_ci_diff'"

**Cause:** You're not at the repo root, or `src/` isn't on `sys.path`.

**Fix:**
```python
import sys, os
sys.path.insert(0, '/content/peft-medical-vqa')  # or wherever
os.chdir('/content/peft-medical-vqa')
```

---

## 14. What you must NOT change

To keep cross-method results comparable, do not modify:

- **`src/data/vqarad_dataset.py`** — especially `SYSTEM_PROMPT` and `build_qwen_prompt`. Different prompt formatting invalidates the comparison.
- **`src/training/data_collator.py`** — especially the prompt-masking logic in the labels tensor.
- **`src/evaluation/metrics.py`** — `normalize_text()` and `compute_all_metrics()`.
- **`min_pixels=256*28*28, max_pixels=768*28*28`** in any processor instantiation.
- **`seed: 42`** in any config (unless we agree as a team to a multi-seed run).

If you find a real bug in any of these, **open a GitHub issue or PR** instead of silently editing — Member 1 will re-run baseline + LoRA so we're all on the same code.

---

## 15. Commit conventions

To keep `git log` readable for the final-report appendix (rubric explicitly grades on "Proper version control and commit history"):

```
[memberN] <area>: <imperative summary>

Optional longer body explaining *why* the change was needed.
```

Examples:

```
[member1] data: add VQA-RAD HuggingFace loader and qtype classifier
[member1] eval: implement bootstrap CI and McNemar exact test
[member1] train: add LoRA fine-tuning pipeline with rank-ablation configs
[member2] qlora: enable 4-bit NF4 via load_in_4bit flag
[member2] dora: add use_dora=True config
[member2] sweep: rank ablation for QLoRA r∈{4,8,16}
[member3] analysis: cross-method LoRA / QLoRA / DoRA comparison
[member3] demo: Gradio app for best-performing adapter
```

Areas: `data`, `eval`, `train`, `docs`, `tests`, `ci`, `demo`, `qlora`, `dora`, `sweep`, `report`, `analysis`.

Imperative present tense. Lowercase summary. One feature per commit.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for full branch / PR guidelines.

---

## 16. Final-results checklist

Before you push your final results to `main`, verify for each method (QLoRA / DoRA):

- [ ] `training_metrics.json` exists in the checkpoint directory
- [ ] `per_example_scores.json` exists and `len(d['ids']) == 451`
- [ ] `lora_predictions.jsonl` exists and has 451 lines
- [ ] `params.trainable` is populated and reasonable (~1–20M for r=4–32)
- [ ] `training.peak_gpu_gb` is populated and reasonable (3–15 GB)
- [ ] You ran the paired-bootstrap CI vs. baseline AND vs. LoRA r=4 (best) AND vs. LoRA at matched rank, recorded all three
- [ ] Numbers fall in the expected ranges in § 9 (or you have a clear explanation)
- [ ] Commit message follows the convention in § 15
- [ ] All 85 tests still pass: `pytest tests/ -q`
- [ ] You updated `docs/MEMBER1_REPORT.md` style of documentation for your section (if applicable)

---

**Questions?** Open a GitHub issue and tag @qz2579 (Member 1). Don't silently edit — discuss first.

Good luck. The pipeline is robust and the numbers are reproducible. 🚀
