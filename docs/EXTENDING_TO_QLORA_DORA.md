# Extending the Pipeline to QLoRA and DoRA

> **Audience:** Team members 2 (Qiujun Zhang) and 3 (Wanrong Dang)
> **Author:** Member 1 (Longkun Xu) — Week 4 hand-off
>
> This document is the bridge between Week 1–4 (Member 1's pipeline) and
> Week 5–8 (Member 2 & 3's PEFT experiments). The pipeline is designed
> so that switching between LoRA / QLoRA / DoRA requires **no code
> changes** — only a different YAML config.

---

## 1. The single source of truth

Everything goes through the same training entry point:

```bash
python -m src.training.train_lora --config configs/<your_config>.yaml
```

The script in `src/training/train_lora.py` (`train_lora`) reads the YAML,
builds the model + LoRA adapters, runs the training loop, and saves the
adapter + metrics + predictions. **You should not need to modify
`train_lora.py` itself.** If you do, propose the change in a PR so it
applies to all of us — diverging implementations break the comparison.

---

## 2. How to run QLoRA (Member 2)

### Step 1 — Copy the config

```bash
cp configs/lora_full.yaml configs/qlora_full.yaml
```

### Step 2 — Edit the config

Open `configs/qlora_full.yaml` and change exactly two lines:

```yaml
load_in_4bit: true               # was: false
output_dir: "checkpoints/qlora_full"   # was: "checkpoints/lora_full"
```

Everything else (rank, target_modules, learning rate, etc.) stays the
same so the comparison is apples-to-apples.

### Step 3 — Train

```bash
python -m src.training.train_lora --config configs/qlora_full.yaml
```

The pipeline automatically:
* Loads Qwen2-VL-2B in 4-bit NF4 quantization
* Wraps it with `prepare_model_for_kbit_training`
* Applies LoRA adapters in fp16 on top of the quantized base
* Records peak GPU memory (will be ~40% lower than fp16 LoRA)

### Step 4 — Statistically compare to LoRA

```python
from src.evaluation.statistical_tests import paired_bootstrap_ci_diff, mcnemar_test
import json

with open('checkpoints/lora_full/per_example_scores.json') as f:
    lora = json.load(f)
with open('checkpoints/qlora_full/per_example_scores.json') as f:
    qlora = json.load(f)
assert lora['ids'] == qlora['ids']

diff = paired_bootstrap_ci_diff(lora['correct'], qlora['correct'], seed=42)
mc   = mcnemar_test(lora['correct'], qlora['correct'])
print(f"QLoRA - LoRA EM diff: {diff['point']:+.4f} 95% CI [{diff['lower']:+.4f}, {diff['upper']:+.4f}]")
print(f"McNemar p-value: {mc['p_value']:.4g}")
```

**Hypothesis to test (RQ1+RQ2):** QLoRA's accuracy is statistically
indistinguishable from LoRA's, and QLoRA's peak GPU memory is
substantially lower. The paired bootstrap CI's lower bound on
`QLoRA − LoRA` should *not* exclude zero in the negative direction —
that is, QLoRA should not be significantly worse.

### Step 5 — Hyperparameter sweep (rank ablation, RQ3)

```bash
for r in 4 8 16 32; do
    python -m src.training.train_lora \
        --config configs/lora_full.yaml \
        --rank $r \
        --output_dir checkpoints/lora_r${r}
done
```

Then plot Exact Match vs. `r`. Expected pattern: monotonically
increasing through r=16, then plateauing.

---

## 3. How to run DoRA (Member 3)

### Step 1 — Copy the config

```bash
cp configs/lora_full.yaml configs/dora_full.yaml
```

### Step 2 — Edit one line

```yaml
use_dora: true                   # was: false
output_dir: "checkpoints/dora_full"
```

### Step 3 — Train

```bash
python -m src.training.train_lora --config configs/dora_full.yaml
```

The HuggingFace PEFT library handles DoRA natively when
`use_dora=True` is set on `LoraConfig`. No additional library needed.

### Step 4 — Compare

Same paired bootstrap pattern as above. The interesting comparisons are:
* DoRA vs. baseline (RQ1: does decomposition help?)
* DoRA vs. LoRA (the headline DoRA paper claim — does decomposition
  beat plain LoRA at matched rank?)

### Step 5 — Build the demo (Week 11–12)

The trained DoRA adapter is the demo's centerpiece. The pipeline already
saves predictions in JSON-Lines format ready for a Gradio app:

```python
# Pseudocode for src/demo/app.py (not yet written — your task)
import gradio as gr
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

base = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="float16", device_map="auto"
)
model = PeftModel.from_pretrained(base, "checkpoints/dora_full")
processor = AutoProcessor.from_pretrained("checkpoints/dora_full")

def predict(image, question):
    from src.evaluation.evaluate_baseline import generate_answer
    return generate_answer(model, processor, image, question)

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Question")],
    outputs=gr.Textbox(label="Model answer"),
    title="DoRA-tuned Qwen2-VL on radiology VQA",
)
demo.launch()
```

---

## 4. The four metrics every method must report

For RQ2 (efficiency), all of us must extract the same four numbers from
each `training_metrics.json`. The pipeline writes them automatically:

| Field | Where in `training_metrics.json` |
|-------|----------------------------------|
| Trainable parameters | `params.trainable` |
| Trainable % of total | `params.trainable_pct` |
| Peak GPU memory (allocated, GB) | `peak_gpu_memory_gb` |
| Mean seconds / epoch | `training.mean_epoch_seconds` |

These plug directly into the headline efficiency table in the final
report. Do not compute them by hand — the pipeline already records them
identically across methods.

---

## 5. Commit conventions

To keep `git log` readable for the final-report appendix, please follow:

```
[member1] data: add VQA-RAD loader
[member1] eval: add bootstrap CIs and McNemar test
[member1] train: LoRA pipeline + quick-run config
[member2] qlora: add 4-bit NF4 config
[member2] sweep: rank ablation r∈{4,8,16,32}
[member3] dora: add use_dora=True config
[member3] demo: Gradio app for DoRA-tuned model
```

One feature per commit, present-tense, lowercase. The TA likely *will*
look at commit history (rubric explicitly mentions "Proper version
control and commit history").

---

## 6. What NOT to change

To keep results comparable across methods, do not alter:

* `src/data/vqarad_dataset.py` — especially `SYSTEM_PROMPT` and
  `build_qwen_prompt`. Different prompt formatting invalidates the
  comparison.
* `src/training/data_collator.py` — especially the prompt-masking logic
  in the labels tensor. If you change this, all methods need re-running.
* `src/evaluation/metrics.py` — the `normalize_text` function and the
  `compute_all_metrics` interface.
* The `seed` field in any config — leave at 42 unless we agree to a
  multi-seed run.

If you discover a bug in any of these, **open an issue / PR** rather
than silently editing — Member 1 will re-run the baseline so all of us
are on the same prompt.

---

## 7. Checklist before you push your final results

For each method (LoRA full / QLoRA / DoRA), verify:

- [ ] `training_metrics.json` exists in the checkpoint directory
- [ ] `per_example_scores.json` exists and has 451 entries
- [ ] `lora_predictions.jsonl` exists and has 451 lines
- [ ] `params.trainable` is populated and reasonable (~1–10M for r=8–32)
- [ ] `peak_gpu_memory_gb` is populated and reasonable (3–15 GB)
- [ ] You ran the paired-bootstrap CI vs. baseline and recorded the result
- [ ] Commit message follows the convention in § 5
