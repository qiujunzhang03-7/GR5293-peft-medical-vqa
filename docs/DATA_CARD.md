# Data Card: VQA-RAD

> One-page summary of the dataset our project uses. Adapted from the
> "Datasheet for Datasets" template (Gebru et al., 2021), trimmed to the
> fields most relevant for a course project.
>
> **Team:** Qiujun Zhang (qz2579), Wanrong Dang (wd2423), Longkun Xu (lx2358)
> **Course:** STAT GR5293 — Generative AI (Spring 2026), Columbia University
> **Primary use:** Single training & evaluation dataset for our LoRA / QLoRA / DoRA / Q-DoRA comparison on Qwen2-VL-2B-Instruct. All 11 PEFT configurations + 3 target-module ablations reported in `REPORT.md` are trained and evaluated on the splits described below.

## Identity

| Field | Value |
|-------|-------|
| Name | VQA-RAD (Visual Question Answering on Radiology) |
| Source | [`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) on HuggingFace |
| Original publication | Lau et al. (2018), *Scientific Data* 5:180251 — [DOI](https://doi.org/10.1038/sdata.2018.251) |
| License | CC0 1.0 Universal (public domain dedication) |
| Domain | Medical visual question answering — radiology |

## Composition

| Property | Value |
|----------|------:|
| Total QA pairs | **2,248** |
| Train pairs | **1,797** |
| Test pairs | **451** |
| Unique images | **315** |
| Modalities | CT, MRI, X-ray (chest / head / abdominal) |
| Image source | MedPix (open-access NIH radiology image database) |
| Image format | PIL.Image, variable resolution and mode (RGB or L) |
| Question/answer language | English (lowercase) |

### Question type breakdown

| Split | Closed (yes/no) | Open-ended | Closed % |
|-------|----------------:|-----------:|---------:|
| Train | ~1019 | ~778 | ~56.7% |
| Test | 251 | 200 | 55.7% |

(Exact numbers reproducible by running `notebooks/01_data_exploration.ipynb`.)

### Length statistics

| Split | Median Q words | 99th-pct Q words | Median A words | 99th-pct A words |
|-------|---------------:|-----------------:|---------------:|-----------------:|
| Train | ~7 | ~17 | 1 | ~9 |
| Test | ~7 | ~16 | 1 | ~9 |

The 99th percentile of answer length is well under 64 tokens, justifying
our `max_new_tokens = 64` choice.

## Collection process

- Clinicians (radiologists / medical students) wrote both the questions
  and the reference answers, looking at each MedPix image.
- Questions span yes/no, organ identification, modality, abnormality
  detection, and attribute description.
- Answers are short, free-form English. No multiple-choice options are
  provided.

## Splits

- The official train/test split is **by question**, not by image. The
  same image can appear in both splits paired with different questions.
- We verified train/test image overlap empirically in
  `notebooks/01_data_exploration.ipynb`.
- We use the official split unchanged — this is the convention adopted
  by all VQA-RAD baselines (LLaVA-Med, BiomedGPT, PubMedCLIP).

## How we use it

We use the dataset across all three project phases. The same train/test split is used for every PEFT configuration to keep comparisons apples-to-apples.

| Stage | Split | n | Purpose | Owner |
|-------|-------|---:|---------|-------|
| Data exploration | train + test | 2,248 | Length stats, qtype breakdown, image overlap analysis (`notebooks/01_data_exploration.ipynb`) | Member 1 |
| Zero-shot baseline | test | 451 | Evaluate Qwen2-VL-2B with no fine-tuning; produces `results/baseline/per_example_scores.json` for paired tests | Member 1 |
| LoRA quick-run pilot | train (200) → test | 200 / 451 | Validate the SFT pipeline end-to-end in ~6 minutes | Member 1 |
| LoRA r ∈ {4, 8, 16} ablation | train → test | 1,797 / 451 | 3-rank LoRA sweep on the full training split (3 epochs each) | Member 1 |
| QLoRA r ∈ {4, 8, 16} ablation | train → test | 1,797 / 451 | Mirror the LoRA sweep with 4-bit NF4 base quantization | Member 2 |
| DoRA r ∈ {4, 8, 16} ablation | train → test | 1,797 / 451 | Mirror the LoRA sweep with weight-magnitude decomposition | Member 2 |
| Q-DoRA r=8 | train → test | 1,797 / 451 | Combined 4-bit + DoRA — checks whether DoRA recovers the QLoRA accuracy gap | Member 2 |
| Target-module ablation (LoRA r=8, attn-only / qv-only / FFN-only) | train → test | 1,797 / 451 | Isolate which Qwen2-VL submodules drive the gain | Member 2 |
| Cross-method analysis | test | 451 | Aggregate the 11 PEFT runs + baseline into headline tables, paired statistical tests, and figures | Member 3 |
| Gradio demo | (live image inputs at inference time) | — | Interactive demo of the best checkpoint (DoRA r=4) | Member 3 |
| Final reporting | test | 451 | Compare PEFT methods to the baseline using paired bootstrap CIs and McNemar exact tests | All members |

**Total dataset interactions:** 1 baseline run + 1 LoRA quick run + 9 full PEFT ablation runs + 3 target-module ablations = **14 separate evaluations**, all on the identical 451-example test split. This is what enables the paired statistical tests in `REPORT.md` § 6.3 to compare any two methods on a per-example basis without re-evaluating either side.

## Preprocessing

- **Train side (fine-tuning):** the raw `(image, question, answer)` triple is wrapped into a Qwen2-VL chat template by `src/data/vqarad_dataset.py`. The custom `QwenVLSFTCollator` in `src/training/data_collator.py` then masks the prompt portion of `labels` with `-100` so that the cross-entropy loss is computed only on answer tokens — without this, the model would also learn to predict the system message and user question, which inflates training loss but degrades test performance. No image resizing is applied; Qwen2-VL handles arbitrary resolutions through its dynamic-tile mechanism.
- **Inference / evaluation side:** we convert grayscale (`L`) images to `RGB` so tensor shapes are consistent. We fix `processor.image_processor.min_pixels = 256·28·28` and `max_pixels = 768·28·28` for **both** the zero-shot baseline and every PEFT run — this prevents the visual-token count from drifting between methods, which is the single most important condition for valid cross-method comparison.
- **Reference answers** are normalized for scoring (lowercased, punctuation stripped, whitespace collapsed); see `src.evaluation.metrics.normalize_text`. Articles (a/an/the) are intentionally **not** stripped because they are diagnostically meaningful in radiology phrasing (e.g., "the right lobe" vs. "right lobe").

## Empirical observations from our experiments

These notes record what we learned about the dataset *while* running the 14 evaluations described above. They are useful context for interpreting the headline numbers in `REPORT.md`.

### 1. The baseline's "yes" bias is large and dataset-level

On the closed-ended test subset (n = 251), the zero-shot Qwen2-VL-2B-Instruct baseline answers **"yes" to roughly two-thirds of questions** regardless of image content. Because the test set has slightly more "no" than "yes" gold answers, this single bias accounts for the bulk of baseline errors. PEFT fine-tuning fixes 68 of these 251 closed cases — the largest single source of measured improvement (`REPORT.md` § 8.1). This is a property of the *baseline × dataset interaction*, not a dataset flaw, but the implication is that any closed-ended VQA-RAD score below ~0.55 is consistent with a near-random "always-yes" predictor.

### 2. Closed-ended is genuinely easier than open-ended at our scale

Across **all 11 PEFT configurations** we observe Closed EM > Open Token-F1 by a factor of ~1.3–1.4×, even though both metrics are bounded in [0,1]. This is consistent with the qualitative wins in `REPORT.md` § 8.2: many open-ended questions ("where is the cavitary lesion?") are answered by the *baseline* with a yes/no, so the gain from PEFT comes partly from format learning rather than image understanding. Researchers using VQA-RAD should report Closed EM and Open Token-F1 separately — averaging them masks this gap.

### 3. Train/test image overlap does not trivialize the task

We confirmed in `notebooks/01_data_exploration.ipynb` that the official split is by question, not by image, so the same image can appear in both halves with different questions. We worried this might inflate scores. In practice our DoRA r=4 reaches Overall EM 0.5455 — well below the 0.84 closed-fine-tuned LLaVA-Med ceiling — which suggests image memorization is not the dominant signal. The 451-example test split therefore measures generalization to *new questions about partly familiar images*, which is the realistic clinical setting (a radiologist asking new questions of a known scan).

### 4. The 1,797-example training set overfits at LoRA rank ≥ 8

The U-shaped rank curve (`REPORT.md` § 7.1) is reproducible across **all three** PEFT methods (LoRA, QLoRA, DoRA), which is mild evidence that the cause is dataset size, not algorithm. With 18 M trainable parameters at LoRA r=16 and only 1,797 training examples, the model can effectively memorize the train set (training loss bottoms out near 0.16). Test EM forms a U-shape as capacity grows. Practically: r=4 is a strong default for VQA-RAD-scale medical-VQA datasets; running r=16 buys nothing on this dataset and may hurt.

### 5. Question-type classification is robust

We classify each example as closed or open by checking whether the *normalized* reference answer equals "yes" or "no". Across all 451 test examples this rule matches the dataset's intent perfectly (manual spot-check of 50 random examples agreed in all 50). The same rule is used by every PEFT run via `src/data/load_vqarad.py`, so the 251/200 closed/open split is identical for every reported metric.



## Caveats and limitations

1. **Small test set (n = 451).** A 95% CI for an accuracy near 0.5 is
   approximately ±4.5 percentage points. We mitigate by reporting
   bootstrap CIs and using paired statistical tests (McNemar /
   paired bootstrap) when comparing methods.
2. **Train/test image overlap.** As noted above, the split is by question.
   The test set therefore measures generalization to *new questions
   about partly familiar images*, not fully unseen images.
3. **Annotation noise.** Free-form answers have inconsistent surface forms
   (e.g., "cardiomegaly" vs. "enlarged heart"). We report multiple
   open-ended metrics (BLEU-1, ROUGE-L, Token-F1) rather than relying
   on Exact Match for open-ended answers.
4. **Dataset size discrepancy in the literature.** Some papers cite
   ~3,515 QA pairs based on the original release including duplicates and
   augmented variants; the canonical HuggingFace release contains 2,248
   after de-duplication. We use the HuggingFace release.
5. **Scope.** Radiology only — findings here may not transfer to
   pathology, dermatology, or ophthalmology VQA without re-validation.
6. **Limited statistical power for cross-method comparisons.** With 451
   paired test examples, the minimum detectable Overall-EM difference
   between two PEFT methods at α = 0.05 / 80% power is roughly
   ±3 percentage points. This is precisely the regime where DoRA's small
   numerical advantage over LoRA falls (`REPORT.md` § 6.3) — our
   "DoRA = LoRA in accuracy" conclusion is therefore "the gap is below
   our detection threshold", not "the gap is provably zero". A larger
   medical-VQA benchmark (or running multiple seeds and averaging) would
   be needed to settle the difference.

## Ethical and licensing notes

- The dataset is in the public domain (CC0 1.0).
- Source images come from NIH MedPix and have been de-identified.
- Sharing predictions and processed splits in a public GitHub repo is
  permissible under CC0.
