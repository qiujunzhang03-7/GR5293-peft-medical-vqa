# Data Card: VQA-RAD

> One-page summary of the dataset our project uses. Adapted from the
> "Datasheet for Datasets" template (Gebru et al., 2021), trimmed to the
> fields most relevant for a course project.

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
| Train | ~734 | ~1063 | ~41% |
| Test | ~184 | ~267 | ~41% |

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

| Stage | Split | Purpose |
|-------|-------|---------|
| Zero-shot baseline (Member 1) | test | Evaluate Qwen2-VL-2B without any fine-tuning |
| LoRA / QLoRA / DoRA training (Members 2, 3) | train | Fine-tune the base model |
| Final reporting | test | Compare PEFT methods to the baseline using paired statistical tests |

## Preprocessing

- **None on the train side** during fine-tuning (Members 2 / 3).
- On the inference / evaluation side, we convert grayscale (`L`) images
  to `RGB` so tensor shapes are consistent. Qwen2-VL handles arbitrary
  resolutions natively, so we do not resize.
- Reference answers are normalized for scoring (lowercased, punctuation
  stripped, whitespace collapsed); see `src.evaluation.metrics.normalize_text`.

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

## Ethical and licensing notes

- The dataset is in the public domain (CC0 1.0).
- Source images come from NIH MedPix and have been de-identified.
- Sharing predictions and processed splits in a public GitHub repo is
  permissible under CC0.
