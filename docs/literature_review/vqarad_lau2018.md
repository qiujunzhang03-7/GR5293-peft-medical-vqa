# VQA-RAD: A Dataset of Clinically Generated Visual Questions and Answers about Radiology Images

## Citation

```bibtex
@article{lau2018vqarad,
  title={A dataset of clinically generated visual questions and answers
         about radiology images},
  author={Lau, Jason J. and Gayen, Soumya and Ben Abacha, Asma and
          Demner-Fushman, Dina},
  journal={Scientific Data},
  volume={5},
  number={1},
  pages={180251},
  year={2018},
  publisher={Nature Publishing Group},
  doi={10.1038/sdata.2018.251}
}
```

## One-paragraph summary

VQA-RAD is the first **clinician-curated** medical visual question
answering benchmark. The authors selected 315 radiology images (CT, MRI,
X-ray) from MedPix and asked clinicians to write **both** questions and
free-form answers about each image, capturing the kinds of queries that
arise in actual diagnostic reading. The result is a small but high-quality
dataset of 2,248 question-answer pairs (post-deduplication; the
HuggingFace release uses these counts) split into 1,797 train / 451 test
by question. Questions span yes/no, organ/region identification,
modality, abnormality detection, and attribute description. VQA-RAD has
become the de-facto small-scale benchmark for medical VQA work because
of its clinical authenticity and open license (CC0 1.0).

## Dataset structure

| Property | Value |
|----------|------:|
| Unique images | 315 |
| Total QA pairs | 2,248 |
| Train / test | 1,797 / 451 |
| Imaging modalities | CT, MRI, X-ray (chest, head, abdominal) |
| Image source | MedPix (open-access) |
| Annotators | clinicians (radiologists / med students) |
| License | CC0 1.0 Universal (public domain) |
| Question types | open-ended (~60%), closed-ended yes/no (~40%) |

The split is **by question, not by image** — the same image can appear
in both train and test with different questions. This is the convention
adopted by all subsequent VQA-RAD baselines, so we follow it unchanged.

## Relevance to our project

VQA-RAD is the **dataset** for our experiments. It's the right choice
for this project because:

1. **Size matches PEFT's regime.** Full fine-tuning a 2B-parameter VLM
   on 1,797 examples would catastrophically overfit; PEFT's small
   trainable parameter count (typically <1% of base model) is exactly
   the right capacity for this data scale.
2. **It has a clean closed/open split.** This lets us report metrics
   that are appropriate for each: Exact Match for yes/no, BLEU/ROUGE/F1
   for free-form. Mixed metrics on a heterogeneous test set are a
   common source of confusion in VQA evaluation; VQA-RAD makes the
   structure explicit.
3. **It's the standard.** LLaVA-Med, BiomedGPT, M3AE, PubMedCLIP, and
   essentially every recent medical VLM reports VQA-RAD numbers, giving
   us reference points to sanity-check against.
4. **Open license** means we can share predictions and processed splits
   in our GitHub repo without legal complications.

## Caveats and statistical considerations

- **Small test set (n=451).** The 95% CI for an accuracy near 0.5 is
  approximately ±4.5 percentage points. We must report bootstrap CIs
  alongside point estimates and use paired tests (McNemar / paired
  bootstrap) when comparing methods, otherwise we'll mistake noise
  for signal.
- **Image-level overlap between train and test.** Because the split is
  by question, not by image, the test set partially measures
  generalization to *new questions about familiar images* rather than
  fully unseen images. This is a known limitation; we follow the
  community's convention but flag it explicitly in the report.
- **Annotation noise.** Free-form answers are written in natural English
  and have inconsistent surface forms ("cardiomegaly" vs "enlarged
  heart"). This is exactly why we report multiple open-ended metrics
  rather than relying on Exact Match alone.
- **The literature inconsistently reports the dataset size.** Some
  papers cite ~3,515 QA pairs based on the original release that
  included duplicates and augmented variants; the canonical
  HuggingFace release contains 2,248 after de-duplication. We use the
  HuggingFace release.

## Typical published results (for sanity-checking)

| Method | Closed acc | Open acc / F1 |
|--------|-----------:|--------------:|
| MEVF + BAN (Nguyen et al., 2019) | 77.2 | 43.9 |
| PubMedCLIP (Eslami et al., 2023) | 80.0 | 60.1 |
| LLaVA-Med (zero-shot, Li et al., 2023) | ≈65 | ≈30 (varies) |
| LLaVA-Med (fine-tuned, Li et al., 2023) | 84.2 | 61.5 |

These are not directly comparable to ours (different base models, different
metric definitions for "open accuracy") but give an order-of-magnitude
sanity check: zero-shot of a non-medical VLM should be in the 50-65 range
for closed questions, and fine-tuned with PEFT should reach the
mid-70s to low-80s.
