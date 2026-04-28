# LLaVA-Med: A Large Language-and-Vision Assistant for Biomedicine

## Citation

```bibtex
@inproceedings{li2023llavamed,
  title={LLaVA-Med: Training a Large Language-and-Vision Assistant for
         Biomedicine in One Day},
  author={Li, Chunyuan and Wong, Cliff and Zhang, Sheng and
          Usuyama, Naoto and Liu, Haotian and Yang, Jianwei and
          Naumann, Tristan and Poon, Hoifung and Gao, Jianfeng},
  booktitle={Advances in Neural Information Processing Systems
             (NeurIPS) — Datasets and Benchmarks Track},
  year={2023},
  url={https://arxiv.org/abs/2306.00890}
}
```

arXiv: **2306.00890** (June 2023; NeurIPS 2023 D&B).

## One-paragraph summary

LLaVA-Med adapts the general-purpose LLaVA recipe to biomedicine by (1)
mining figure–caption pairs from PubMed Central via PMC-15M, (2) using
GPT-4 to generate self-instruct conversational data from the captions,
and (3) doing a two-stage curriculum: first concept alignment on raw
caption pairs, then end-to-end instruction tuning on the GPT-generated
conversations. The whole training run fits in <15 hours on 8 A100s — the
"in one day" of the title. On VQA-RAD, SLAKE, and PathVQA, LLaVA-Med
substantially outperforms general-domain LLaVA, demonstrating that
domain-specific instruction tuning matters even when the base model
is already a strong VLM.

## Core mechanism

Two training stages (much like LLaVA itself, adapted for biomed):

1. **Stage 1: Concept alignment.** Freeze the LLM and the vision encoder;
   train only the linear projector connecting them on raw
   PMC figure–caption pairs. Goal: extend the vocabulary of aligned
   image-text tokens into the biomedical domain.
2. **Stage 2: Instruction tuning.** Unfreeze the LLM, keep the vision
   encoder frozen, and fine-tune end-to-end on GPT-4-generated
   instruction-following conversations grounded in PMC images.

The base LLM is Vicuna-7B/13B; the vision encoder is CLIP ViT-L/14.
Total trainable parameters in stage 2 = projector + LLM, which is
**not** parameter-efficient — this is full fine-tuning of the LLM
component.

## Reported results (selection)

VQA-RAD, closed-ended accuracy:

| Model | Closed | Open (recall) |
|-------|-------:|--------------:|
| LLaVA general | ≈58 | ≈14 |
| LLaVA-Med (Stage 1 only) | ≈61 | ≈25 |
| LLaVA-Med (full) | **84.2** | **61.5** |
| Prior SOTA M3AE | 83.3 | 67.2 |

Trains in 15 hours on 8 A100s — comparable to but cheaper than full
fine-tuning of vanilla LLaVA on the same data.

## Relevance to our project

LLaVA-Med is the **closest published comparison** to what we're trying
to do. Specifically:

1. **It is a "non-PEFT upper bound."** LLaVA-Med fine-tunes the LLM
   side end-to-end. If our LoRA/QLoRA/DoRA experiments on Qwen2-VL-2B
   approach LLaVA-Med's reported numbers, that's evidence PEFT can
   match full FT in the medical-VQA regime — a finding directly
   responsive to RQ1.
2. **It defines the scoring conventions** we adopt. The "closed" /
   "open" split and the choice of recall-style metrics for open-ended
   answers come from this paper's evaluation protocol; we mirror them.
3. **It is a candidate base model in our proposal.** Our proposal lists
   LLaVA-Med as a fallback base model. We chose Qwen2-VL-2B as the
   main one because it's smaller (fits Colab T4), but the LLaVA-Med
   numbers tell us what's reachable on this dataset.

## Open questions / caveats

- LLaVA-Med uses **full** LLM fine-tuning, not PEFT. Our project tests
  the implicit hypothesis that PEFT can recover most of LLaVA-Med-level
  gains at ~1% of the trainable parameters. The literature has not
  systematically tested this on VQA-RAD with a 2B-class base model.
- The reported numbers depend heavily on whether the test set is
  held out at the **image** or **question** level. LLaVA-Med uses the
  official VQA-RAD split (by question), same as us, so the numbers
  are roughly comparable.
- LLaVA-Med-1.5 (an updated checkpoint, released in 2024) reports
  somewhat different numbers using a stronger CLIP-336 visual encoder.
  We cite the original v1 here; the v1.5 numbers are useful for
  cross-checking but not required to cite.
