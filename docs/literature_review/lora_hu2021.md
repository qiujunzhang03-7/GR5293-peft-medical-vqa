# LoRA: Low-Rank Adaptation of Large Language Models

## Citation

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and
          Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  url={https://arxiv.org/abs/2106.09685}
}
```

arXiv: **2106.09685** (June 2021; ICLR 2022).

## One-paragraph summary

LoRA freezes the pretrained weights of a transformer and injects a pair of
trainable low-rank matrices `A ∈ R^(r×k)`, `B ∈ R^(d×r)` into each target
linear layer, so the effective update is `ΔW = BA` with rank at most `r`.
With `r` chosen small (typically 4–16), the trainable parameter count is
orders of magnitude smaller than full fine-tuning, GPU memory drops
dramatically, and inference latency is unchanged because `BA` can be
merged back into the frozen weight at deploy time. On RoBERTa, DeBERTa,
GPT-2, and GPT-3 the authors show LoRA matches or beats full fine-tuning
across GLUE and downstream generation tasks while training ≈10000× fewer
parameters on GPT-3.

## Core mechanism

For a frozen weight `W₀ ∈ R^(d×k)`, the forward pass becomes
`h = W₀ x + (α/r) · B A x` where `α` is a fixed scaling constant.
Initialization: `A` Gaussian, `B = 0`, so the network starts identical
to the base model. Only `A` and `B` are trained; `W₀` stays frozen.

Two important practical choices:

- **Where to apply LoRA** — the original paper applies it to the query
  and value projection matrices `W_q, W_v` of every attention layer.
  Subsequent work (and the HuggingFace PEFT default) often applies it
  to all four attention projections plus MLP.
- **Rank `r`** — the paper studies `r ∈ {1, 2, 4, 8, 64}` and finds that
  even `r = 1` works surprisingly well; the key claim is that the
  required update is "intrinsically low-rank".

## Reported results (selection)

| Model | Trainable params | GLUE / WikiSQL / etc. |
|-------|------------------|-----------------------|
| GPT-3 175B full FT | 175 B | baseline |
| GPT-3 175B LoRA r=8 | 4.7 M | matches or beats full FT |
| RoBERTa-large LoRA | 0.3 M | matches full FT on GLUE |

Memory savings on GPT-3 175B: VRAM dropped from 1.2 TB to 350 GB.

## Relevance to our project

LoRA is the **direct ancestor and one of three methods we compare**.
Specifically, it:

1. Defines the low-rank-update paradigm that QLoRA and DoRA both extend.
2. Sets the hyperparameter family `(r, α, target_modules)` that we will
   sweep in the ablation phase.
3. Provides the conceptual benchmark — if our LoRA results on medical VQA
   are wildly out of line with what one expects from this paper's
   experiments on natural-domain text, that's a signal to debug.

## Open questions / caveats

- LoRA was developed and validated almost entirely on **language-only**
  models; its behavior on the visual side of a VLM (i.e. modulating the
  visual encoder vs the LLM-side projector vs the LLM) is much less
  studied.
- The paper reports task accuracy only; it does not deeply analyze
  *which* layers benefit most from LoRA. Member 1's rank ablation (r ∈ {4, 8, 16})
  partially fills this gap; Member 2's QLoRA experiments will extend it.
- The "intrinsic low rank" claim assumes the target task is a small
  perturbation of the pretrained distribution. Medical VQA may not be —
  radiology is far from web text, so even if r=4 works for GLUE, we may
  need higher r here.
