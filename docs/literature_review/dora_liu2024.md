# DoRA: Weight-Decomposed Low-Rank Adaptation

## Citation

```bibtex
@inproceedings{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and
          Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting
          and Chen, Min-Hung},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024},
  note={Oral presentation},
  url={https://arxiv.org/abs/2402.09353}
}
```

arXiv: **2402.09353** (Feb 2024; ICML 2024 Oral).

## One-paragraph summary

DoRA argues that the gap between LoRA and full fine-tuning isn't because
LoRA can't represent the right `ΔW` — it's because LoRA jointly updates
both the *magnitude* and *direction* of each weight column, while full FT
updates them in qualitatively different ways. The fix: decompose each
pretrained weight `W₀ = m · (V / ||V||_c)` into a magnitude vector `m`
and a unit-norm direction `V`, then learn the directional update with
LoRA but the magnitude update directly. The result is that DoRA learns
much closer to the FT pattern, with no extra inference cost (after merging)
and only marginally more trainable parameters. On commonsense reasoning,
visual instruction tuning (LLaVA), and image/video-text tasks, DoRA beats
LoRA at matched parameter budget — sometimes by 2-3 points.

## Core mechanism

Each weight matrix is decomposed columnwise:

```
W = m · (V / ||V||_c),  where m ∈ R^(1×k), V ∈ R^(d×k)
```

Then the parameters trained are:
- `m` directly (a small vector, one scalar per output column)
- `V` *via* a LoRA-style update: `V = W₀ + B A`, with `A, B` low-rank

Final weight at each forward pass:

```
W_DoRA = m · ((W₀ + BA) / ||W₀ + BA||_c)
```

Like LoRA, this can be merged back at deploy time so inference cost is
unchanged. Trainable parameter count = LoRA's parameters + the magnitude
vector (negligible compared to BA).

The key analytical contribution: a **weight decomposition analysis**
that visualizes how full FT and LoRA update direction vs. magnitude. FT
shows a strong negative correlation between magnitude change and direction
change; LoRA does not; DoRA does.

## Reported results (selection)

| Method | Trainable params | LLaMA-7B (commonsense, 8 tasks avg) |
|--------|------------------|------------------------------------:|
| LoRA r=16 | 0.83% | 76.4 |
| DoRA r=16 | 0.84% | **78.4** |
| LoRA r=32 | 1.66% | 77.7 |
| DoRA r=32 | 1.67% | **78.9** |

On VL-BART image captioning and visual instruction tuning (LLaVA-1.5),
DoRA also outperforms LoRA at matched rank.

## Relevance to our project

DoRA is the third PEFT method we compare. It is also the **most recent**
of the three, so its behavior on multimodal medical tasks is the least
characterized — making it the most interesting empirically.

Three specific reasons we care:

1. **RQ1** (PEFT vs baseline) — the paper claims DoRA closes more of the
   FT-vs-LoRA gap; if true, DoRA should give the largest accuracy lift
   over our zero-shot baseline.
2. **RQ2** (efficiency) — DoRA adds the magnitude vector, so trainable
   params are ~0.01% higher than LoRA. Small in absolute terms but worth
   measuring whether it materially shifts training speed or VRAM.
3. **HuggingFace integration** — DoRA is supported in `peft >= 0.10` via
   `LoraConfig(use_dora=True)`, so the engineering cost vs. LoRA is
   essentially zero. This means the experiment is well-posed: any
   accuracy difference can be attributed to the decomposition.

## Open questions / caveats

- The paper's strongest gains are at **small rank** (r=4 to r=16). At
  higher ranks LoRA closes the gap. We should sweep `r ∈ {4, 8, 16, 32}`
  in the ablation to see where this curve lies for medical VQA.
- DoRA was evaluated mostly on **text-instructed multimodal tasks**
  (LLaVA, VL-BART) using natural-image inputs. Whether the magnitude/
  direction decomposition still helps when the visual content is
  out-of-domain (radiology) is exactly the question our project can
  answer.
- Computational overhead of the column-wise normalization at every
  forward pass is small but non-zero; we should measure training-time
  per epoch alongside the headline accuracy.
