## Table 1: Main Results — Qwen2-VL-2B on VQA-RAD test split (n=451)

| Method | Trainable params | Closed EM (n=251) | Open Token-F1 (n=200) | Overall EM (n=451) |
|---|---:|---:|---:|---:|
| Zero-shot baseline | 0 (0.00%) | 0.5657 [0.5060, 0.6255] | 0.2008 [0.1508, 0.2540] | 0.3792 [0.3348, 0.4257] |
| LoRA quick (n_train=200, attn-only) | 2,179,072 (0.0985%) | 0.5339 [0.4741, 0.5976] | 0.2795 [0.2251, 0.3368] | 0.3814 [0.3370, 0.4279] |
| **LoRA full r=4 (best)** | 4,616,192 (0.2085%) | 0.7570 [0.7012, 0.8088] | 0.3561 [0.2962, 0.4166] | 0.5432 [0.4967, 0.5876] |
| LoRA full r=8 | 9,232,384 (0.4162%) | 0.7371 [0.6813, 0.7888] | 0.3223 [0.2633, 0.3831] | 0.5211 [0.4767, 0.5654] |
| LoRA full r=16 | 18,464,768 (0.8290%) | 0.7371 [0.6813, 0.7928] | 0.3561 [0.2957, 0.4172] | 0.5344 [0.4878, 0.5809] |

Numbers in brackets are 95% bootstrap CIs (10,000 resamples).

## Table 2: Statistical Significance — Improvement over zero-shot baseline (paired tests, same 451 examples)

| Method | Closed EM Δ | Open ROUGE-L Δ | Open BLEU-1 Δ | Overall EM Δ | McNemar p (Overall) |
|---|---:|---:|---:|---:|---:|
| LoRA full r=4 | +0.1912 [+0.1235, +0.2590] *** | +0.1543 [+0.1037, +0.2072] *** | +0.1533 [+0.1032, +0.2057] *** | +0.1641 [+0.1197, +0.2106] *** | 0.000000 *** |
| LoRA full r=8 | +0.1713 [+0.0956, +0.2430] *** | +0.1195 [+0.0669, +0.1738] *** | +0.1246 [+0.0730, +0.1779] *** | +0.1419 [+0.0953, +0.1885] *** | 0.000000 *** |
| LoRA full r=16 | +0.1713 [+0.0956, +0.2470] *** | +0.1533 [+0.1032, +0.2070] *** | +0.1557 [+0.1071, +0.2078] *** | +0.1552 [+0.1086, +0.2040] *** | 0.000000 *** |

CIs from 10,000-resample paired bootstrap on per-example score differences.
McNemar p-values from exact test on Overall EM. *** = p < 0.001.
