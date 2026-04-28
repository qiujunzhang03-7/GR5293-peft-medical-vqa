# QLoRA: Efficient Finetuning of Quantized LLMs

## Citation

```bibtex
@inproceedings{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and
          Zettlemoyer, Luke},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023},
  url={https://arxiv.org/abs/2305.14314}
}
```

arXiv: **2305.14314** (May 2023; NeurIPS 2023).

## One-paragraph summary

QLoRA is a memory-saving training recipe that combines (a) loading the
frozen base model in **4-bit precision** using a custom data type called
**NF4 (NormalFloat-4)**, (b) **double quantization** of the quantization
constants themselves to claw back another ~0.4 bits/param, and (c)
**paged optimizers** that spill optimizer states to CPU RAM under
pressure. With these three tricks layered on top of vanilla LoRA,
the authors fine-tune a 65 B-parameter LLaMA on a single 48 GB GPU
and obtain a chatbot ("Guanaco") that hits 99.3% of ChatGPT performance
on the Vicuna benchmark — a result that previously would have required
a multi-GPU cluster. The key empirical claim is that 4-bit quantization
**preserves full 16-bit fine-tuning task performance** when paired
with low-rank adapters that compensate for the quantization noise.

## Core mechanism

Three orthogonal contributions stacked together:

1. **NF4 (NormalFloat-4)** — a 4-bit data type whose levels are placed at
   the quantiles of a unit Gaussian. Since LLM weights empirically
   follow approximately Gaussian distributions, NF4 is information-
   theoretically optimal in a way that uniform 4-bit ints / floats are not.
2. **Double quantization** — the per-block scaling constants used by
   block-wise quantization are themselves quantized (8-bit → 8-bit), saving
   ~0.37 bits/param on average. For a 65 B model this is ~3 GB.
3. **Paged optimizers** — uses NVIDIA unified memory to swap optimizer
   states (Adam needs 2 extra fp32 tensors per parameter) to CPU when
   GPU memory is tight. Avoids OOM crashes during long-sequence batches.

The LoRA adapters themselves are trained in bf16; gradients flow back
through the dequantization op into the adapters, while the 4-bit base
weights remain frozen.

## Reported results (selection)

| Setting | VRAM (GPU) | Vicuna benchmark |
|---------|----------:|-----------------:|
| LLaMA-65B full fine-tuning (16-bit) | 780 GB | (baseline) |
| LLaMA-65B QLoRA (4-bit + LoRA) | 48 GB | 99.3% of ChatGPT |
| LLaMA-13B QLoRA | 14 GB | 92.4% of ChatGPT |

The headline practical claim: **QLoRA matches the task quality of 16-bit
LoRA**, while reducing VRAM by ~3×.

## Relevance to our project

QLoRA is one of our three compared methods, and it's the one most
directly motivated by our resource constraints. We're training on a
Colab T4 (16 GB) — exactly the regime where the difference between LoRA
and QLoRA matters. Specifically:

1. **RQ2** (efficiency) — QLoRA should give us measurably lower peak GPU
   memory than LoRA at similar accuracy, validating the paper's central
   claim in the medical-VQA setting.
2. **Hyperparameter sensitivity** — the paper tests bnb-4bit-quant-type
   {`fp4`, `nf4`} and finds NF4 better; we'll verify this on Qwen2-VL-2B.
3. **Negative claim to test** — the paper claims "no degradation". A
   recent clinical-data-extraction study (Stark et al., 2025, PMC12633606)
   reported QLoRA ≈ LoRA accuracy but **28-32% slower training** due to
   dequantization overhead. Our timing measurement should detect this
   trade-off.

## Open questions / caveats

- Most QLoRA validation is on **text-only** decoder LLMs (LLaMA, GPT-NeoX).
  Vision-language models with frozen vision encoders are less explored;
  the encoder is generally not quantized, so the memory savings are
  smaller than for pure-text models of comparable size.
- "No degradation" is shown on instruction-following benchmarks; whether
  it holds for narrow specialty tasks (like radiology VQA) where the
  base model's prior is weak is one of the open questions our project
  can contribute evidence on.
