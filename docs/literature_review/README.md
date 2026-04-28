# Literature Review

This folder collects the team's reading notes on the foundational and most
relevant work for this project. Each file is a structured summary of one
paper, written so it can be lifted into the final report's *Related Work*
section with minimal editing.

## Reading list

### Core PEFT methods (the three our project compares)

| Paper | File |
|-------|------|
| Hu et al., 2021 — **LoRA**: Low-Rank Adaptation of Large Language Models | [`lora_hu2021.md`](lora_hu2021.md) |
| Dettmers et al., 2023 — **QLoRA**: Efficient Finetuning of Quantized LLMs | [`qlora_dettmers2023.md`](qlora_dettmers2023.md) |
| Liu et al., 2024 — **DoRA**: Weight-Decomposed Low-Rank Adaptation | [`dora_liu2024.md`](dora_liu2024.md) |

### Domain context: medical VQA & vision-language models

| Paper | File |
|-------|------|
| Lau et al., 2018 — **VQA-RAD** dataset | [`vqarad_lau2018.md`](vqarad_lau2018.md) |
| Li et al., 2023 — **LLaVA-Med**: medical instruction-tuned VLM | [`llavamed_li2023.md`](llavamed_li2023.md) |
| Wang et al., 2024 — **Qwen2-VL** technical report | [`qwen2vl_wang2024.md`](qwen2vl_wang2024.md) |

## How these papers fit the project

The project's RQ1 ("how much does PEFT improve a base VLM?") sits at the
intersection of three lines of work:

1. **PEFT methods themselves** — LoRA established the low-rank-update
   paradigm, QLoRA showed quantization makes it accessible on commodity
   GPUs, and DoRA refined it further by separating magnitude from
   direction. These are the three knobs we directly compare.
2. **Medical VQA evaluation** — VQA-RAD is the standard small-scale
   benchmark in this area; LLaVA-Med is the canonical instruction-tuned
   medical VLM and gives us the literature context for what
   medical-domain PEFT can achieve.
3. **The base VLM** — Qwen2-VL-2B-Instruct is what we fine-tune; its
   technical report tells us what its visual encoder and dynamic-resolution
   handling look like, which informs our prompt design and image
   preprocessing.

## How to read these notes

Each note follows the same structure:

- **Citation** — full BibTeX
- **One-paragraph summary** — what's the paper's claim?
- **Core mechanism** — what's the actual algorithm/contribution?
- **Reported results** — what numbers do they report, on what tasks?
- **Relevance to our project** — why we cite this, and what hypotheses it
  motivates
- **Open questions / caveats** — what the paper doesn't answer, or where
  its claims may not generalize to our setting
