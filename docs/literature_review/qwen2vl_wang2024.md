# Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution

## Citation

```bibtex
@article{wang2024qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the
         World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and
          Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and
          Wang, Jialin and Ge, Wenbin and others},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024},
  url={https://arxiv.org/abs/2409.12191}
}
```

arXiv: **2409.12191** (Sept 2024).

## One-paragraph summary

Qwen2-VL is Alibaba's open-weight vision-language model family
(2B / 7B / 72B parameters), with a key architectural innovation: the
visual encoder uses **Naive Dynamic Resolution** plus **Multimodal
Rotary Position Embedding (M-RoPE)**, allowing it to ingest images at
their native aspect ratio and resolution rather than fixed 336×336 or
224×224 squares. This matters for downstream tasks where small details
or unusual aspect ratios carry information — including medical imaging,
where chest X-rays are tall, abdominal CT slices are square, and
pathology images vary widely. The 2B model is competitive with
much larger general-purpose VLMs on standard benchmarks and is
small enough to fine-tune on a single consumer GPU.

## Core mechanism (the parts we care about)

- **Architecture**: a 675M-parameter visual encoder (modified ViT) +
  a Qwen2 LLM backbone (1.5B for the 2B variant; 7B for the 7B variant).
- **Naive Dynamic Resolution** — instead of resizing every input image
  to a fixed grid, the encoder accepts the image's native size (within
  configurable min/max-pixel bounds) and produces a variable-length
  visual token sequence. The minimum-pixel default is 4 × 28 × 28 = 3136
  and the max-pixel default is 16384 × 28 × 28 ≈ 12.8M.
- **M-RoPE** — rotary positional embeddings are extended to use three
  axes (temporal, height, width) instead of one, allowing the LLM
  to reason about 2D spatial layout natively.
- **Chat-format input**: messages are expressed as a list of role-typed
  blocks; image content is inserted as a `{"type": "image"}` block
  inside the user turn.

For our purposes, the practical implication is that **we don't need to
manually resize VQA-RAD images** — the processor handles arbitrary
resolutions, which is convenient because radiology images vary widely.

## Reported results (selection)

Qwen2-VL-2B on standard VLM benchmarks:

| Benchmark | Qwen2-VL-2B | LLaVA-1.5-7B (ref) |
|-----------|------------:|-------------------:|
| MMBench-EN | 72.9 | 64.3 |
| MMMU (val) | 41.1 | 35.3 |
| DocVQA (test) | 90.1 | 28.1 |

The headline claim: at 2B parameters, Qwen2-VL approaches or matches
much larger open VLMs on text- and document-heavy benchmarks.

## Relevance to our project

Qwen2-VL-2B-Instruct is **the base model** for all our experiments.
This paper:

1. Tells us **how to format inputs** — the chat-template structure used
   in `src.data.vqarad_dataset.build_qwen_prompt` follows the convention
   defined here.
2. Justifies our **min preprocessing approach** — because of the dynamic
   resolution support, we do no manual resizing. This is empirically
   important: forcing radiology images to 336×336 destroys diagnostic
   detail.
3. Establishes the **size class** of our experiment. At 2B, the model
   is large enough to have non-trivial multimodal reasoning, but small
   enough that LoRA/QLoRA/DoRA can be trained on a single 16 GB GPU
   in a few hours. Larger Qwen2-VL variants (7B, 72B) would be
   academically interesting but exceed the proposal's resource budget.

## Open questions / caveats

- Qwen2-VL was trained primarily on **general-domain** internet
  data (web images, OCR, charts, diagrams). Its **medical priors are
  weak**, which is exactly why the zero-shot baseline is expected to
  be low and why fine-tuning is expected to help. This is the implicit
  hypothesis our project tests.
- Newer versions exist as of early 2026 (Qwen2.5-VL series and
  Qwen3-VL series). We deliberately use Qwen2-VL-2B-Instruct because
  it matches the proposal exactly and has mature HuggingFace +
  PEFT integration. Future work could re-run on newer checkpoints.
- The technical report doesn't deeply discuss the model's behavior on
  domain-shifted data; "what happens when you fine-tune Qwen2-VL with
  only ~1800 medical examples" is the gap our experiments fill.
