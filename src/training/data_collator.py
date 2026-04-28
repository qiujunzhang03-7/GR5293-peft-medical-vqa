"""
Qwen2-VL data collation for supervised fine-tuning.

Why a custom collator? Qwen2-VL's processor handles a single
chat-format example end-to-end (text + image → input_ids, pixel_values,
image_grid_thw, attention_mask). For a *batch*, we need to:

1. Apply the chat template per example (with the answer appended for
   teacher forcing).
2. Run the processor on the whole batch with padding.
3. Build a ``labels`` tensor where the **prompt portion is masked**
   (label = -100) so the loss is computed *only* on answer tokens.
4. Re-attach the visual fields (``pixel_values``, ``image_grid_thw``)
   that the processor returned.

Step 3 is the easy-to-get-wrong part. If we don't mask the prompt, the
model "learns" to predict the system message, which destroys task
performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from src.data.vqarad_dataset import build_qwen_prompt

LABEL_IGNORE_INDEX = -100  # PyTorch CE-loss convention


@dataclass
class QwenVLSFTCollator:
    """Collator for Qwen2-VL supervised fine-tuning on VQA-RAD.

    Each input dict is what ``VQARadDataset.__getitem__`` returns
    (image, question, answer). The collator:

    1. Builds a Qwen2-VL chat-format prompt (system + user(image+text)).
    2. Tokenizes the prompt + the answer (assistant turn) together.
    3. Records the prompt token length so we can mask it out of labels.
    4. Pads the batch and returns a model-ready dict.

    Parameters
    ----------
    processor : transformers.AutoProcessor
        The Qwen2-VL processor (tokenizer + image processor combined).
    max_length : int
        Truncate input_ids to this length. 1024 is generous given that
        VQA-RAD answers are <10 tokens and questions <20.
    """

    processor: Any
    max_length: int = 1024

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts: List[str] = []
        full_texts: List[str] = []
        images = []

        for ex in examples:
            messages = build_qwen_prompt(ex["image"], ex["question"])
            # Prompt = chat template up to (but not including) the assistant turn
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Full = prompt + answer + EOS-like end marker for the assistant turn
            answer = str(ex["answer"]).strip()
            full_text = prompt_text + answer + "<|im_end|>"
            prompts.append(prompt_text)
            full_texts.append(full_text)
            images.append(ex["image"])

        # Tokenize full sequences with padding
        enc = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Build labels by cloning input_ids and masking out:
        #   (a) padding tokens
        #   (b) the prompt portion of each example (so loss is on answer only)
        labels = enc["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = LABEL_IGNORE_INDEX

        # Compute prompt length per example, mask out
        for i, prompt in enumerate(prompts):
            prompt_ids = self.processor.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            n_prompt = min(len(prompt_ids), labels.shape[1])
            labels[i, :n_prompt] = LABEL_IGNORE_INDEX

        enc["labels"] = labels
        return enc
