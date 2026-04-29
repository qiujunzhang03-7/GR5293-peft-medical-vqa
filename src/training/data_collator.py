"""
Qwen2-VL data collation for supervised fine-tuning on VQA-RAD.

Per-sample processing approach: each example is encoded individually with
the processor (no batched padding), then the resulting sequences are padded
manually. This avoids a known Qwen2-VL processor issue where the number of
``<|image_pad|>`` tokens inserted into ``input_ids`` does not match the
visual encoder's output token count when processing variable-resolution
images in a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from src.data.vqarad_dataset import build_qwen_prompt

LABEL_IGNORE_INDEX = -100


@dataclass
class QwenVLSFTCollator:
    """Per-sample Qwen2-VL collator for SFT on VQA-RAD."""

    processor: Any
    max_length: int = 1024

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        per_example_input_ids = []
        per_example_attention_mask = []
        per_example_labels = []
        per_example_pixel_values = []
        per_example_image_grid_thw = []

        for ex in examples:
            user_messages = build_qwen_prompt(ex["image"], ex["question"])
            answer = str(ex["answer"]).strip()
            full_msgs = user_messages + [{"role": "assistant", "content": answer}]

            full_text = self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False,
            )
            prompt_text = self.processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True,
            )

            image_inputs, _ = process_vision_info([full_msgs])

            # Encode this single example WITHOUT batched padding
            full_enc = self.processor(
                text=[full_text],
                images=image_inputs,
                videos=None,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_enc = self.processor(
                text=[prompt_text],
                images=image_inputs,
                videos=None,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = full_enc["input_ids"][0]
            attention_mask = full_enc["attention_mask"][0]
            n_prompt = prompt_enc["input_ids"].shape[1]
            n_prompt = min(n_prompt, input_ids.shape[0])

            # Build labels: copy input_ids, mask the prompt portion
            labels = input_ids.clone()
            labels[:n_prompt] = LABEL_IGNORE_INDEX

            per_example_input_ids.append(input_ids)
            per_example_attention_mask.append(attention_mask)
            per_example_labels.append(labels)
            per_example_pixel_values.append(full_enc["pixel_values"])
            per_example_image_grid_thw.append(full_enc["image_grid_thw"])

        # Pad all sequences to the max length in this batch (right-padding)
        pad_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(t.shape[0] for t in per_example_input_ids)

        def _pad(t, value):
            if t.shape[0] >= max_len:
                return t[:max_len]
            pad_amount = max_len - t.shape[0]
            return torch.cat([t, torch.full((pad_amount,), value, dtype=t.dtype)], dim=0)

        batch_input_ids = torch.stack([_pad(t, pad_id) for t in per_example_input_ids])
        batch_attention_mask = torch.stack(
            [_pad(t, 0) for t in per_example_attention_mask]
        )
        batch_labels = torch.stack(
            [_pad(t, LABEL_IGNORE_INDEX) for t in per_example_labels]
        )
        # pixel_values and image_grid_thw concatenate along the "image" axis,
        # not stack — Qwen2-VL handles variable image counts this way.
        batch_pixel_values = torch.cat(per_example_pixel_values, dim=0)
        batch_image_grid_thw = torch.cat(per_example_image_grid_thw, dim=0)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
        }
