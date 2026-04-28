"""
PyTorch-style Dataset wrapper and Qwen2-VL prompt construction for VQA-RAD.

This module is the **interface contract** for downstream PEFT experiments:
Members 2 and 3 must construct their training inputs through
``build_qwen_prompt`` so that baseline and fine-tuned models see byte-identical
prompt formatting. Any divergence here invalidates the comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from .load_vqarad import classify_question_type, load_vqarad

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Kept short and explicit. We tested longer prompts (chain-of-thought style,
# longer role descriptions): they actively hurt accuracy on Qwen2-VL-2B —
# the 2B model is too small to benefit from CoT, and longer prompts produce
# more verbose answers that hurt Exact Match.
SYSTEM_PROMPT = (
    "You are a helpful medical assistant analyzing a radiology image. "
    "Answer the question concisely and accurately. "
    "For yes/no questions, reply with just 'yes' or 'no'. "
    "For other questions, give a short factual answer (a few words)."
)


def build_qwen_prompt(
    image: Image.Image,
    question: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Build a Qwen2-VL chat-format message list.

    The output is intended for
    ``processor.apply_chat_template(..., add_generation_prompt=True)``.

    Parameters
    ----------
    image : PIL.Image.Image
        The radiology image. Any size is accepted; the Qwen2-VL processor
        handles dynamic resolution natively.
    question : str
        The question text (English, lowercase, taken from the dataset).
    system_prompt : str, optional
        Override the default system prompt (e.g. for ablation studies).

    Returns
    -------
    list of dict
        Two-message chat structure: ``[system, user(image+text)]``.
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class VQARadDataset(Dataset):
    """Thin wrapper around the HuggingFace VQA-RAD split.

    Adds three things on top of the raw HuggingFace ``Dataset``:

    1. Automatic question-type classification (``"closed"`` / ``"open"``).
    2. A stable integer ``id`` per example (= row index within the split).
    3. RGB-mode normalization (some images are 1-channel grayscale).

    Each item is returned as a plain dict::

        {
            "id":       int,
            "image":    PIL.Image.Image,        # always RGB
            "question": str,
            "answer":   str,
            "qtype":    "closed" | "open",
        }

    Parameters
    ----------
    split : "train" | "test"
        Which split to wrap.
    cache_dir : str, optional
        HuggingFace cache directory.
    max_examples : int, optional
        If set, truncate to first N examples (handy for smoke tests and CI).
    """

    def __init__(
        self,
        split: str = "test",
        cache_dir: Optional[str] = None,
        max_examples: Optional[int] = None,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        ds = load_vqarad(cache_dir=cache_dir)
        self._split = ds[split]
        if max_examples is not None:
            self._split = self._split.select(
                range(min(max_examples, len(self._split)))
            )
        self.split_name = split

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._split[int(idx)]
        image = ex["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return {
            "id": int(idx),
            "image": image,
            "question": ex["question"],
            "answer": ex["answer"],
            "qtype": classify_question_type(ex["answer"]),
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def question_type_counts(self) -> Dict[str, int]:
        """Return ``{"closed": n_yn, "open": n_other}`` for this split."""
        counts = {"closed": 0, "open": 0}
        for ans in self._split["answer"]:
            counts[classify_question_type(ans)] += 1
        return counts


if __name__ == "__main__":
    # Smoke test
    test_ds = VQARadDataset(split="test", max_examples=5)
    print(f"Loaded {len(test_ds)} examples (truncated)")
    print(f"Question type counts: {test_ds.question_type_counts()}")
    sample = test_ds[0]
    print(f"\nSample[0]:")
    print(f"  id:       {sample['id']}")
    print(f"  question: {sample['question']}")
    print(f"  answer:   {sample['answer']}")
    print(f"  qtype:    {sample['qtype']}")
    print(f"  image:    {sample['image'].size} {sample['image'].mode}")

    msgs = build_qwen_prompt(sample["image"], sample["question"])
    print(f"\nQwen prompt: {len(msgs)} messages, "
          f"user content blocks: {len(msgs[1]['content'])}")
