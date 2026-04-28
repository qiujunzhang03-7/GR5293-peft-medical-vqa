"""
VQA-RAD dataset loading utilities.

VQA-RAD (Lau et al., 2018, *Scientific Data*) is a clinician-curated medical
visual question answering benchmark built from radiology images sourced from
MedPix. The HuggingFace release ``flaviagiammarino/vqa-rad`` provides the
official train/test split:

* train: 1,797 question-answer pairs
* test:    451 question-answer pairs
* 315 unique radiology images (one image typically associated with multiple
  questions; the split is by question, not by image)

License: CC0 1.0 Universal (public domain dedication).

References
----------
Lau, J. J., Gayen, S., Ben Abacha, A., & Demner-Fushman, D. (2018).
A dataset of clinically generated visual questions and answers about
radiology images. Scientific Data, 5(1), 180251.
"""

from __future__ import annotations

import re
from typing import Dict, Literal, Optional

from datasets import DatasetDict, load_dataset

DATASET_ID = "flaviagiammarino/vqa-rad"

# VQA-RAD answers that fall into the closed-ended (binary) bucket.
# Standard convention in the medical VQA literature (e.g. LLaVA-Med,
# BiomedGPT): an example is "closed" iff its reference answer normalizes
# to "yes" or "no", and "open" otherwise.
_CLOSED_TOKENS = frozenset({"yes", "no"})


def load_vqarad(
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> DatasetDict:
    """Load VQA-RAD from HuggingFace.

    Parameters
    ----------
    cache_dir : str, optional
        Where to cache downloaded files. On Colab, point this at a Google
        Drive path (e.g. ``/content/drive/MyDrive/peft-vqa-cache``) to
        survive runtime restarts.
    streaming : bool
        If True, returns iterable datasets without materializing.
        Default False — the full dataset is small (~85 MB).

    Returns
    -------
    DatasetDict
        With splits ``"train"`` and ``"test"``. Each example has fields
        ``image`` (PIL.Image, variable resolution),
        ``question`` (str), and ``answer`` (str, lowercase).

    Raises
    ------
    RuntimeError
        If the loaded dataset's structure has unexpectedly changed.
    """
    ds = load_dataset(DATASET_ID, cache_dir=cache_dir, streaming=streaming)
    expected = {"train", "test"}
    if not expected.issubset(set(ds.keys())):
        raise RuntimeError(
            f"VQA-RAD: expected splits {expected}, got {set(ds.keys())}. "
            "The HuggingFace dataset may have changed; check the dataset card."
        )
    return ds


def classify_question_type(answer: str) -> Literal["closed", "open"]:
    """Classify a VQA-RAD example as closed-ended or open-ended.

    Classification is based on the ground-truth ``answer``, since VQA-RAD
    does not ship a question-type label. We follow the convention adopted
    in medical VQA papers (LLaVA-Med, BiomedGPT, PubMedCLIP):

    * **closed**: answer normalizes to "yes" or "no" (case-insensitive,
      after stripping punctuation and whitespace).
    * **open**:   anything else.

    The two types are scored with different metrics: Exact Match for closed,
    and BLEU/ROUGE/F1 for open.

    Parameters
    ----------
    answer : str
        Ground-truth answer string.

    Returns
    -------
    "closed" | "open"
    """
    if answer is None:
        return "open"
    normalized = re.sub(r"[^\w\s]", "", answer.strip().lower())
    return "closed" if normalized in _CLOSED_TOKENS else "open"


def split_statistics(ds: DatasetDict) -> Dict[str, Dict[str, int]]:
    """Compute per-split summary statistics (sanity check).

    Returns
    -------
    dict
        ``{split_name: {"total": N, "closed": N_yn, "open": N_other,
                         "unique_images": K_or_-1}}``
        ``unique_images`` may be ``-1`` if the underlying image objects
        cannot be hashed cheaply; this is informational only.
    """
    stats: Dict[str, Dict[str, int]] = {}
    for split_name, split in ds.items():
        closed = sum(
            1 for a in split["answer"] if classify_question_type(a) == "closed"
        )
        try:
            unique_images = len({id(img) for img in split["image"]})
        except Exception:  # noqa: BLE001
            unique_images = -1
        stats[split_name] = {
            "total": len(split),
            "closed": closed,
            "open": len(split) - closed,
            "unique_images": unique_images,
        }
    return stats


if __name__ == "__main__":
    # Smoke test: run this file directly to verify the loader works.
    print("Loading VQA-RAD from HuggingFace...")
    ds = load_vqarad()
    print(f"Splits: {list(ds.keys())}")
    print(f"Train: {len(ds['train'])} examples")
    print(f"Test:  {len(ds['test'])} examples")
    print()
    print("Sample example (train[0]):")
    ex = ds["train"][0]
    print(f"  Question: {ex['question']}")
    print(f"  Answer:   {ex['answer']}")
    print(f"  Image:    {ex['image'].size} {ex['image'].mode}")
    print(f"  Type:     {classify_question_type(ex['answer'])}")
    print()
    print("Per-split statistics:")
    for split, s in split_statistics(ds).items():
        print(f"  {split}: {s}")
