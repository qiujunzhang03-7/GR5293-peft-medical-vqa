"""
Evaluation metrics for medical Visual Question Answering.

Closed-ended (yes/no) → Exact Match (after text normalization)
Open-ended            → BLEU-1, ROUGE-L, Token-level F1

These are the canonical metrics used in the medical VQA literature
(LLaVA-Med, BiomedGPT, PubMedCLIP). Reporting all four lets us cover
both lexical-overlap (BLEU/ROUGE) and content-overlap (F1) views of
free-form answers, while keeping closed-ended scoring unambiguous.

Every function here is pure-Python and has zero ML-framework dependencies,
so the test suite can exercise them without GPU.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    This matches the SQuAD/VQA convention. **Articles ("a", "an", "the")
    are deliberately not removed** because they can be diagnostically
    meaningful in radiology questions (e.g., "is there a mass?" → the
    presence/absence of an article changes the syntactic question type).
    """
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    """Whitespace-tokenize after normalization."""
    return normalize_text(s).split()


# ---------------------------------------------------------------------------
# Closed-ended: Exact Match
# ---------------------------------------------------------------------------
def exact_match(predictions: List[str], references: List[str]) -> float:
    """Fraction of predictions that match references after normalization.

    Returns
    -------
    float in [0, 1]
        The mean of normalized-string equality.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(references)} references"
        )
    if not predictions:
        return 0.0
    matches = sum(
        1 for p, r in zip(predictions, references)
        if normalize_text(p) == normalize_text(r)
    )
    return matches / len(predictions)


def per_example_correct(
    predictions: List[str], references: List[str]
) -> List[int]:
    """Return a list of 0/1 indicators for each (pred, ref) pair.

    Useful for computing bootstrap CIs and McNemar tests downstream.
    """
    if len(predictions) != len(references):
        raise ValueError("Length mismatch")
    return [
        int(normalize_text(p) == normalize_text(r))
        for p, r in zip(predictions, references)
    ]


# ---------------------------------------------------------------------------
# Open-ended: token-level F1, BLEU-1, ROUGE-L
# ---------------------------------------------------------------------------
def _f1_pair(pred: str, ref: str) -> float:
    """Token-level F1 for a single (prediction, reference) pair."""
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(predictions: List[str], references: List[str]) -> float:
    """Mean token-level F1 across pairs.

    More forgiving than EM for open-ended answers — e.g. ``"cardiomegaly"``
    vs ``"enlarged heart, cardiomegaly"`` gets partial credit.
    """
    if len(predictions) != len(references):
        raise ValueError("Length mismatch")
    if not predictions:
        return 0.0
    return sum(
        _f1_pair(p, r) for p, r in zip(predictions, references)
    ) / len(predictions)


def _bleu1_pair(pred: str, ref: str) -> float:
    """Sentence-level BLEU-1 (unigram precision with brevity penalty).

    We implement BLEU-1 directly rather than calling ``sacrebleu`` because
    BLEU is conventionally a *corpus-level* metric, and using it
    sentence-wise via sacrebleu yields zero scores on short answers due to
    smoothing edge cases. The standalone implementation here matches the
    "BLEU-1 with brevity penalty" used in VQA papers.
    """
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0

    # Brevity penalty
    if len(pred_tokens) > len(ref_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
    return bp * precision


def _rouge_l_pair(pred: str, ref: str) -> float:
    """ROUGE-L F-measure for a single pair, computed via LCS.

    Uses the ``rouge_score`` package (Google's reference implementation)
    when available, falling back to a self-contained LCS DP otherwise.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(normalize_text(ref), normalize_text(pred))
        return scores["rougeL"].fmeasure
    except ImportError:
        pred_tokens = _tokenize(pred)
        ref_tokens = _tokenize(ref)
        if not pred_tokens or not ref_tokens:
            return 0.0
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        if lcs == 0:
            return 0.0
        precision = lcs / m
        recall = lcs / n
        return 2 * precision * recall / (precision + recall)


def open_ended_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute the open-ended metric suite.

    Returns
    -------
    dict
        ``{"bleu1": float, "rougeL": float, "f1": float}``, each in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError("Length mismatch")
    if not predictions:
        return {"bleu1": 0.0, "rougeL": 0.0, "f1": 0.0}
    n = len(predictions)
    bleu1 = sum(_bleu1_pair(p, r) for p, r in zip(predictions, references)) / n
    rouge = sum(_rouge_l_pair(p, r) for p, r in zip(predictions, references)) / n
    f1 = token_f1(predictions, references)
    return {"bleu1": bleu1, "rougeL": rouge, "f1": f1}


# ---------------------------------------------------------------------------
# Top-level: compute everything at once, broken down by question type
# ---------------------------------------------------------------------------
def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    qtypes: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute all metrics, broken down by question type.

    Parameters
    ----------
    predictions : list[str]
        Model-generated answers.
    references : list[str]
        Ground-truth answers from VQA-RAD.
    qtypes : list[str]
        Per-example question type, each in ``{"closed", "open"}``.

    Returns
    -------
    dict
        Top-level keys ``"closed"``, ``"open"``, ``"overall"``::

            {
                "closed":  {"n": int, "exact_match": float},
                "open":    {"n": int, "bleu1": float, "rougeL": float, "f1": float},
                "overall": {"n": int, "exact_match": float},
            }

        ``overall.exact_match`` is the headline single-number accuracy
        used in the proposal's RQ1 comparison. **Always report the
        breakdown alongside it**, since closed/open ratios differ across
        splits and methods.
    """
    if not (len(predictions) == len(references) == len(qtypes)):
        raise ValueError(
            f"Length mismatch: preds={len(predictions)}, "
            f"refs={len(references)}, qtypes={len(qtypes)}"
        )

    closed_preds, closed_refs = [], []
    open_preds, open_refs = [], []
    for p, r, q in zip(predictions, references, qtypes):
        if q == "closed":
            closed_preds.append(p)
            closed_refs.append(r)
        else:
            open_preds.append(p)
            open_refs.append(r)

    closed_metrics = {
        "n": len(closed_preds),
        "exact_match": (
            exact_match(closed_preds, closed_refs) if closed_preds else 0.0
        ),
    }
    open_block = (
        open_ended_metrics(open_preds, open_refs)
        if open_preds
        else {"bleu1": 0.0, "rougeL": 0.0, "f1": 0.0}
    )
    open_metrics = {"n": len(open_preds), **open_block}
    overall_metrics = {
        "n": len(predictions),
        "exact_match": exact_match(predictions, references),
    }
    return {
        "closed": closed_metrics,
        "open": open_metrics,
        "overall": overall_metrics,
    }


if __name__ == "__main__":
    # Mini self-test
    preds = ["yes", "no", "cardiomegaly", "left lung"]
    refs = ["yes", "no", "cardiomegaly", "left lung"]
    qtypes = ["closed", "closed", "open", "open"]
    m = compute_all_metrics(preds, refs, qtypes)
    print("Perfect predictions:")
    print(f"  closed EM: {m['closed']['exact_match']:.3f}  (expect 1.000)")
    print(f"  open F1:   {m['open']['f1']:.3f}  (expect 1.000)")
    print(f"  overall:   {m['overall']['exact_match']:.3f}  (expect 1.000)")
