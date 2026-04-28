"""Unit tests for evaluation metrics.

These tests run on CPU only — no GPU, no network, no model download.
They guarantee the metric implementations are correct and stable across
refactors. Run with::

    pytest tests/ -v
"""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    exact_match,
    normalize_text,
    open_ended_metrics,
    per_example_correct,
    token_f1,
)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------
class TestNormalize:
    def test_lowercase(self):
        assert normalize_text("Yes") == "yes"
        assert normalize_text("CARDIOMEGALY") == "cardiomegaly"

    def test_strip_punctuation(self):
        assert normalize_text("yes.") == "yes"
        assert normalize_text("yes!!!") == "yes"
        assert normalize_text("yes, definitely") == "yes definitely"

    def test_collapse_whitespace(self):
        assert normalize_text("  yes  ") == "yes"
        assert normalize_text("left  lung") == "left lung"
        assert normalize_text("left\tlung\n") == "left lung"

    def test_articles_preserved(self):
        # We deliberately keep articles (a/an/the) for radiology semantics
        assert normalize_text("a mass") == "a mass"
        assert normalize_text("the heart") == "the heart"

    def test_handles_none(self):
        assert normalize_text(None) == ""

    def test_handles_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------
class TestExactMatch:
    def test_perfect(self):
        assert exact_match(["yes", "no"], ["yes", "no"]) == 1.0

    def test_all_wrong(self):
        assert exact_match(["yes", "no"], ["no", "yes"]) == 0.0

    def test_half(self):
        assert exact_match(["yes", "yes"], ["yes", "no"]) == 0.5

    def test_robust_to_case_punct(self):
        assert exact_match(["YES.", "no!"], ["yes", "no"]) == 1.0

    def test_empty(self):
        assert exact_match([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            exact_match(["yes"], ["yes", "no"])


# ---------------------------------------------------------------------------
# per_example_correct
# ---------------------------------------------------------------------------
class TestPerExampleCorrect:
    def test_basic(self):
        assert per_example_correct(["yes", "no"], ["yes", "no"]) == [1, 1]
        assert per_example_correct(["yes", "no"], ["no", "yes"]) == [0, 0]
        assert per_example_correct(["yes", "no"], ["yes", "yes"]) == [1, 0]

    def test_robust_to_case(self):
        assert per_example_correct(["YES."], ["yes"]) == [1]


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------
class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1(["left lung"], ["left lung"]) == 1.0

    def test_partial_overlap(self):
        # "cardiomegaly" vs "enlarged heart cardiomegaly":
        # 1 common token / 1 pred / 3 ref → P=1.0, R=1/3, F1=2*1*0.333/(1+0.333) = 0.5
        f1 = token_f1(["cardiomegaly"], ["enlarged heart cardiomegaly"])
        assert abs(f1 - 0.5) < 1e-9

    def test_no_overlap(self):
        assert token_f1(["pneumonia"], ["cardiomegaly"]) == 0.0

    def test_both_empty(self):
        assert token_f1([""], [""]) == 1.0

    def test_one_empty(self):
        assert token_f1([""], ["something"]) == 0.0
        assert token_f1(["something"], [""]) == 0.0


# ---------------------------------------------------------------------------
# open_ended_metrics
# ---------------------------------------------------------------------------
class TestOpenEndedMetrics:
    def test_perfect(self):
        m = open_ended_metrics(["left lung"], ["left lung"])
        assert m["bleu1"] == 1.0
        assert m["rougeL"] == 1.0
        assert m["f1"] == 1.0

    def test_returns_all_keys(self):
        m = open_ended_metrics(["foo"], ["bar"])
        assert set(m.keys()) == {"bleu1", "rougeL", "f1"}

    def test_empty(self):
        m = open_ended_metrics([], [])
        assert m == {"bleu1": 0.0, "rougeL": 0.0, "f1": 0.0}

    def test_in_unit_interval(self):
        # All metrics must be in [0, 1]
        m = open_ended_metrics(
            ["the patient has a small left lung mass"],
            ["left lung mass"],
        )
        for v in m.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# compute_all_metrics — integration
# ---------------------------------------------------------------------------
class TestComputeAllMetrics:
    def test_perfect_predictions(self):
        preds = ["yes", "no", "cardiomegaly", "left lung"]
        refs = preds[:]
        qtypes = ["closed", "closed", "open", "open"]
        m = compute_all_metrics(preds, refs, qtypes)
        assert m["closed"]["exact_match"] == 1.0
        assert m["closed"]["n"] == 2
        assert m["open"]["n"] == 2
        assert m["open"]["f1"] == 1.0
        assert m["overall"]["exact_match"] == 1.0
        assert m["overall"]["n"] == 4

    def test_partition_by_qtype(self):
        preds = ["yes", "cardiomegaly"]
        refs = ["yes", "cardiomegaly"]
        qtypes = ["closed", "open"]
        m = compute_all_metrics(preds, refs, qtypes)
        assert m["closed"]["n"] == 1
        assert m["open"]["n"] == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_all_metrics(["a"], ["a", "b"], ["open"])

    def test_only_closed(self):
        m = compute_all_metrics(["yes"], ["yes"], ["closed"])
        assert m["closed"]["n"] == 1
        assert m["open"]["n"] == 0

    def test_only_open(self):
        m = compute_all_metrics(["lung"], ["lung"], ["open"])
        assert m["open"]["n"] == 1
        assert m["closed"]["n"] == 0
