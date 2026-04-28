"""Unit tests for statistical significance utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.statistical_tests import (
    bootstrap_ci,
    mcnemar_test,
    paired_bootstrap_ci_diff,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------
class TestBootstrapCI:
    def test_returns_required_keys(self):
        ci = bootstrap_ci([1, 0, 1, 0, 1, 1, 0], seed=42)
        assert set(ci.keys()) == {"point", "lower", "upper", "se"}

    def test_point_is_mean(self):
        scores = [1, 0, 1, 1, 0]
        ci = bootstrap_ci(scores, seed=42)
        assert abs(ci["point"] - 0.6) < 1e-9

    def test_ci_contains_point(self):
        ci = bootstrap_ci([1, 0, 1, 0, 1, 1], seed=42)
        assert ci["lower"] <= ci["point"] <= ci["upper"]

    def test_ci_within_unit_interval_for_indicators(self):
        ci = bootstrap_ci([1, 0, 1, 0, 1, 1, 0], seed=42)
        assert 0 <= ci["lower"] <= 1
        assert 0 <= ci["upper"] <= 1

    def test_recovers_true_proportion(self):
        # With n=2000 trials at p=0.7, the 95% CI should contain 0.7
        rng = np.random.default_rng(0)
        scores = rng.binomial(1, 0.7, size=2000)
        ci = bootstrap_ci(scores, seed=42, n_resamples=2000)
        assert ci["lower"] <= 0.7 <= ci["upper"]

    def test_empty_input(self):
        ci = bootstrap_ci([])
        assert ci == {"point": 0.0, "lower": 0.0, "upper": 0.0, "se": 0.0}

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1, 0], confidence=0.0)
        with pytest.raises(ValueError):
            bootstrap_ci([1, 0], confidence=1.0)

    def test_deterministic_with_seed(self):
        scores = [1, 0, 1, 1, 0, 1, 0, 0, 1]
        a = bootstrap_ci(scores, seed=123)
        b = bootstrap_ci(scores, seed=123)
        assert a == b


# ---------------------------------------------------------------------------
# paired_bootstrap_ci_diff
# ---------------------------------------------------------------------------
class TestPairedBootstrap:
    def test_zero_difference(self):
        x = [1, 0, 1, 0, 1, 1, 0]
        d = paired_bootstrap_ci_diff(x, x, seed=42)
        assert abs(d["point"]) < 1e-9
        assert d["lower"] <= 0 <= d["upper"]

    def test_positive_difference_detected(self):
        # Method B reliably better than A by ~5pp on a large set
        rng = np.random.default_rng(0)
        a = rng.binomial(1, 0.7, size=2000)
        b = (a | (rng.uniform(size=2000) < 0.05)).astype(int)
        d = paired_bootstrap_ci_diff(a, b, seed=42)
        assert d["point"] > 0
        assert d["lower"] > 0  # CI excludes 0 → significant
        assert d["p_two_sided"] < 0.05

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci_diff([1, 0], [1])

    def test_empty(self):
        d = paired_bootstrap_ci_diff([], [])
        assert d["point"] == 0.0


# ---------------------------------------------------------------------------
# mcnemar_test
# ---------------------------------------------------------------------------
class TestMcNemar:
    def test_identical_predictions(self):
        # No disagreement at all → p = 1.0
        x = [1, 0, 1, 1, 0]
        m = mcnemar_test(x, x)
        assert m["p_value"] == 1.0
        assert m["b"] == 0 and m["c"] == 0

    def test_b_and_c_counts(self):
        a = [1, 1, 0, 0]
        b = [1, 0, 1, 0]
        # Index 1: A=1,B=0  → b counter
        # Index 2: A=0,B=1  → c counter
        m = mcnemar_test(a, b)
        assert m["b"] == 1
        assert m["c"] == 1
        # 1 vs 1 → exactly null → p = 1.0
        assert abs(m["p_value"] - 1.0) < 1e-9

    def test_strong_signal(self):
        # B fixes 30 out of 40 of A's errors and adds 0 new errors
        # → b=0, c=30, n=30, p ≈ 2 * 0.5^30 ≈ 1.86e-9
        a = [0] * 40 + [1] * 60
        b = [1] * 30 + [0] * 10 + [1] * 60
        m = mcnemar_test(a, b)
        assert m["b"] == 0
        assert m["c"] == 30
        assert m["p_value"] < 1e-6

    def test_invalid_inputs_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([1, 0], [1])  # length mismatch
        with pytest.raises(ValueError):
            mcnemar_test([1, 2], [1, 0])  # not 0/1
