"""
Statistical significance tests for VQA evaluation.

The course rubric explicitly grades on "Statistical significance of results"
under the Final Project Presentation (Evaluation, 10%) component. Reporting
point estimates alone is insufficient: when LoRA / QLoRA / DoRA differ from
the baseline (or each other) by a few percentage points on a 451-example
test set, we need confidence intervals and paired hypothesis tests to know
whether the differences are real.

This module provides three complementary tools.

1. ``bootstrap_ci`` — non-parametric confidence interval for any aggregate
   metric, computed by resampling examples with replacement.

2. ``paired_bootstrap_ci_diff`` — confidence interval for the *difference*
   between two methods on the *same* test set. Pairing eliminates the
   between-example variance and gives much tighter intervals than testing
   each method's CI separately.

3. ``mcnemar_test`` — exact binomial McNemar test for paired binary
   outcomes (correct/incorrect). The textbook test for "did Method B fix
   errors that Method A made?" on the same evaluation set.

All three functions are deterministic given a fixed seed.

References
----------
* Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.*
  Chapman & Hall.
* Dietterich, T. G. (1998). Approximate statistical tests for comparing
  supervised classification learning algorithms. *Neural Computation,*
  10(7), 1895-1923.   (Recommends McNemar for paired classifier comparison.)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap CI for a single aggregate metric
# ---------------------------------------------------------------------------
def bootstrap_ci(
    per_example_scores: Sequence[float],
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
    aggregator: Callable[[np.ndarray], float] = np.mean,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for an aggregate metric.

    Parameters
    ----------
    per_example_scores : sequence of float
        One score per evaluation example. For Exact Match, pass the 0/1
        per-example correctness vector. For BLEU/ROUGE/F1, pass the
        per-example scores returned by the metric.
    n_resamples : int
        Number of bootstrap resamples. 10,000 is the conventional default
        and is more than adequate for two-decimal-place CIs on n ≈ 500.
    confidence : float
        Two-sided confidence level (default 0.95 → 95% CI).
    seed : int
        RNG seed for reproducibility.
    aggregator : callable
        How to aggregate per-example scores into a single metric.
        Default ``np.mean`` is correct for EM, BLEU, ROUGE, F1.

    Returns
    -------
    dict
        ``{"point": float, "lower": float, "upper": float, "se": float}``,
        the point estimate (sample aggregate), CI bounds, and bootstrap
        standard error.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    scores = np.asarray(per_example_scores, dtype=float)
    n = len(scores)
    if n == 0:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "se": 0.0}

    rng = np.random.default_rng(seed)
    # Vectorized resampling: shape (n_resamples, n)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = scores[idx]
    boot_stats = np.array([aggregator(row) for row in resampled])

    alpha = 1 - confidence
    lower = float(np.quantile(boot_stats, alpha / 2))
    upper = float(np.quantile(boot_stats, 1 - alpha / 2))
    point = float(aggregator(scores))
    se = float(np.std(boot_stats, ddof=1))
    return {"point": point, "lower": lower, "upper": upper, "se": se}


# ---------------------------------------------------------------------------
# Paired bootstrap for the difference between two methods
# ---------------------------------------------------------------------------
def paired_bootstrap_ci_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap CI for ``mean(B) - mean(A)`` on the same examples.

    "Paired" means both methods are evaluated on the exact same test set
    in the same order; ``scores_a[i]`` and ``scores_b[i]`` refer to the
    same example. Resampling indices jointly preserves the pairing and
    produces tighter CIs than two unpaired bootstraps.

    Parameters
    ----------
    scores_a, scores_b : sequence of float
        Per-example scores for methods A and B. Must be the same length.
    n_resamples, confidence, seed : see ``bootstrap_ci``.

    Returns
    -------
    dict
        ``{"point", "lower", "upper", "se", "p_two_sided"}``. The point
        estimate is ``mean(B) - mean(A)``; positive means B is better.
        ``p_two_sided`` is the bootstrap p-value for the null
        ``mean(B) == mean(A)`` (achieved significance level).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"Length mismatch: {len(a)} vs {len(b)}. Methods must be "
            "evaluated on the same examples in the same order."
        )
    n = len(a)
    if n == 0:
        return {
            "point": 0.0, "lower": 0.0, "upper": 0.0,
            "se": 0.0, "p_two_sided": 1.0,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)

    alpha = 1 - confidence
    point = float(b.mean() - a.mean())
    lower = float(np.quantile(diffs, alpha / 2))
    upper = float(np.quantile(diffs, 1 - alpha / 2))
    se = float(np.std(diffs, ddof=1))
    # Two-sided bootstrap p-value: 2 * min(P(diff <= 0), P(diff >= 0))
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p = float(min(p, 1.0))
    return {
        "point": point, "lower": lower, "upper": upper,
        "se": se, "p_two_sided": p,
    }


# ---------------------------------------------------------------------------
# McNemar's exact test for paired binary outcomes
# ---------------------------------------------------------------------------
def mcnemar_test(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
) -> Dict[str, float]:
    """McNemar's exact binomial test for paired classifier comparison.

    Given per-example correctness 0/1 vectors for methods A and B on the
    same test set, McNemar tests the null that the two methods have the
    same probability of being correct on a random example.

    Concretely, let

    * ``b`` = #{A correct, B wrong}     (cases where A wins)
    * ``c`` = #{A wrong, B correct}     (cases where B wins)

    Under the null, b and c are exchangeable, so b ~ Binomial(b+c, 0.5).
    We compute the exact two-sided binomial p-value.

    Parameters
    ----------
    correct_a, correct_b : sequence of int (0 or 1)
        Per-example correctness, same length, same example order.

    Returns
    -------
    dict
        ``{"b": int, "c": int, "n_disagree": int, "p_value": float}``.
        A small p-value rejects "the two methods are equally accurate".
    """
    a = np.asarray(correct_a, dtype=int)
    b_arr = np.asarray(correct_b, dtype=int)
    if a.shape != b_arr.shape:
        raise ValueError("correct_a and correct_b must have the same length")
    if not np.all((a == 0) | (a == 1)) or not np.all((b_arr == 0) | (b_arr == 1)):
        raise ValueError("Inputs must be 0/1 indicator vectors")

    # b = A correct AND B wrong;  c = A wrong AND B correct
    b_count = int(np.sum((a == 1) & (b_arr == 0)))
    c_count = int(np.sum((a == 0) & (b_arr == 1)))
    n_disagree = b_count + c_count
    if n_disagree == 0:
        # Identical predictions → cannot reject null
        return {"b": b_count, "c": c_count, "n_disagree": 0, "p_value": 1.0}

    # Exact two-sided binomial test against p = 0.5
    k = min(b_count, c_count)
    # P(X <= k) under Binomial(n_disagree, 0.5), times 2 for two-sided
    log_pmf = (
        _log_binom_coef(n_disagree, np.arange(k + 1))
        - n_disagree * np.log(2)
    )
    one_tail = float(np.exp(log_pmf).sum())
    p_value = float(min(2 * one_tail, 1.0))
    return {
        "b": b_count, "c": c_count,
        "n_disagree": n_disagree, "p_value": p_value,
    }


def _log_binom_coef(n: int, k: np.ndarray) -> np.ndarray:
    """Log of binomial coefficient C(n, k) using lgamma — numerically stable."""
    from math import lgamma
    return np.array([
        lgamma(n + 1) - lgamma(int(ki) + 1) - lgamma(n - int(ki) + 1)
        for ki in k
    ])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. Bootstrap CI on a coin-flip-ish accuracy
    flips = rng.binomial(1, 0.7, size=500)
    ci = bootstrap_ci(flips, seed=42)
    print(f"Bootstrap CI: point={ci['point']:.3f}, "
          f"95% CI=[{ci['lower']:.3f}, {ci['upper']:.3f}]  "
          f"(true p=0.7)")

    # 2. Paired bootstrap diff: B is reliably 5pp better than A
    a = rng.binomial(1, 0.7, size=500)
    b = (a | (rng.uniform(size=500) < 0.05)).astype(int)  # B fixes 5% of A's errors
    diff = paired_bootstrap_ci_diff(a, b, seed=42)
    print(f"Paired diff: point={diff['point']:+.3f}, "
          f"95% CI=[{diff['lower']:+.3f}, {diff['upper']:+.3f}], "
          f"p={diff['p_two_sided']:.4f}")

    # 3. McNemar
    mc = mcnemar_test(a, b)
    print(f"McNemar: b={mc['b']}, c={mc['c']}, p={mc['p_value']:.4g}")

    # 4. Edge cases
    assert bootstrap_ci([])["point"] == 0.0
    assert mcnemar_test([1, 1, 0], [1, 1, 0])["p_value"] == 1.0
    print("\nAll edge cases handled.")
