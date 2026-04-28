"""
Profiling utilities: GPU memory, wall-clock time, trainable parameter counts.

These are the four metrics our project's RQ2 (efficiency) hinges on, plus
Q1's denominator (efficiency-per-accuracy-point). We need them measured
identically across baseline / LoRA / QLoRA / DoRA runs, so this module is
the single source of truth.

Members 2 and 3 must use these helpers (not roll their own) so the
numbers in the final report are directly comparable.
"""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch


# ---------------------------------------------------------------------------
# Trainable parameter counts
# ---------------------------------------------------------------------------
def count_parameters(model) -> Dict[str, int]:
    """Return total / trainable / frozen parameter counts.

    For a PEFT-adapted model, ``trainable`` is the LoRA / DoRA adapter
    size; ``total`` includes the frozen base model. The trainable
    fraction is what determines the "% trainable" column in the
    headline efficiency table.

    Returns
    -------
    dict
        ``{"total", "trainable", "frozen", "trainable_pct"}``.
        ``trainable_pct`` is a float in [0, 100].
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": (100.0 * trainable / total) if total > 0 else 0.0,
    }


def format_param_count(n: int) -> str:
    """Format ``n`` as e.g. '1.50B', '83.2M', '4.18K'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def print_parameter_summary(model, label: str = "Model") -> Dict[str, int]:
    """Pretty-print and return the parameter counts."""
    p = count_parameters(model)
    print(f"\n{label} parameters")
    print(f"  Total      : {format_param_count(p['total']):>10}  ({p['total']:,})")
    print(f"  Trainable  : {format_param_count(p['trainable']):>10}  ({p['trainable']:,})")
    print(f"  Frozen     : {format_param_count(p['frozen']):>10}  ({p['frozen']:,})")
    print(f"  Trainable %: {p['trainable_pct']:>10.4f}%")
    return p


# ---------------------------------------------------------------------------
# GPU memory tracking
# ---------------------------------------------------------------------------
def reset_peak_gpu_memory() -> None:
    """Reset the CUDA peak-memory counter so we can measure a region."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory(device: Optional[int] = None) -> Dict[str, float]:
    """Return current and peak allocated GPU memory in GB.

    "Allocated" is what PyTorch is actively using; "reserved" is what
    PyTorch has *cached* for future allocations. The reserved number is
    often what users actually see in `nvidia-smi`, so we report both.

    Returns
    -------
    dict
        ``{"allocated_gb", "reserved_gb", "peak_allocated_gb",
        "peak_reserved_gb"}``. All zero on CPU.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0, "reserved_gb": 0.0,
            "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0,
        }
    return {
        "allocated_gb":      torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb":       torch.cuda.memory_reserved(device)  / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_gb":  torch.cuda.max_memory_reserved(device)  / 1e9,
    }


# ---------------------------------------------------------------------------
# Wall-clock timing
# ---------------------------------------------------------------------------
@dataclass
class StageTimer:
    """Records named stages so we can break out epoch / step times."""
    starts: Dict[str, float]
    elapsed: Dict[str, float]

    def __init__(self) -> None:
        self.starts = {}
        self.elapsed = {}

    def start(self, name: str) -> None:
        self.starts[name] = time.time()

    def stop(self, name: str) -> float:
        if name not in self.starts:
            raise KeyError(f"Stage {name!r} was never started")
        dt = time.time() - self.starts.pop(name)
        self.elapsed[name] = self.elapsed.get(name, 0.0) + dt
        return dt


@contextlib.contextmanager
def timed(name: str = "block") -> Iterator[Dict[str, float]]:
    """Context manager that records elapsed seconds for a code block.

    Usage::

        with timed("epoch") as t:
            train_one_epoch(...)
        print(f"Epoch took {t['seconds']:.1f}s")
    """
    out = {"seconds": 0.0}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out["seconds"] = time.time() - t0


# ---------------------------------------------------------------------------
# All-in-one snapshot (for results reporting)
# ---------------------------------------------------------------------------
def profile_snapshot(model) -> Dict[str, float]:
    """Single dict combining param counts + current memory + peak memory.

    Designed to be merged into a `metrics.json` dump alongside accuracy.
    """
    out: Dict[str, float] = {}
    out.update({f"params_{k}": v for k, v in count_parameters(model).items()})
    out.update({f"gpu_{k}": v for k, v in get_gpu_memory().items()})
    return out


def cleanup_gpu() -> None:
    """Aggressive GPU memory cleanup. Call between runs to avoid leaks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    # Self-test (CPU-friendly)
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
    for p in m[0].parameters():
        p.requires_grad = False
    p = print_parameter_summary(m, "Test model")
    assert p["total"] > 0
    assert 0 <= p["trainable_pct"] <= 100

    with timed("sleep") as t:
        time.sleep(0.1)
    print(f"Block took {t['seconds']:.3f}s (expect ~0.1)")
    print("\n✓ profiling utilities OK")
