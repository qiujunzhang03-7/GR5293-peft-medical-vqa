"""Tests for profiling utilities (CPU-only)."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.utils.profiling import (
    count_parameters,
    format_param_count,
    get_gpu_memory,
    timed,
)


class TestParameterCounts:
    def test_count_simple_model(self):
        m = nn.Linear(100, 50)  # 100*50 + 50 = 5050 params
        p = count_parameters(m)
        assert p["total"] == 5050
        assert p["trainable"] == 5050
        assert p["frozen"] == 0
        assert p["trainable_pct"] == 100.0

    def test_partial_freeze(self):
        m = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 1))
        for p in m[0].parameters():
            p.requires_grad = False
        p = count_parameters(m)
        assert p["frozen"] == 10*10 + 10  # first layer
        assert p["trainable"] == 10 + 1   # second layer
        assert p["total"] == p["frozen"] + p["trainable"]
        assert 0 < p["trainable_pct"] < 100

    def test_all_frozen(self):
        m = nn.Linear(10, 1)
        for p in m.parameters():
            p.requires_grad = False
        out = count_parameters(m)
        assert out["trainable"] == 0
        assert out["trainable_pct"] == 0.0

    def test_empty_model(self):
        m = nn.Module()  # no params
        p = count_parameters(m)
        assert p["total"] == 0
        assert p["trainable_pct"] == 0.0


class TestFormatParamCount:
    def test_billions(self):
        assert format_param_count(1_500_000_000) == "1.50B"

    def test_millions(self):
        assert format_param_count(83_200_000) == "83.20M"

    def test_thousands(self):
        assert format_param_count(4_180) == "4.18K"

    def test_small(self):
        assert format_param_count(42) == "42"


class TestGPUMemory:
    def test_returns_required_keys(self):
        m = get_gpu_memory()
        for key in ("allocated_gb", "reserved_gb",
                    "peak_allocated_gb", "peak_reserved_gb"):
            assert key in m
            assert m[key] >= 0


class TestTiming:
    def test_timed_context_manager(self):
        with timed("test") as t:
            time.sleep(0.05)
        assert 0.04 < t["seconds"] < 0.5  # very loose bounds

    def test_timed_zero_work(self):
        with timed("nothing") as t:
            pass
        assert t["seconds"] >= 0
