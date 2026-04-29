"""Tests for the LoRA training configuration plumbing.

These don't load the actual model (no GPU, no network); they verify that
the YAML parsing, default values, and CLI-override paths work correctly.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from src.training.train_lora import LoRATrainingConfig


class TestConfigDefaults:
    def test_default_construct(self):
        cfg = LoRATrainingConfig()
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16
        assert cfg.use_dora is False
        assert cfg.load_in_4bit is False
        assert cfg.train_max_examples == 200          # quick run
        assert cfg.gradient_checkpointing is True

    def test_target_modules_default(self):
        cfg = LoRATrainingConfig()
        assert "q_proj" in cfg.target_modules
        assert "v_proj" in cfg.target_modules

    def test_seed_default(self):
        assert LoRATrainingConfig().seed == 42


class TestConfigYAML:
    def test_load_quick_yaml(self):
        cfg = LoRATrainingConfig.from_yaml("configs/lora_quick.yaml")
        assert cfg.lora_r == 8
        assert cfg.use_dora is False
        assert cfg.load_in_4bit is False
        assert cfg.train_max_examples == 200

    def test_load_rank8_yaml(self):
        cfg = LoRATrainingConfig.from_yaml("configs/lora_rank8.yaml")
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16  # alpha = 2*r convention
        assert cfg.train_max_examples is None  # full split
        assert "gate_proj" in cfg.target_modules

    def test_yaml_with_unknown_keys_is_tolerated(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"lora_r": 16, "completely_unknown_field": "ignored"}, f)
            path = f.name
        try:
            cfg = LoRATrainingConfig.from_yaml(path)
            assert cfg.lora_r == 16
        finally:
            os.unlink(path)


class TestQLoRADoRASwitches:
    """The QLoRA / DoRA switches are 1-line config changes — verify they
    propagate cleanly. (Members 2 & 3's contracts.)"""

    def test_qlora_switch(self):
        cfg = LoRATrainingConfig(load_in_4bit=True)
        assert cfg.load_in_4bit is True
        assert cfg.use_dora is False

    def test_dora_switch(self):
        cfg = LoRATrainingConfig(use_dora=True)
        assert cfg.use_dora is True
        assert cfg.load_in_4bit is False

    def test_qdora_combined_works(self):
        # Q-DoRA: 4-bit + DoRA. Should be representable in config.
        cfg = LoRATrainingConfig(load_in_4bit=True, use_dora=True)
        assert cfg.load_in_4bit and cfg.use_dora


class TestQuickRunSafety:
    """The quick run config must actually be quick — guard against
    accidentally setting it too large."""

    def test_quick_run_under_500_examples(self):
        cfg = LoRATrainingConfig.from_yaml("configs/lora_quick.yaml")
        assert (cfg.train_max_examples or 0) <= 500, (
            "lora_quick.yaml should keep train_max_examples small for fast iteration"
        )

    def test_quick_run_one_epoch(self):
        cfg = LoRATrainingConfig.from_yaml("configs/lora_quick.yaml")
        assert cfg.num_epochs == 1
