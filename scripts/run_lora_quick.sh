#!/usr/bin/env bash
# Run the LoRA quick-run training pipeline.
# Usage:
#   bash scripts/run_lora_quick.sh
#   bash scripts/run_lora_quick.sh --max_train 50    # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.training.train_lora --config configs/lora_quick.yaml "$@"
