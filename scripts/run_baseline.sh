#!/usr/bin/env bash
# Run the zero-shot baseline evaluation with default settings.
# Usage:
#   bash scripts/run_baseline.sh                # full test split
#   bash scripts/run_baseline.sh --max_examples 5   # smoke test

set -euo pipefail

# Move to repo root regardless of where this script is invoked from
cd "$(dirname "$0")/.."

python -m src.evaluation.evaluate_baseline \
    --model_id "Qwen/Qwen2-VL-2B-Instruct" \
    --split test \
    --output_dir results/baseline \
    --seed 42 \
    "$@"
