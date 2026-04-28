"""Pytest configuration: ensure ``src`` is importable from tests."""

import sys
from pathlib import Path

# Add the repo root to sys.path so ``import src...`` works under pytest
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
