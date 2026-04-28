"""Set RNG seeds across Python, NumPy, and PyTorch for reproducibility."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs.

    Parameters
    ----------
    seed : int
        Seed value applied to all RNGs.
    deterministic : bool
        If True, additionally configure PyTorch for fully deterministic
        cuDNN ops. This is **slower** and only relevant for training, so
        we leave it False by default; greedy inference is already
        deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
