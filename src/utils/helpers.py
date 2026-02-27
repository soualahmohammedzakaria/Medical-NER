"""
helpers.py

Reproducibility, configuration loading, and device detection utilities.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set the random seed for Python, NumPy, and PyTorch.

    Also configures CuDNN for deterministic behaviour when a CUDA device
    is available (may reduce performance slightly).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file (default: config/config.yaml).

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    preference : str
        "auto" picks CUDA > MPS > CPU.  Pass "cpu", "cuda", or "mps"
        to force a specific backend.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)