"""Configuration globale : graine, périphérique de calcul (CUDA / MPS / CPU)."""
from __future__ import annotations

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Retourne le meilleur périphérique disponible : CUDA (Colab/Kaggle) > MPS (MacBook) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Fixe les graines Python, NumPy et PyTorch pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
