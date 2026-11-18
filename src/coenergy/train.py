"""Boucle d'entraînement : AdamW + OneCycle, AMP (CUDA uniquement), early stopping."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn

from .data import TargetScaler, iterate_batches
from .models import predict_encoded


@dataclass
class TrainConfig:
    """Hyperparamètres d'entraînement."""

    epochs: int = 15
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 4
    grad_clip: float = 1.0
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    seed: int = 42


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Perte Huber (smooth L1) moyenne pondérée par colonne, sur cibles standardisées."""
    per_col = nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=1.0).mean(0)
    return (per_col * w).sum() / w.sum()


@torch.no_grad()
def evaluate_loss(model: nn.Module, x, z, idx, scaler: TargetScaler, cfg: TrainConfig, device) -> float:
    """Perte moyenne sur un sous-ensemble (validation)."""
    model.eval()
    w = torch.tensor(cfg.loss_weights, device=device)
    tot, n = 0.0, 0
    for xb, zb in iterate_batches(x, z, idx, cfg.batch_size * 2, False, device):
        loss = weighted_huber(predict_encoded(model(xb).float(), scaler), zb, w)
        tot += loss.item() * len(xb)
        n += len(xb)
    return tot / n


def fit(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    scaler: TargetScaler,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Entraîne ``model`` et restaure les meilleurs poids (val loss minimale).

    Returns:
        Historique ``{"train": [...], "val": [...]}`` par époque.
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * ((len(idx_train) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.2)
    use_amp = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    w = torch.tensor(cfg.loss_weights, device=device)
    gen = torch.Generator().manual_seed(cfg.seed)

    hist: Dict[str, List[float]] = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, zb in iterate_batches(x, z, idx_train, cfg.batch_size, True, device, gen):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                z5 = model(xb)
            loss = weighted_huber(predict_encoded(z5.float(), scaler), zb, w)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            amp_scaler.step(opt)
            amp_scaler.update()
            sched.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        tr, va = tot / n, evaluate_loss(model, x, z, idx_val, scaler, cfg, device)
        hist["train"].append(tr)
        hist["val"].append(va)
        print(f"Epoch {ep + 1:02d}/{cfg.epochs} | train {tr:.4f} | val {va:.4f} | {time.time() - t0:.0f}s")
        if va < best - 1e-4:
            best, bad = va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
