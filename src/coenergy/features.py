"""Features agrégées + baseline Ridge (sans réseau de neurones, en PyTorch pur)."""
from __future__ import annotations

import torch

from .data import TargetScaler


def summary_features(x: torch.Tensor, chunk: int = 50_000) -> torch.Tensor:
    if len(x) > chunk:
        return torch.cat([summary_features(x[i : i + chunk], chunk) for i in range(0, len(x), chunk)], dim=0)
    t, u = x[:, 0], x[:, 1]
    dt = t[:, 1:] - t[:, :-1]
    cov = ((t - t.mean(1, keepdim=True)) * (u - u.mean(1, keepdim=True))).mean(1, keepdim=True)
    feats = [
        t.mean(1, keepdim=True), t.std(1, keepdim=True), t.amin(1, keepdim=True), t.amax(1, keepdim=True),
        u.mean(1, keepdim=True), u.std(1, keepdim=True), u.amin(1, keepdim=True), u.amax(1, keepdim=True),
        cov, dt.abs().mean(1, keepdim=True), dt.std(1, keepdim=True),
    ]
    return torch.cat(feats, dim=1)


class RidgeBaseline:
    """Régression Ridge multi-sorties (forme close) sur features agrégées -> 5 sorties libres."""
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.w: torch.Tensor | None = None

    def fit(self, feats: torch.Tensor, z6: torch.Tensor) -> "RidgeBaseline":
        """Ajuste sur ``z6`` (cibles encodées) en ne gardant que les 5 colonnes libres."""
        a = torch.cat([feats, torch.ones(len(feats), 1)], dim=1).double()
        reg = self.alpha * torch.eye(a.shape[1], dtype=torch.double)
        reg[-1, -1] = 0.0
        self.w = torch.linalg.solve(a.T @ a + reg, a.T @ z6[:, 1:].double()).float()
        return self

    def predict_physical(self, feats: torch.Tensor, scaler: TargetScaler) -> torch.Tensor:
        """Prédit les 6 indicateurs physiques (HLC = somme des Hb)."""
        assert self.w is not None, "fit() doit être appelé d'abord"
        a = torch.cat([feats, torch.ones(len(feats), 1)], dim=1)
        return scaler.decode_free(a @ self.w)
