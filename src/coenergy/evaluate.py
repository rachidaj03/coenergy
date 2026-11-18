"""Inférence et métriques par famille d'indicateurs (HLC, Hb, CDT) + analyse d'erreurs."""
from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from .data import TargetScaler, iterate_batches
from .train import TrainConfig, fit

FAMILIES: Dict[str, List[str]] = {
    "HLC": ["HLC"],
    "Hb": ["HLC_mit", "HLC_ext"],
    "CDT": ["tau_b", "tau_n", "tau_inf"],
}


@torch.no_grad()
def predict_physical(
    model: nn.Module, x: torch.Tensor, idx: torch.Tensor, scaler: TargetScaler,
    device: torch.device, batch_size: int = 2048,
) -> torch.Tensor:
    """Prédictions physiques ``[len(idx), 6]`` (CPU) pour les simulations ``idx``."""
    model.eval()
    dummy = torch.zeros(len(x), 1)
    out = [
        scaler.decode_free(model(xb).float()).cpu()
        for xb, _ in iterate_batches(x, dummy, idx, batch_size, False, device)
    ]
    return torch.cat(out, 0)


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, labels: List[str]) -> pd.DataFrame:
    """MAE, RMSE, R², erreur relative médiane et P90 par indicateur (unités physiques)."""
    err = y_pred - y_true
    ss_res = (err ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    rel = err.abs() / y_true.abs().clamp_min(1e-6)
    rows = {
        "MAE": err.abs().mean(0),
        "RMSE": (err ** 2).mean(0).sqrt(),
        "R2": 1 - ss_res / ss_tot,
        "MedRelErr_%": rel.median(0).values * 100,
        "P90RelErr_%": rel.quantile(0.9, dim=0) * 100 if len(rel) < 16_000_000 else rel.median(0).values * 100,
    }
    return pd.DataFrame({k: v.numpy() for k, v in rows.items()}, index=labels).round(4)


def physical_consistency(y_pred: torch.Tensor) -> Dict[str, float]:
    """Contrôles physiques : écart HLC vs somme(Hb) et taux de violation de tau_b >= tau_inf >= tau_n (ordre observé)."""
    gap = (y_pred[:, 0] - y_pred[:, 1:3].sum(1)).abs().max().item()
    order = ((y_pred[:, 3] < y_pred[:, 5]) | (y_pred[:, 5] < y_pred[:, 4])).float().mean().item()
    return {"max |HLC - sum(Hb)|": gap, "violation ordre tau_b>=tau_inf>=tau_n (%)": 100 * order}


def error_by_feature(
    feature: torch.Tensor, rel_err: torch.Tensor, name: str, bins: int = 10
) -> pd.DataFrame:
    """Erreur relative médiane par décile d'une variable explicative (analyse critique)."""
    q = torch.quantile(feature, torch.linspace(0, 1, bins + 1))
    cat = torch.bucketize(feature, q[1:-1])
    rows = [
        {name: f"[{q[b]:.3g}, {q[b + 1]:.3g}]", "median_rel_err_%": 100 * rel_err[cat == b].median().item()}
        for b in range(bins) if (cat == b).any()
    ]
    return pd.DataFrame(rows)


def excitation_features(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    n, _, length = x.shape
    u = x[:, 1].float()
    spec = torch.fft.rfft(u - u.mean(1, keepdim=True), dim=-1).abs() ** 2  # [N, L/2+1]
    freqs = torch.arange(spec.shape[1], dtype=torch.float32) / length
    power = spec[:, 1:]
    f = freqs[1:]
    peak = power.argmax(1) + 1
    period = length / peak.float()
    centroid = (power * f).sum(1) / power.sum(1).clamp_min(1e-9)
    bandwidth = ((power * (f - centroid[:, None]) ** 2).sum(1) / power.sum(1).clamp_min(1e-9)).sqrt()
    out = {
        "periode_dominante_U": period,
        "largeur_bande_U": bandwidth,
        "amplitude_T": x[:, 0].amax(1) - x[:, 0].amin(1),
        "std_U": u.std(1),
    }
    return out


def error_table(feature: torch.Tensor, rel_err: torch.Tensor, labels: List[str], bins: int = 8) -> pd.DataFrame:
    """Erreur relative médiane (%) par tranche (quantiles) d'une variable, pour chaque indicateur."""
    q = torch.unique(torch.quantile(feature, torch.linspace(0, 1, bins + 1)))
    cat = torch.bucketize(feature, q[1:-1])
    rows = {}
    for b in range(len(q) - 1):
        m = cat == b
        if m.any():
            rows[f"[{q[b]:.3g}, {q[b + 1]:.3g}]"] = 100 * rel_err[m].median(0).values.numpy()
    return pd.DataFrame(rows, index=labels).T.round(2)


def plot_error_heatmaps(features: Dict[str, torch.Tensor], rel_err: torch.Tensor, labels: List[str],
                        bins: int = 8) -> plt.Figure:
    """Cartes thermiques : erreur relative médiane par indicateur selon chaque propriété de l'excitation."""
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4.5))
    for ax, (name, f) in zip(axes, features.items()):
        t = error_table(f, rel_err, labels, bins)
        im = ax.imshow(t.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(labels)), labels, rotation=45)
        ax.set_yticks(range(len(t)), t.index, fontsize=7)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, label="erreur rel. médiane (%)")
    fig.tight_layout()
    return fig


def learning_curve(
    build_model, fractions: List[float], x: torch.Tensor, z: torch.Tensor,
    idx_train: torch.Tensor, idx_val: torch.Tensor, idx_test: torch.Tensor,
    scaler: TargetScaler, y_true_test: torch.Tensor, labels: List[str],
    cfg: TrainConfig, device: torch.device,
) -> pd.DataFrame:
    """Entraîne un modèle neuf sur une fraction du train et mesure le R² test (val/test inchangés).

    ``idx_train`` étant une permutation aléatoire, ses ``k`` premiers indices forment un sous-échantillon aléatoire.
    """
    rows = []
    for fr in fractions:
        k = int(len(idx_train) * fr)
        model = build_model()
        print(f"\n--- courbe d'apprentissage : {k:,} simulations d'entraînement ---")
        fit(model, x, z, idx_train[:k], idx_val, scaler, cfg, device)
        met = regression_metrics(y_true_test, predict_physical(model, x, idx_test, scaler, device), labels)
        rows.append({"n_train": k, "R2_moyen": met["R2"].mean(), **{f"R2_{l}": met.loc[l, "R2"] for l in labels}})
        del model
    return pd.DataFrame(rows)


def calibration_table(y_true: torch.Tensor, y_pred: torch.Tensor, labels: List[str], bins: int = 10) -> pd.DataFrame:
    """Pente de régression ``prédit ~ réel`` par indicateur (1 = calibré ; < 1 = retour vers la moyenne).

    Un modèle peu informatif prédit proche de la moyenne : pente faible, et erreurs relatives plus grandes
    aux extrêmes de la cible (artefact à ne pas confondre avec un effet de l'excitation).
    """
    rows = {}
    for j, lab in enumerate(labels):
        t, p = y_true[:, j], y_pred[:, j]
        slope = ((t - t.mean()) * (p - p.mean())).mean() / t.var(unbiased=False)
        rows[lab] = {"pente pred~reel": slope.item(), "corr": torch.corrcoef(torch.stack([t, p]))[0, 1].item()}
    return pd.DataFrame(rows).T.round(3)


def plot_calibration(y_true: torch.Tensor, y_pred: torch.Tensor, labels: List[str], bins: int = 10) -> plt.Figure:
    """Moyenne prédite par décile de la valeur réelle, par indicateur (diagonale = calibration parfaite)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, j, lab in zip(axes.ravel(), range(len(labels)), labels):
        t, p = y_true[:, j], y_pred[:, j]
        q = torch.quantile(t, torch.linspace(0, 1, bins + 1))
        cat = torch.bucketize(t, q[1:-1])
        mt = torch.stack([t[cat == b].mean() for b in range(bins)])
        mp = torch.stack([p[cat == b].mean() for b in range(bins)])
        ax.plot(mt, mp, "o-"); ax.plot([mt.min(), mt.max()], [mt.min(), mt.max()], "r--")
        ax.set_title(lab); ax.set_xlabel("réel (moyenne du décile)"); ax.set_ylabel("prédit")
    fig.tight_layout()
    return fig
