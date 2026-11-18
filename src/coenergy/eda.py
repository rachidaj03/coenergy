"""Analyse exploratoire : statistiques, contraintes physiques, graphiques."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import Simulations


def target_summary(sims: Simulations) -> pd.DataFrame:
    """Statistiques descriptives des 6 indicateurs (min, quantiles, max, asymétrie)."""
    d = pd.DataFrame(sims.y.numpy(), columns=sims.labels)
    s = d.describe(percentiles=[0.01, 0.5, 0.99]).T
    s["skew"] = d.skew()
    s["n_nonpositive"] = (d <= 0).sum()
    return s.round(3)


def check_physics(sims: Simulations) -> pd.DataFrame:
    """Vérifie HLC = somme Hb et l'ordre tau_b >= tau_inf >= tau_n sur les données."""
    y = sims.y
    gap = (y[:, 0] - y[:, 1:3].sum(1)).abs()
    return pd.DataFrame(
        {
            "valeur": [
                gap.max().item(),
                gap.median().item(),
                100 * (y[:, 3] >= y[:, 5]).float().mean().item(),
                100 * (y[:, 5] >= y[:, 4]).float().mean().item(),
            ]
        },
        index=["max|HLC-sum(Hb)|", "médiane|HLC-sum(Hb)|", "% tau_b>=tau_inf", "% tau_inf>=tau_n"],
    )


def input_summary(sims: Simulations, n: int = 50_000) -> pd.DataFrame:
    """Statistiques globales de T et U (sous-échantillon), dont fraction de NaN."""
    x = sims.x[:n]
    rows = {}
    for i, name in enumerate(["T_int", "U"]):
        c = x[:, i]
        rows[name] = {
            "min": c.min().item(), "max": c.max().item(), "mean": c.mean().item(),
            "std": c.std().item(), "nan_%": 100 * torch.isnan(c).float().mean().item(),
            "std_inter_sim": c.mean(1).std().item(),
        }
    return pd.DataFrame(rows).T.round(3)


def plot_target_distributions(sims: Simulations, max_n: int = 200_000) -> plt.Figure:
    """Histogrammes (échelle log en x) des 6 indicateurs."""
    y = sims.y[:max_n].numpy()
    fig, ax = plt.subplots(2, 3, figsize=(14, 6))
    for a, j, lab in zip(ax.ravel(), range(6), sims.labels):
        v = y[:, j]
        v = v[v > 0]
        a.hist(np.log10(v), bins=80, color="#2a7f62")
        a.set_title(lab)
        a.set_xlabel("log10(valeur)")
    fig.tight_layout()
    return fig


def plot_examples(sims: Simulations, idx: Optional[list] = None) -> plt.Figure:
    """Séries T_int(t) et U(t) pour quelques simulations."""
    idx = idx or [0, 1, 2]
    fig, ax = plt.subplots(2, len(idx), figsize=(5 * len(idx), 6), sharex=True)
    ax = np.atleast_2d(ax).reshape(2, -1)
    for k, i in enumerate(idx):
        ax[0, k].plot(sims.x[i, 0].numpy(), color="crimson")
        ax[1, k].plot(sims.x[i, 1].numpy(), color="steelblue")
        lbl = ", ".join(f"{n}={v:.1f}" for n, v in zip(sims.labels, sims.y[i].tolist()))
        ax[0, k].set_title(f"sim {i}\n{lbl}", fontsize=7)
        ax[0, k].set_ylabel("T_int")
        ax[1, k].set_ylabel("U")
        ax[1, k].set_xlabel("pas de temps")
    fig.tight_layout()
    return fig


def plot_correlations(feats: torch.Tensor, sims: Simulations, feat_names: list, n: int = 200_000) -> plt.Figure:
    """Corrélation de Spearman (rangs) entre features agrégées et log-indicateurs."""
    f = pd.DataFrame(feats[:n].numpy(), columns=feat_names)
    y = pd.DataFrame(torch.log(sims.y[:n].clamp_min(1e-3)).numpy(), columns=sims.labels)
    c = pd.concat([f, y], axis=1).corr(method="spearman").loc[feat_names, sims.labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sims.labels)), sims.labels, rotation=45)
    ax.set_yticks(range(len(feat_names)), feat_names)
    fig.colorbar(im)
    ax.set_title("Corrélation de Spearman : features vs log(indicateurs)")
    fig.tight_layout()
    return fig
