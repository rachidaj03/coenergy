"""Chargement, assemblage des cibles, découpage et normalisation des simulations.

Convention des indicateurs (identique au PDF) :
``[HLC, HLC_mit, HLC_ext, tau_b, tau_n, tau_inf]``  (6 colonnes, unités physiques).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import torch

from .loader import load_single_batch

CDT_LABELS: List[str] = ["tau_b", "tau_n", "tau_inf"]
EPS: float = 1e-3 


@dataclass
class Simulations:
    """Séries d'entrée et indicateurs cibles (tenseurs CPU float32).

    Attributes:
        x: ``[N, 2, L]`` — canal 0 = T_int, canal 1 = U.
        y: ``[N, 6]`` — indicateurs physiques (``labels``).
        labels: noms des 6 colonnes de ``y``.
    """

    x: torch.Tensor
    y: torch.Tensor
    labels: List[str]

    def __len__(self) -> int:
        return self.x.shape[0]


def load_simulations(
    folder: Union[str, Path],
    max_batches: Optional[int] = None,
    round_inputs: bool = True,
) -> Simulations:
    """Charge ``batch_*.pt`` et empile entrées / cibles sans doublon mémoire.

    Args:
        folder: dossier contenant les batchs.
        max_batches: limite le nombre de batchs (utile sur MacBook). ``None`` = tous.
        round_inputs: arrondit T à 2 décimales et U à 1 (précision des capteurs, cf. PDF).
    """
    files = sorted(
        glob.glob(str(Path(folder) / "batch_*.pt")),
        key=lambda p: int(Path(p).stem.split("_")[1]),
    )
    if not files:
        raise FileNotFoundError(f"Aucun batch_*.pt dans {folder}")
    if max_batches is not None:
        files = files[:max_batches]

    x = y = None
    n = 0
    hb_labels: List[str] = []
    for f in files:
        b = load_single_batch(f)
        t, u = b.indoor_temperature.float(), b.indoor_U.float()
        if round_inputs:
            t, u = t.round(decimals=2), u.round(decimals=1)
        yb = torch.cat([b.HLC.float().unsqueeze(1), b.Hb.float(), b.CDT.float()], dim=1)
        if x is None:
            cap = len(t) * len(files)
            x = torch.empty(cap, 2, t.shape[1])
            y = torch.empty(cap, yb.shape[1])
        x[n : n + len(t), 0], x[n : n + len(t), 1] = t, u
        y[n : n + len(t)] = yb
        n += len(t)
        hb_labels = list(b.Hb_labels)
        del b, t, u, yb
    return Simulations(x=x[:n], y=y[:n], labels=["HLC"] + hb_labels + CDT_LABELS)


def split_indices(
    n: int, fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Découpage aléatoire reproductible train / val / test par simulation."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("Les fractions doivent sommer à 1.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_tr, n_va = int(n * fractions[0]), int(n * fractions[1])
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]


class InputScaler:
    """Standardisation globale par canal (T, U) — préserve la dynamique temporelle."""

    def __init__(self) -> None:
        self.mean = torch.zeros(1, 2, 1)
        self.std = torch.ones(1, 2, 1)

    def fit(self, x: torch.Tensor, max_samples: int = 200_000) -> "InputScaler":
        """Estime moyenne/écart-type sur (un sous-échantillon du) train."""
        xs = x[:max_samples]
        self.mean = xs.mean(dim=(0, 2), keepdim=True)
        self.std = xs.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        return self

    def transform_(self, x: torch.Tensor) -> torch.Tensor:
        """Normalisation en place (évite de dupliquer plusieurs Go)."""
        return x.sub_(self.mean).div_(self.std)


class TargetScaler:
    """Transformation des cibles en espace standardisé, colonne par colonne.

    Deux paramétrisations (``mode``) :

    * ``"log"`` : ``z = (log y - mu) / sigma`` pour les 6 indicateurs.
    * ``"ratio"`` : paramétrisation **décorrélée** des constantes de temps, qui exploite leur forte
      corrélation (tau_b ~ tau_inf) : colonnes ``[log HLC, log HLC_mit, log HLC_ext, log tau_b,
      logit(tau_inf/tau_b), logit(tau_n/tau_inf)]``. Au décodage, ``tau_inf = tau_b*sigmoid(.)`` et
      ``tau_n = tau_inf*sigmoid(.)`` : l'ordre ``tau_b > tau_inf > tau_n`` est **garanti par construction**.

    Dans les deux cas HLC n'est jamais une sortie libre (somme des Hb au décodage).
    """

    def __init__(self, mode: str = "log") -> None:
        if mode not in ("log", "ratio"):
            raise ValueError(f"mode inconnu : {mode}")
        self.mode = mode
        self.mu = torch.zeros(6)
        self.sigma = torch.ones(6)

    def _transform(self, y: torch.Tensor) -> torch.Tensor:
        """Physique ``[N,6]`` -> espace transformé non standardisé ``[N,6]``."""
        l = torch.log(y.clamp_min(EPS))
        if self.mode == "log":
            return l
        r1 = (y[:, 5] / y[:, 3].clamp_min(EPS)).clamp(1e-4, 1 - 1e-4)  # tau_inf / tau_b
        r2 = (y[:, 4] / y[:, 5].clamp_min(EPS)).clamp(1e-4, 1 - 1e-4)  # tau_n / tau_inf
        logit = lambda r: torch.log(r) - torch.log1p(-r)
        return torch.stack([l[:, 0], l[:, 1], l[:, 2], l[:, 3], logit(r1), logit(r2)], dim=1)

    def fit(self, y: torch.Tensor) -> "TargetScaler":
        t = self._transform(y)
        self.mu, self.sigma = t.mean(0), t.std(0).clamp_min(1e-6)
        return self

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """Physique ``[N,6]`` -> espace standardisé ``[N,6]``."""
        mu, sg = self.mu.to(y.device), self.sigma.to(y.device)
        return (self._transform(y) - mu) / sg

    def decode_free(self, z5: torch.Tensor) -> torch.Tensor:
        """Sorties libres ``[N,5]`` (HLC_mit, HLC_ext, tau_b, tau_n, tau_inf en mode log ;
        HLC_mit, HLC_ext, tau_b, r1, r2 en mode ratio) -> indicateurs physiques ``[N,6]``
        ``[HLC, HLC_mit, HLC_ext, tau_b, tau_n, tau_inf]`` avec **HLC = HLC_mit + HLC_ext exact**.
        """
        mu, sg = self.mu.to(z5.device), self.sigma.to(z5.device)
        raw = z5 * sg[1:] + mu[1:]
        if self.mode == "log":
            phys = torch.exp(raw.clamp(-20.0, 20.0))
        else:
            h = torch.exp(raw[:, :3].clamp(-20.0, 20.0))  # HLC_mit, HLC_ext, tau_b
            tau_inf = h[:, 2] * torch.sigmoid(raw[:, 3])
            tau_n = tau_inf * torch.sigmoid(raw[:, 4])
            phys = torch.stack([h[:, 0], h[:, 1], h[:, 2], tau_n, tau_inf], dim=1)
        hlc = phys[:, :2].sum(dim=1, keepdim=True)
        return torch.cat([hlc, phys], dim=1)


def iterate_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Itère sur des mini-batchs déjà normalisés, en transférant uniquement le batch courant.

    Plus rapide qu'un ``DataLoader`` ici : les données sont en RAM et l'indexation est vectorisée.
    """
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=generator)]
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        yield x[b].to(device, non_blocking=True), y[b].to(device, non_blocking=True)


def make_targets(
    y: torch.Tensor, idx_train: torch.Tensor, mode: str = "log"
) -> Tuple[torch.Tensor, TargetScaler]:
    """Ajuste un :class:`TargetScaler` sur le train et encode toutes les cibles (``mode`` : log / ratio)."""
    ts = TargetScaler(mode).fit(y[idx_train])
    return ts.encode(y), ts


def prepare_data(
    sims: Simulations, seed: int = 42, fractions: Sequence[float] = (0.8, 0.1, 0.1), mode: str = "log"
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], InputScaler, TargetScaler]:
    """Pipeline complet : split, ajustement des scalers sur le train, normalisation en place.

    Returns:
        ``x_norm [N,2,L]``, ``z [N,6]`` (cibles encodées), ``(idx_tr, idx_va, idx_te)``,
        scalers d'entrée et de cible. Les cibles physiques restent dans ``sims.y``.
    """
    idx = split_indices(len(sims), tuple(fractions), seed)  # type: ignore[arg-type]
    xs = InputScaler().fit(sims.x[idx[0][:200_000]])  # sous-échantillon : évite une copie de plusieurs Go
    x = xs.transform_(sims.x)
    z, ts = make_targets(sims.y, idx[0], mode)
    return x, z, idx, xs, ts
