"""Architectures PyTorch : encodeur TCN résiduel + têtes multi-indicateurs contraintes."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import TargetScaler
from .features import summary_features


class ResidualBlock(nn.Module):
    """Bloc résiduel de convolutions dilatées (non causales) avec GroupNorm."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class TCNEncoder(nn.Module):
    """Encodeur de séries ``[B, 2, L]`` -> vecteur ``[B, 3*channels]`` (moyenne, max, dernier pas).
    Un canal dérivé dT/dt est ajouté : la physique (bilan d'énergie C dT/dt = U - HLC (T - Text))
    relie directement la pente de T à U, d'où l'intérêt de l'exposer au réseau.
    """

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(3, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels, kernel_size, d, dropout) for d in dilations])
        self.out_dim = 3 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dT = F.pad(x[:, :1, 1:] - x[:, :1, :-1], (1, 0))  # [B,1,L]
        h = self.blocks(self.stem(torch.cat([x, dT * 10.0], dim=1)))
        return torch.cat([h.mean(-1), h.amax(-1), h[..., -1]], dim=1)


class LSTMEncoder(nn.Module):
    """Encodeur LSTM bidirectionnel ``[B, 2, L]`` -> ``[B, 3*2*hidden]`` (moyenne, max, dernier état).

    Même entrée que le TCN (T, U, dT/dt) et même résumé temporel, pour une comparaison équitable.
    """

    def __init__(self, hidden: int = 96, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(3, hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_dim = 3 * 2 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dT = F.pad(x[:, :1, 1:] - x[:, :1, :-1], (1, 0))
        seq = torch.cat([x, dT * 10.0], dim=1).transpose(1, 2)  # [B, L, 3]
        out, (h_n, _) = self.lstm(seq)
        last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # derniers états avant/arrière
        return torch.cat([out.mean(1), out.amax(1), last], dim=1)


class SpectralBranch(nn.Module):
    def __init__(self, n_bins: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.n_bins, self.out_dim = n_bins, out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(5 * n_bins), nn.Linear(5 * n_bins, 256), nn.GELU(),
            nn.Linear(256, out_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            k = self.n_bins
            ft = torch.fft.rfft(xf[:, 0], dim=-1)[:, :k]
            fu = torch.fft.rfft(xf[:, 1], dim=-1)[:, :k]
            lt, lu = torch.log(ft.abs() + 1e-3), torch.log(fu.abs() + 1e-3)
            dphi = torch.angle(ft) - torch.angle(fu)
            feats = torch.cat([lt, lu, (lt - lu).clamp(-10, 10), torch.cos(dphi), torch.sin(dphi)], dim=1)
            return self.net(feats)


class StatsBranch(nn.Module):
    """Branche de statistiques globales (moyenne, écart-type, min, max de T et U, covariance, |dT|)."""

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.BatchNorm1d(11), nn.Linear(11, out_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.net(summary_features(x.float()))


class ThermalNet(nn.Module):

    def __init__(self, encoder: str = "tcn", hidden: int = 64, dropout: float = 0.1,
                 head_width: int = 256, use_aux: bool = False) -> None:
        super().__init__()
        if encoder == "tcn":
            self.encoder: nn.Module = TCNEncoder(channels=hidden, dropout=dropout)
        elif encoder == "lstm":
            self.encoder = LSTMEncoder(hidden=hidden, dropout=dropout)
        else:
            raise ValueError(f"encoder inconnu : {encoder}")
        d = self.encoder.out_dim  # type: ignore[attr-defined]
        self.spectral = SpectralBranch() if use_aux else None
        self.stats = StatsBranch() if use_aux else None
        if use_aux:
            d += self.spectral.out_dim + self.stats.out_dim  # type: ignore[union-attr]
        self.head_hb = nn.Sequential(nn.Linear(d, head_width), nn.GELU(), nn.Linear(head_width, 2))
        self.head_tau = nn.Sequential(nn.Linear(d, head_width), nn.GELU(), nn.Linear(head_width, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne ``z5 [B,5]`` (espace standardisé) ; décoder avec :func:`predict_encoded`."""
        h = self.encoder(x)
        if self.spectral is not None:
            h = torch.cat([h, self.spectral(x).to(h.dtype), self.stats(x).to(h.dtype)], dim=1)
        return torch.cat([self.head_hb(h), self.head_tau(h)], dim=1)


def predict_encoded(z5: torch.Tensor, scaler: TargetScaler) -> torch.Tensor:
    """Convertit ``z5`` en espace standardisé complet ``z6`` ``[B,6]`` (HLC dérivé inclus).

    Utilisé par la perte pour comparer toutes les cibles, HLC compris, aux cibles encodées.
    """
    phys = scaler.decode_free(z5)
    return scaler.encode(phys)


def count_parameters(model: nn.Module) -> int:
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
