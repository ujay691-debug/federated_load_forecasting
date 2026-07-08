"""Local season residual correction experiment for decomposed net-load forecasting.

This script is standalone. It validates a local, single-client framework:

1. Decompose net_load into trend + season with Autoformer-style moving average.
2. Forecast trend with a causal TCN.
3. Forecast season with one of three encoders plus the same MLP decoder.
4. Optionally correct the season forecast with causal, non-overlapping historical
   season residual blocks.

No federated learning, weather features, MoE, distillation, or spectral flatness
regularization is used here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from utils.runtime_env import ensure_conda_dll_paths

    ensure_conda_dll_paths()
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


DEFAULT_CLIENT_IDS = list(range(1, 10))
DEFAULT_ENCODERS = ["TimesNet", "ModernTCN", "Transformer"]
DEFAULT_CORRECTORS = ["GRU", "LowRankAdapter"]
DEFAULT_RESIDUAL_LENGTHS = [6, 12, 24]
EPS = 1e-6


@dataclass
class TorchScalerState:
    name: str
    mean: float = 0.0
    scale: float = 1.0
    min_value: float = 0.0


class LocalResidualDataset(Dataset):
    def __init__(
        self,
        starts: np.ndarray,
        trend_scaled: np.ndarray,
        season_scaled: np.ndarray,
        net_scaled: np.ndarray,
        net_raw: np.ndarray,
        seq_len: int,
        horizon: int,
        residual_length: int = 0,
    ) -> None:
        self.starts = np.asarray(starts, dtype=np.int64)
        self.trend_scaled = trend_scaled.astype(np.float32)
        self.season_scaled = season_scaled.astype(np.float32)
        self.net_scaled = net_scaled.astype(np.float32)
        self.net_raw = net_raw.astype(np.float32)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.residual_length = int(residual_length)
        if self.residual_length % self.horizon != 0:
            raise ValueError("residual_length must be a multiple of horizon.")
        self.num_residual_blocks = self.residual_length // self.horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = int(self.starts[idx])
        l = self.seq_len
        h = self.horizon

        item: Dict[str, torch.Tensor] = {
            "x_trend": torch.from_numpy(self.trend_scaled[t - l : t, None]),
            "x_season": torch.from_numpy(self.season_scaled[t - l : t, None]),
            "y_trend": torch.from_numpy(self.trend_scaled[t : t + h]),
            "y_season": torch.from_numpy(self.season_scaled[t : t + h]),
            "y_net_scaled": torch.from_numpy(self.net_scaled[t : t + h]),
            "y_net_raw": torch.from_numpy(self.net_raw[t : t + h]),
        }

        if self.num_residual_blocks > 0:
            ctx_blocks: List[np.ndarray] = []
            hist_blocks: List[np.ndarray] = []
            # Non-overlapping causal residual blocks, ordered oldest -> newest.
            #
            # For q = m * H and current target S[t:t+H-1], block j predicts
            # Y_hist_j = S[t-jH:t-(j-1)H-1] from the 48 hours immediately
            # before that block: X_ctx_j = S[t-jH-L:t-jH-1].
            # The newest block is j=1, ending at t-1. No future target point
            # S[t:t+H-1] is used by the corrector.
            for j in range(self.num_residual_blocks, 0, -1):
                block_start = t - j * h
                ctx_start = block_start - l
                ctx_blocks.append(self.season_scaled[ctx_start:block_start, None])
                hist_blocks.append(self.season_scaled[block_start : block_start + h])
            item["residual_ctx"] = torch.from_numpy(np.stack(ctx_blocks, axis=0).astype(np.float32))
            item["residual_hist"] = torch.from_numpy(np.stack(hist_blocks, axis=0).astype(np.float32))

        return item


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return F.relu(out + residual)


class TrendTCNForecaster(nn.Module):
    def __init__(
        self,
        horizon: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        in_channels = 1
        for dilation in dilations:
            blocks.append(TCNResidualBlock(in_channels, hidden_channels, kernel_size, dilation, dropout))
            in_channels = hidden_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        hidden = self.network(x)
        return self.head(hidden[:, :, -1])


class Inception2D(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.conv_1x3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1), groups=1)
        self.conv_3x1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0), groups=1)
        self.conv_3x3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1), groups=1)
        self.mix = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (self.conv_1x3(x) + self.conv_3x1(x) + self.conv_3x3(x)) / 3.0
        return self.mix(F.gelu(out))


class TimesBlock(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, top_k: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = Inception2D(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _periods_from_fft(self, x: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        # x: batch, seq_len, hidden_dim
        fft = torch.fft.rfft(x, dim=1)
        amplitude = fft.abs().mean(dim=(0, 2)).clone()
        if len(amplitude) > 0:
            amplitude[0] = 0.0
        k = min(self.top_k, max(1, len(amplitude) - 1))
        top_values, top_indices = torch.topk(amplitude, k=k)
        periods: List[int] = []
        for idx in top_indices.detach().cpu().tolist():
            idx = max(int(idx), 1)
            periods.append(max(1, self.seq_len // idx))
        weights = torch.softmax(top_values, dim=0)
        return periods, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = x.shape
        periods, weights = self._periods_from_fft(x)
        period_outputs: List[torch.Tensor] = []
        for period in periods:
            padded_len = int(math.ceil(seq_len / period) * period)
            pad_len = padded_len - seq_len
            x_pad = F.pad(x, (0, 0, 0, pad_len)) if pad_len > 0 else x
            # batch, padded_len, hidden -> batch, hidden, blocks, period
            x_2d = x_pad.reshape(bsz, padded_len // period, period, hidden_dim).permute(0, 3, 1, 2)
            y_2d = self.conv(x_2d)
            y = y_2d.permute(0, 2, 3, 1).reshape(bsz, padded_len, hidden_dim)[:, :seq_len, :]
            period_outputs.append(y)
        stacked = torch.stack(period_outputs, dim=-1)
        weights = weights.to(device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        out = (stacked * weights).sum(dim=-1)
        return self.norm(x + self.dropout(out))


class TimesNetEncoder(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int = 64, top_k: int = 3, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([TimesBlock(seq_len, hidden_dim, top_k, dropout) for _ in range(layers)])
        self.output_dim = hidden_dim * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(x)
        for block in self.blocks:
            hidden = block(hidden)
        return pool_last_mean_max(hidden)


class ModernTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, groups=hidden_dim, padding=0)
        self.ffn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)
        self.left_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.dwconv(F.pad(y, (self.left_padding, 0)))
        y = self.ffn(y).transpose(1, 2)
        return residual + self.dropout(y)


class ModernTCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, layers: int = 3, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([ModernTCNBlock(hidden_dim, kernel_size, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(x)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return pool_last_mean_max(hidden)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerSeasonEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_dim = d_model * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.positional_encoding(self.embedding(x))
        hidden = self.encoder(hidden)
        return pool_last_mean_max(hidden)


class SeasonBaseModel(nn.Module):
    def __init__(self, encoder_name: str, seq_len: int, horizon: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        encoder_name = normalize_encoder_name(encoder_name)
        if encoder_name == "TimesNet":
            self.encoder = TimesNetEncoder(seq_len=seq_len, hidden_dim=hidden_dim, dropout=dropout)
        elif encoder_name == "ModernTCN":
            self.encoder = ModernTCNEncoder(hidden_dim=hidden_dim, kernel_size=15, dropout=dropout)
        elif encoder_name == "Transformer":
            self.encoder = TransformerSeasonEncoder(seq_len=seq_len, d_model=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unsupported season encoder: {encoder_name}")
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class GRUResidualCorrector(nn.Module):
    def __init__(self, horizon: int, hidden_size: int = 64, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon))

    def forward(self, residual_history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(residual_history.unsqueeze(-1))
        return self.head(hidden[-1])


class LowRankAdapterCorrector(nn.Module):
    def __init__(self, residual_length: int, horizon: int, rank: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(residual_length, rank), nn.ReLU(), nn.Linear(rank, horizon))

    def forward(self, residual_history: torch.Tensor) -> torch.Tensor:
        return self.net(residual_history)


def pool_last_mean_max(hidden: torch.Tensor) -> torch.Tensor:
    return torch.cat([hidden[:, -1, :], hidden.mean(dim=1), hidden.max(dim=1).values], dim=-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(values: Sequence[str]) -> List[int]:
    out: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def parse_str_list(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def normalize_encoder_name(name: str) -> str:
    mapping = {"timesnet": "TimesNet", "moderntcn": "ModernTCN", "transformer": "Transformer"}
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported season encoder '{name}'.")
    return mapping[key]


def normalize_corrector_name(name: str) -> str:
    mapping = {"gru": "GRU", "lowrankadapter": "LowRankAdapter", "lowrank": "LowRankAdapter"}
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported residual corrector '{name}'.")
    return mapping[key]


def resolve_device(device_name: str) -> torch.device:
    if device_name.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is unavailable. Falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(device_name)


def client_csv_path(data_dir: Path, client_id: int) -> Path:
    return data_dir / f"client_{client_id}_load_weather_1h.csv"


def read_client_series(data_path: Path, start_time: str, end_time: str) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Client CSV not found: {data_path}")
    df = pd.read_csv(data_path)
    if "local_aest_time" not in df.columns:
        raise ValueError(f"{data_path} is missing local_aest_time.")
    if "net_load" in df.columns:
        net_load = pd.to_numeric(df["net_load"], errors="coerce")
    elif {"gc", "gg"}.issubset(df.columns):
        net_load = pd.to_numeric(df["gc"], errors="coerce") - pd.to_numeric(df["gg"], errors="coerce")
    else:
        raise ValueError(f"{data_path} must contain net_load or both gc and gg.")
    out = pd.DataFrame(
        {
            "local_aest_time": pd.to_datetime(df["local_aest_time"], errors="coerce"),
            "net_load": net_load.astype(float),
        }
    )
    out = out.dropna(subset=["local_aest_time", "net_load"])
    out = out.sort_values("local_aest_time").drop_duplicates(subset=["local_aest_time"], keep="first")
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    out = out[(out["local_aest_time"] >= start) & (out["local_aest_time"] <= end)]
    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No data in requested range for {data_path}.")
    return out


def moving_average_autoformer_same(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    total_pad = window - 1
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    trend = np.convolve(padded, kernel, mode="valid")
    if len(trend) != len(x):
        raise RuntimeError("Autoformer same moving average changed sequence length.")
    return trend


def decompose_net_load(net_load: np.ndarray, smooth_window: int) -> Tuple[np.ndarray, np.ndarray]:
    trend = moving_average_autoformer_same(net_load, smooth_window)
    season = np.asarray(net_load, dtype=np.float64) - trend
    return trend, season


def make_sklearn_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "none":
        return None
    raise ValueError("scaler must be standard, minmax, or none.")


def fit_transform_scaler(name: str, values: np.ndarray, split_idx: int) -> Tuple[object, TorchScalerState, np.ndarray]:
    scaler = make_sklearn_scaler(name)
    values_2d = values.reshape(-1, 1)
    if scaler is None:
        return None, TorchScalerState(name="none"), values.astype(np.float32)
    scaler.fit(values[:split_idx].reshape(-1, 1))
    transformed = scaler.transform(values_2d).reshape(-1).astype(np.float32)
    if name == "standard":
        state = TorchScalerState(
            name="standard",
            mean=float(scaler.mean_[0]),
            scale=float(scaler.scale_[0]) if abs(float(scaler.scale_[0])) > EPS else 1.0,
        )
    else:
        state = TorchScalerState(
            name="minmax",
            scale=float(scaler.scale_[0]) if abs(float(scaler.scale_[0])) > EPS else 1.0,
            min_value=float(scaler.min_[0]),
        )
    return scaler, state, transformed


def inverse_transform_numpy(scaler, values: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(values, dtype=np.float64)
    original_shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape).astype(np.float64)


def inverse_transform_torch(values: torch.Tensor, state: TorchScalerState) -> torch.Tensor:
    if state.name == "standard":
        return values * state.scale + state.mean
    if state.name == "minmax":
        return (values - state.min_value) / state.scale
    return values


def transform_real_to_scaled_torch(values: torch.Tensor, state: TorchScalerState) -> torch.Tensor:
    if state.name == "standard":
        return (values - state.mean) / state.scale
    if state.name == "minmax":
        return values * state.scale + state.min_value
    return values


def build_starts(n: int, split_idx: int, seq_len: int, horizon: int, residual_length: int, train: bool) -> np.ndarray:
    min_t = seq_len + residual_length
    if train:
        start = min_t
        stop = split_idx - horizon + 1
    else:
        start = max(split_idx, min_t)
        stop = n - horizon + 1
    if stop <= start:
        raise ValueError(
            f"Not enough samples: n={n}, split_idx={split_idx}, seq_len={seq_len}, "
            f"horizon={horizon}, residual_length={residual_length}, train={train}."
        )
    return np.arange(start, stop, dtype=np.int64)


def make_loader(dataset: Dataset, batch_size: int, device: torch.device, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def create_corrector(name: str, residual_length: int, horizon: int, args: argparse.Namespace) -> nn.Module:
    name = normalize_corrector_name(name)
    if name == "GRU":
        return GRUResidualCorrector(
            horizon=horizon,
            hidden_size=args.corrector_hidden_dim,
            layers=args.corrector_gru_layers,
            dropout=args.dropout,
        )
    if name == "LowRankAdapter":
        return LowRankAdapterCorrector(residual_length=residual_length, horizon=horizon, rank=args.low_rank)
    raise ValueError(f"Unsupported residual corrector: {name}")


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def compute_residual_history(
    season_model: nn.Module,
    residual_ctx: torch.Tensor,
    residual_hist: torch.Tensor,
) -> torch.Tensor:
    bsz, num_blocks, seq_len, channels = residual_ctx.shape
    ctx_flat = residual_ctx.reshape(bsz * num_blocks, seq_len, channels)
    pred_flat = season_model(ctx_flat)
    pred = pred_flat.reshape(bsz, num_blocks, -1)
    residual = residual_hist - pred
    return residual.reshape(bsz, num_blocks * residual.shape[-1])


def compute_base_outputs(
    trend_model: nn.Module,
    season_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    trend_state: TorchScalerState,
    season_state: TorchScalerState,
    net_state: TorchScalerState,
) -> Dict[str, torch.Tensor]:
    trend_hat = trend_model(batch["x_trend"])
    season_hat = season_model(batch["x_season"])
    y_hat_real = inverse_transform_torch(trend_hat, trend_state) + inverse_transform_torch(season_hat, season_state)
    y_hat_scaled = transform_real_to_scaled_torch(y_hat_real, net_state)
    return {
        "trend_hat": trend_hat,
        "season_base_hat": season_hat,
        "season_final_hat": season_hat,
        "y_hat_real": y_hat_real,
        "y_hat_scaled": y_hat_scaled,
        "delta": torch.zeros_like(season_hat),
    }


def compute_corrected_outputs(
    trend_model: nn.Module,
    season_model: nn.Module,
    corrector: nn.Module,
    batch: Dict[str, torch.Tensor],
    trend_state: TorchScalerState,
    season_state: TorchScalerState,
    net_state: TorchScalerState,
) -> Dict[str, torch.Tensor]:
    base = compute_base_outputs(trend_model, season_model, batch, trend_state, season_state, net_state)
    residual_history = compute_residual_history(season_model, batch["residual_ctx"], batch["residual_hist"])
    delta = corrector(residual_history)
    season_final = base["season_base_hat"] + delta
    y_hat_real = inverse_transform_torch(base["trend_hat"], trend_state) + inverse_transform_torch(
        season_final, season_state
    )
    y_hat_scaled = transform_real_to_scaled_torch(y_hat_real, net_state)
    base.update(
        {
            "season_final_hat": season_final,
            "y_hat_real": y_hat_real,
            "y_hat_scaled": y_hat_scaled,
            "delta": delta,
            "residual_history": residual_history,
        }
    )
    return base


def train_warmup(
    trend_model: nn.Module,
    season_model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    states: Tuple[TorchScalerState, TorchScalerState, TorchScalerState],
) -> List[Dict[str, float]]:
    trend_state, season_state, net_state = states
    trend_model.to(device)
    season_model.to(device)
    params = list(trend_model.parameters()) + list(season_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr_base)
    criterion = nn.L1Loss()
    logs: List[Dict[str, float]] = []

    for epoch in range(1, args.warmup_epochs + 1):
        trend_model.train()
        season_model.train()
        totals = {"train_loss": 0.0, "loss_y": 0.0, "loss_trend": 0.0, "loss_season": 0.0, "loss_delta": 0.0}
        count = 0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = compute_base_outputs(trend_model, season_model, batch, trend_state, season_state, net_state)
            loss_y = criterion(outputs["y_hat_scaled"], batch["y_net_scaled"])
            loss_trend = criterion(outputs["trend_hat"], batch["y_trend"])
            loss_season = criterion(outputs["season_base_hat"], batch["y_season"])
            loss = args.lambda_y * loss_y + args.lambda_T * loss_trend + args.lambda_S * loss_season
            loss.backward()
            optimizer.step()
            batch_count = len(batch["x_trend"])
            count += batch_count
            totals["train_loss"] += float(loss.detach().cpu()) * batch_count
            totals["loss_y"] += float(loss_y.detach().cpu()) * batch_count
            totals["loss_trend"] += float(loss_trend.detach().cpu()) * batch_count
            totals["loss_season"] += float(loss_season.detach().cpu()) * batch_count
        logs.append(
            {
                "epoch": epoch,
                "stage": "warmup",
                **{key: value / max(count, 1) for key, value in totals.items()},
            }
        )
    return logs


def train_joint(
    trend_model: nn.Module,
    season_model: nn.Module,
    corrector: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    states: Tuple[TorchScalerState, TorchScalerState, TorchScalerState],
    start_epoch: int = 0,
) -> List[Dict[str, float]]:
    trend_state, season_state, net_state = states
    trend_model.to(device)
    season_model.to(device)
    corrector.to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": trend_model.parameters(), "lr": args.lr_base},
            {"params": season_model.parameters(), "lr": args.lr_base},
            {"params": corrector.parameters(), "lr": args.lr_corr},
        ]
    )
    criterion = nn.L1Loss()
    logs: List[Dict[str, float]] = []

    for epoch in range(1, args.joint_epochs + 1):
        trend_model.train()
        season_model.train()
        corrector.train()
        totals = {"train_loss": 0.0, "loss_y": 0.0, "loss_trend": 0.0, "loss_season": 0.0, "loss_delta": 0.0}
        count = 0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = compute_corrected_outputs(
                trend_model, season_model, corrector, batch, trend_state, season_state, net_state
            )
            loss_y = criterion(outputs["y_hat_scaled"], batch["y_net_scaled"])
            loss_season = criterion(outputs["season_final_hat"], batch["y_season"])
            loss_delta = torch.mean(outputs["delta"] ** 2)
            loss = args.lambda_y * loss_y + args.lambda_S_final * loss_season + args.lambda_delta * loss_delta
            loss.backward()
            optimizer.step()
            batch_count = len(batch["x_trend"])
            count += batch_count
            totals["train_loss"] += float(loss.detach().cpu()) * batch_count
            totals["loss_y"] += float(loss_y.detach().cpu()) * batch_count
            totals["loss_season"] += float(loss_season.detach().cpu()) * batch_count
            totals["loss_delta"] += float(loss_delta.detach().cpu()) * batch_count
        logs.append(
            {
                "epoch": start_epoch + epoch,
                "stage": "joint",
                **{key: value / max(count, 1) for key, value in totals.items()},
            }
        )
    return logs


def collect_predictions(
    trend_model: nn.Module,
    season_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    states: Tuple[TorchScalerState, TorchScalerState, TorchScalerState],
    season_scaler,
    corrector: Optional[nn.Module] = None,
) -> Dict[str, np.ndarray]:
    trend_state, season_state, net_state = states
    trend_model.eval()
    season_model.eval()
    if corrector is not None:
        corrector.eval()
    y_true_net: List[np.ndarray] = []
    y_pred_net: List[np.ndarray] = []
    y_base_net: List[np.ndarray] = []
    season_true: List[np.ndarray] = []
    season_base: List[np.ndarray] = []
    season_final: List[np.ndarray] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if corrector is None:
                outputs = compute_base_outputs(trend_model, season_model, batch, trend_state, season_state, net_state)
                base_outputs = outputs
            else:
                base_outputs = compute_base_outputs(
                    trend_model, season_model, batch, trend_state, season_state, net_state
                )
                outputs = compute_corrected_outputs(
                    trend_model, season_model, corrector, batch, trend_state, season_state, net_state
                )
            y_true_net.append(batch["y_net_raw"].detach().cpu().numpy())
            y_pred_net.append(outputs["y_hat_real"].detach().cpu().numpy())
            y_base_net.append(base_outputs["y_hat_real"].detach().cpu().numpy())
            season_true.append(inverse_transform_numpy(season_scaler, batch["y_season"].detach().cpu().numpy()))
            season_base.append(
                inverse_transform_numpy(season_scaler, base_outputs["season_base_hat"].detach().cpu().numpy())
            )
            season_final.append(
                inverse_transform_numpy(season_scaler, outputs["season_final_hat"].detach().cpu().numpy())
            )

    return {
        "y_true_net": np.concatenate(y_true_net, axis=0),
        "y_pred_net": np.concatenate(y_pred_net, axis=0),
        "y_base_net": np.concatenate(y_base_net, axis=0),
        "season_true": np.concatenate(season_true, axis=0),
        "season_base": np.concatenate(season_base, axis=0),
        "season_final": np.concatenate(season_final, axis=0),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = true - pred
    mae = np.mean(np.abs(err))
    rmse = math.sqrt(np.mean(err**2))
    smape = np.mean(2.0 * np.abs(err) / (np.abs(true) + np.abs(pred) + eps)) * 100.0
    wape = np.sum(np.abs(err)) / (np.sum(np.abs(true)) + eps) * 100.0
    nrmse = rmse / (np.mean(np.abs(true)) + eps) * 100.0
    ss_res = np.sum(err**2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = float("nan") if ss_tot <= eps else 1.0 - ss_res / ss_tot
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "sMAPE": float(smape),
        "WAPE": float(wape),
        "NRMSE_mean_abs": float(nrmse),
        "R2": float(r2),
    }


def evaluate_predictions(
    predictions: Dict[str, np.ndarray],
    mode: str,
    client_id: int,
    client_name: str,
    encoder: str,
    corrector: str,
    q: int,
    train_samples: int,
    test_samples: int,
    args: argparse.Namespace,
    train_final_loss: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    net_metrics = compute_metrics(predictions["y_true_net"], predictions["y_pred_net"], eps=args.eps)
    season_metrics = compute_metrics(predictions["season_true"], predictions["season_final"], eps=args.eps)
    summary = {
        "client_id": client_id,
        "client_name": client_name,
        "season_encoder": encoder,
        "residual_corrector": corrector,
        "q": q,
        "mode": mode,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "warmup_epochs": args.warmup_epochs,
        "joint_epochs": 0 if mode == "base_only" else args.joint_epochs,
        **net_metrics,
        "season_MAE": season_metrics["MAE"],
        "season_RMSE": season_metrics["RMSE"],
        "season_sMAPE": season_metrics["sMAPE"],
        "season_WAPE": season_metrics["WAPE"],
        "train_final_loss": float(train_final_loss),
    }
    step_rows: List[Dict[str, object]] = []
    for step in range(args.horizon):
        step_metrics = compute_metrics(
            predictions["y_true_net"][:, step],
            predictions["y_pred_net"][:, step],
            eps=args.eps,
        )
        step_rows.append(
            {
                "client_id": client_id,
                "season_encoder": encoder,
                "residual_corrector": corrector,
                "q": q,
                "mode": mode,
                "step": f"step_{step + 1}",
                **step_metrics,
            }
        )
    return summary, step_rows


def plot_three_curves(
    y_true: np.ndarray,
    y_base: np.ndarray,
    y_corrected: np.ndarray,
    save_path: Path,
    title: str,
    ylabel: str,
    show_n: int = 200,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(show_n, len(y_true))
    if n <= 0:
        return
    plt.figure(figsize=(12, 4.8))
    plt.plot(y_true[:n, 0], label="true step_1", linewidth=1.7)
    plt.plot(y_base[:n, 0], label="base step_1", linewidth=1.4)
    plt.plot(y_corrected[:n, 0], label="corrected step_1", linewidth=1.4)
    plt.title(title)
    plt.xlabel("test sample index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_prediction_csv(predictions: Dict[str, np.ndarray], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {"sample_index": np.arange(len(predictions["y_true_net"]))}
    horizon = predictions["y_true_net"].shape[1]
    for step in range(horizon):
        step_no = step + 1
        payload[f"true_net_step_{step_no}"] = predictions["y_true_net"][:, step]
        payload[f"base_net_step_{step_no}"] = predictions["y_base_net"][:, step]
        payload[f"pred_net_step_{step_no}"] = predictions["y_pred_net"][:, step]
        payload[f"true_season_step_{step_no}"] = predictions["season_true"][:, step]
        payload[f"base_season_step_{step_no}"] = predictions["season_base"][:, step]
        payload[f"pred_season_step_{step_no}"] = predictions["season_final"][:, step]
    pd.DataFrame(payload).to_csv(save_path, index=False, encoding="utf-8-sig")


def build_pair_compare(summary_df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_df = summary_df[summary_df["mode"] == "base_only"]
    corrected_df = summary_df[summary_df["mode"] == "residual_corrected"]
    for _, corr in corrected_df.iterrows():
        base_match = base_df[
            (base_df["client_id"] == corr["client_id"]) & (base_df["season_encoder"] == corr["season_encoder"])
        ]
        if base_match.empty:
            continue
        base = base_match.iloc[0]
        base_smape = float(base["sMAPE"])
        corr_smape = float(corr["sMAPE"])
        base_wape = float(base["WAPE"])
        corr_wape = float(corr["WAPE"])
        rows.append(
            {
                "client_id": corr["client_id"],
                "season_encoder": corr["season_encoder"],
                "residual_corrector": corr["residual_corrector"],
                "q": corr["q"],
                "base_sMAPE": base_smape,
                "corrected_sMAPE": corr_smape,
                "delta_sMAPE": corr_smape - base_smape,
                "improve_sMAPE_percent": (base_smape - corr_smape) / (base_smape + eps) * 100.0,
                "base_WAPE": base_wape,
                "corrected_WAPE": corr_wape,
                "delta_WAPE": corr_wape - base_wape,
                "improve_WAPE_percent": (base_wape - corr_wape) / (base_wape + eps) * 100.0,
                "is_improved_by_sMAPE": bool(corr_smape < base_smape),
                "is_improved_by_WAPE": bool(corr_wape < base_wape),
            }
        )
    return pd.DataFrame(rows)


def summarize_group(summary_df: pd.DataFrame, group_cols: List[str], corrected_only: bool = False) -> pd.DataFrame:
    df = summary_df.copy()
    if corrected_only:
        df = df[df["mode"] == "residual_corrected"]
    metrics = [
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "NRMSE_mean_abs",
        "R2",
        "season_MAE",
        "season_RMSE",
        "season_sMAPE",
        "season_WAPE",
    ]
    return df.groupby(group_cols)[metrics].mean().reset_index()


def write_overall_verdict(summary_df: pd.DataFrame, pair_df: pd.DataFrame, save_path: Path) -> None:
    lines: List[str] = ["Local season residual correction verdict", ""]
    if summary_df.empty:
        lines.append("No results were produced.")
        save_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("By season encoder and mode:")
    enc = summarize_group(summary_df, ["season_encoder", "mode"])
    for _, row in enc.iterrows():
        lines.append(
            f"  {row['season_encoder']} / {row['mode']}: "
            f"sMAPE={row['sMAPE']:.3f}, WAPE={row['WAPE']:.3f}"
        )
    lines.append("")

    corrected = summary_df[summary_df["mode"] == "residual_corrected"]
    if not corrected.empty:
        lines.append("By residual corrector:")
        for _, row in summarize_group(summary_df, ["residual_corrector"], corrected_only=True).iterrows():
            lines.append(f"  {row['residual_corrector']}: sMAPE={row['sMAPE']:.3f}, WAPE={row['WAPE']:.3f}")
        lines.append("")

        lines.append("By residual length q:")
        for _, row in summarize_group(summary_df, ["q"], corrected_only=True).iterrows():
            lines.append(f"  q={int(row['q'])}: sMAPE={row['sMAPE']:.3f}, WAPE={row['WAPE']:.3f}")
        lines.append("")

        best = corrected.sort_values("sMAPE", ascending=True).iloc[0]
        lines.append(
            "Best residual-corrected combination by sMAPE: "
            f"client_{int(best['client_id'])}, {best['season_encoder']}, "
            f"{best['residual_corrector']}, q={int(best['q'])}, "
            f"sMAPE={best['sMAPE']:.3f}, WAPE={best['WAPE']:.3f}."
        )
    else:
        lines.append("No residual-corrected rows were produced.")
    lines.append("")

    if not pair_df.empty:
        total = len(pair_df)
        smape_count = int(pair_df["is_improved_by_sMAPE"].sum())
        wape_count = int(pair_df["is_improved_by_WAPE"].sum())
        lines.append(
            f"Residual correction improved sMAPE in {smape_count}/{total} paired runs "
            f"({100.0 * smape_count / total:.2f}%)."
        )
        lines.append(
            f"Residual correction improved WAPE in {wape_count}/{total} paired runs "
            f"({100.0 * wape_count / total:.2f}%)."
        )
        lines.append(
            f"Average sMAPE improvement: {pair_df['improve_sMAPE_percent'].mean():.3f}%; "
            f"average WAPE improvement: {pair_df['improve_WAPE_percent'].mean():.3f}%."
        )
    else:
        lines.append("No base/residual pairs were available for improvement comparison.")

    lines.append("")
    lines.append("Conclusion is based only on the generated local experiment results.")
    save_path.write_text("\n".join(lines), encoding="utf-8")


def save_training_log(log_rows: List[Dict[str, float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(save_path, index=False, encoding="utf-8-sig")


def save_config(args: argparse.Namespace, output_dir: Path, data_paths: List[Path]) -> None:
    payload = vars(args).copy()
    payload["device"] = str(args.device)
    payload["data_paths"] = [str(path) for path in data_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_config(args: argparse.Namespace, data_paths: List[Path]) -> None:
    print("\n=== Local Season Residual Correction Config ===")
    print("data_paths:")
    for path in data_paths:
        print(f"  - {path}")
    print(f"time_range: {args.start_time} -> {args.end_time}")
    print(f"seq_len: {args.seq_len}")
    print(f"horizon: {args.horizon}")
    print(f"smooth_window: {args.smooth_window}")
    print(f"season_encoders: {args.season_encoders}")
    print(f"residual_correctors: {args.residual_correctors}")
    print(f"residual_lengths: {args.residual_lengths}")
    print(f"warmup_epochs: {args.warmup_epochs}")
    print(f"joint_epochs: {args.joint_epochs}")
    print(f"device: {args.device}")
    print(f"train_ratio: {args.train_ratio:.4f}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr_base: {args.lr_base}")
    print(f"lr_corr: {args.lr_corr}")
    print(f"scaler: {args.scaler}")
    print(f"output_dir: {args.output_dir}")
    print("================================================\n")


def run_client_encoder(
    client_id: int,
    data_path: Path,
    encoder: str,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    summary_rows: List[Dict[str, object]],
    step_rows: List[Dict[str, object]],
) -> None:
    client_name = f"client_{client_id}"
    df = read_client_series(data_path, args.start_time, args.end_time)
    net_raw = df["net_load"].to_numpy(dtype=np.float64)
    trend_raw, season_raw = decompose_net_load(net_raw, args.smooth_window)
    split_idx = int(math.floor(len(net_raw) * args.train_ratio))
    _, trend_state, trend_scaled = fit_transform_scaler(args.scaler, trend_raw, split_idx)
    season_scaler, season_state, season_scaled = fit_transform_scaler(args.scaler, season_raw, split_idx)
    _, net_state, net_scaled = fit_transform_scaler(args.scaler, net_raw, split_idx)
    states = (trend_state, season_state, net_state)

    print(f"[Client] {client_name}: rows={len(df)}, split_idx={split_idx}, encoder={encoder}")

    base_train_starts = build_starts(len(net_raw), split_idx, args.seq_len, args.horizon, 0, train=True)
    base_test_starts = build_starts(len(net_raw), split_idx, args.seq_len, args.horizon, 0, train=False)
    base_train_ds = LocalResidualDataset(
        base_train_starts,
        trend_scaled,
        season_scaled,
        net_scaled,
        net_raw.astype(np.float32),
        args.seq_len,
        args.horizon,
        residual_length=0,
    )
    base_test_ds = LocalResidualDataset(
        base_test_starts,
        trend_scaled,
        season_scaled,
        net_scaled,
        net_raw.astype(np.float32),
        args.seq_len,
        args.horizon,
        residual_length=0,
    )
    base_train_loader = make_loader(base_train_ds, args.batch_size, device, shuffle=False)
    base_test_loader = make_loader(base_test_ds, args.batch_size, device, shuffle=False)

    set_seed(args.seed + client_id * 1000 + DEFAULT_ENCODERS.index(encoder) * 100)
    trend_model = TrendTCNForecaster(
        horizon=args.horizon,
        hidden_channels=args.trend_hidden_channels,
        kernel_size=3,
        dropout=args.dropout,
    )
    season_model = SeasonBaseModel(
        encoder_name=encoder,
        seq_len=args.seq_len,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    print(f"[Warm-up] {client_name} | encoder={encoder}")
    warm_logs = train_warmup(trend_model, season_model, base_train_loader, args, device, states)
    base_final_loss = warm_logs[-1]["train_loss"] if warm_logs else float("nan")
    base_predictions = collect_predictions(
        trend_model,
        season_model,
        base_test_loader,
        device,
        states,
        season_scaler,
        corrector=None,
    )
    base_summary, base_step_rows = evaluate_predictions(
        base_predictions,
        mode="base_only",
        client_id=client_id,
        client_name=client_name,
        encoder=encoder,
        corrector="none",
        q=0,
        train_samples=len(base_train_ds),
        test_samples=len(base_test_ds),
        args=args,
        train_final_loss=base_final_loss,
    )
    summary_rows.append(base_summary)
    step_rows.extend(base_step_rows)

    base_run_dir = output_dir / "training_logs" / client_name / f"{encoder}_base_only"
    save_training_log(warm_logs, base_run_dir / "training_log.csv")
    save_prediction_csv(base_predictions, output_dir / "predictions" / f"{client_name}_{encoder}_base_only.csv")

    warmed_trend_state = copy.deepcopy(trend_model.state_dict())
    warmed_season_state = copy.deepcopy(season_model.state_dict())

    for corrector_name in args.residual_correctors:
        for q in args.residual_lengths:
            print(f"[Joint] {client_name} | encoder={encoder} | corrector={corrector_name} | q={q}")
            train_starts = build_starts(len(net_raw), split_idx, args.seq_len, args.horizon, q, train=True)
            test_starts = build_starts(len(net_raw), split_idx, args.seq_len, args.horizon, q, train=False)
            train_ds = LocalResidualDataset(
                train_starts,
                trend_scaled,
                season_scaled,
                net_scaled,
                net_raw.astype(np.float32),
                args.seq_len,
                args.horizon,
                residual_length=q,
            )
            test_ds = LocalResidualDataset(
                test_starts,
                trend_scaled,
                season_scaled,
                net_scaled,
                net_raw.astype(np.float32),
                args.seq_len,
                args.horizon,
                residual_length=q,
            )
            train_loader = make_loader(train_ds, args.batch_size, device, shuffle=False)
            test_loader = make_loader(test_ds, args.batch_size, device, shuffle=False)

            trend_joint = TrendTCNForecaster(
                horizon=args.horizon,
                hidden_channels=args.trend_hidden_channels,
                kernel_size=3,
                dropout=args.dropout,
            )
            season_joint = SeasonBaseModel(
                encoder_name=encoder,
                seq_len=args.seq_len,
                horizon=args.horizon,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )
            trend_joint.load_state_dict(warmed_trend_state)
            season_joint.load_state_dict(warmed_season_state)
            corrector = create_corrector(corrector_name, q, args.horizon, args)

            joint_logs = train_joint(
                trend_joint,
                season_joint,
                corrector,
                train_loader,
                args,
                device,
                states,
                start_epoch=args.warmup_epochs,
            )
            all_logs = warm_logs + joint_logs
            final_loss = all_logs[-1]["train_loss"] if all_logs else float("nan")
            predictions = collect_predictions(
                trend_joint,
                season_joint,
                test_loader,
                device,
                states,
                season_scaler,
                corrector=corrector,
            )
            summary, steps = evaluate_predictions(
                predictions,
                mode="residual_corrected",
                client_id=client_id,
                client_name=client_name,
                encoder=encoder,
                corrector=corrector_name,
                q=q,
                train_samples=len(train_ds),
                test_samples=len(test_ds),
                args=args,
                train_final_loss=final_loss,
            )
            summary_rows.append(summary)
            step_rows.extend(steps)

            run_name = f"{encoder}_{corrector_name}_q{q}"
            run_dir = output_dir / "training_logs" / client_name / run_name
            save_training_log(all_logs, run_dir / "training_log.csv")
            save_prediction_csv(predictions, output_dir / "predictions" / f"{client_name}_{run_name}.csv")
            (output_dir / "model_checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "trend_model": trend_joint.state_dict(),
                    "season_model": season_joint.state_dict(),
                    "corrector": corrector.state_dict(),
                    "encoder": encoder,
                    "corrector_name": corrector_name,
                    "q": q,
                    "args": vars(args),
                },
                output_dir / "model_checkpoints" / f"{client_name}_{run_name}.pth",
            )

            if client_id in args.plot_client_ids:
                plot_base = output_dir / "prediction_curves" / client_name
                filename_base = f"{client_name}_{encoder}_{corrector_name}_q{q}"
                title_base = f"{client_name} | {encoder} | {corrector_name} | q={q}"
                plot_three_curves(
                    predictions["y_true_net"],
                    predictions["y_base_net"],
                    predictions["y_pred_net"],
                    plot_base / f"{filename_base}_step1_base_vs_corrected_net.png",
                    title=f"{title_base} | net_load step_1",
                    ylabel="net_load",
                    show_n=args.prediction_plot_samples,
                )
                plot_three_curves(
                    predictions["y_true_net"],
                    predictions["y_base_net"],
                    predictions["y_pred_net"],
                    plot_base / f"{filename_base}_net_true_base_corrected_first200.png",
                    title=f"{title_base} | true/base/corrected net_load",
                    ylabel="net_load",
                    show_n=args.prediction_plot_samples,
                )
                plot_three_curves(
                    predictions["season_true"],
                    predictions["season_base"],
                    predictions["season_final"],
                    plot_base / f"{filename_base}_season_true_base_corrected_first200.png",
                    title=f"{title_base} | true/base/corrected season",
                    ylabel="season",
                    show_n=args.prediction_plot_samples,
                )

            print(
                f"[Done] {client_name} | {encoder} | {corrector_name} | q={q} | "
                f"sMAPE={summary['sMAPE']:.3f}, WAPE={summary['WAPE']:.3f}"
            )


def run_experiment(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.client_ids = [1]
        args.season_encoders = ["Transformer"]
        args.residual_correctors = ["LowRankAdapter"]
        args.residual_lengths = [6]
        args.warmup_epochs = 1
        args.joint_epochs = 1

    args.season_encoders = [normalize_encoder_name(name) for name in args.season_encoders]
    args.residual_correctors = [normalize_corrector_name(name) for name in args.residual_correctors]
    args.device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_paths = [client_csv_path(data_dir, client_id) for client_id in args.client_ids]
    set_seed(args.seed)
    print_config(args, data_paths)
    save_config(args, output_dir, data_paths)

    summary_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []

    for client_id, data_path in zip(args.client_ids, data_paths):
        for encoder in args.season_encoders:
            run_client_encoder(
                client_id=client_id,
                data_path=data_path,
                encoder=encoder,
                args=args,
                output_dir=output_dir,
                device=args.device,
                summary_rows=summary_rows,
                step_rows=step_rows,
            )

    summary_df = pd.DataFrame(summary_rows)
    step_df = pd.DataFrame(step_rows)
    pair_df = build_pair_compare(summary_df, eps=args.eps)
    encoder_df = summarize_group(summary_df, ["season_encoder", "mode"])
    corrector_df = summarize_group(summary_df, ["residual_corrector"], corrected_only=True)
    q_df = summarize_group(summary_df, ["q"], corrected_only=True)

    summary_df.to_csv(output_dir / "local_correction_summary.csv", index=False, encoding="utf-8-sig")
    step_df.to_csv(output_dir / "local_correction_step_metrics.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(output_dir / "local_correction_pair_compare.csv", index=False, encoding="utf-8-sig")
    encoder_df.to_csv(output_dir / "encoder_compare_summary.csv", index=False, encoding="utf-8-sig")
    corrector_df.to_csv(
        output_dir / "residual_corrector_compare_summary.csv", index=False, encoding="utf-8-sig"
    )
    q_df.to_csv(output_dir / "residual_length_compare_summary.csv", index=False, encoding="utf-8-sig")
    write_overall_verdict(summary_df, pair_df, output_dir / "overall_verdict.txt")

    print("\n=== Outputs saved ===")
    print(output_dir / "local_correction_summary.csv")
    print(output_dir / "local_correction_step_metrics.csv")
    print(output_dir / "local_correction_pair_compare.csv")
    print(output_dir / "overall_verdict.txt")
    print(output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local season residual correction experiment.")
    parser.add_argument("--data_dir", type=str, default="per_client_merged_1h")
    parser.add_argument("--start_time", type=str, default="2011-06-01 00:00:00")
    parser.add_argument("--end_time", type=str, default="2013-07-31 23:59:59")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--smooth_window", type=int, default=24)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--season_encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--residual_correctors", nargs="+", default=DEFAULT_CORRECTORS)
    parser.add_argument("--residual_lengths", nargs="+", default=[str(v) for v in DEFAULT_RESIDUAL_LENGTHS])
    parser.add_argument("--warmup_epochs", type=int, default=8)
    parser.add_argument("--joint_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_base", type=float, default=5e-4)
    parser.add_argument("--lr_corr", type=float, default=1e-3)
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax", "none"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="runs/local_season_residual_correction")
    parser.add_argument("--client_ids", nargs="+", default=[str(v) for v in DEFAULT_CLIENT_IDS])
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--trend_hidden_channels", type=int, default=32)
    parser.add_argument("--corrector_hidden_dim", type=int, default=64)
    parser.add_argument("--corrector_gru_layers", type=int, default=1)
    parser.add_argument("--low_rank", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_y", type=float, default=1.0)
    parser.add_argument("--lambda_T", type=float, default=0.5)
    parser.add_argument("--lambda_S", type=float, default=1.0)
    parser.add_argument("--lambda_S_final", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_client_ids", nargs="+", default=["1", "5"])
    parser.add_argument("--prediction_plot_samples", type=int, default=200)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.client_ids = parse_int_list(args.client_ids)
    args.plot_client_ids = parse_int_list(args.plot_client_ids)
    args.season_encoders = parse_str_list(args.season_encoders)
    args.residual_correctors = parse_str_list(args.residual_correctors)
    args.residual_lengths = parse_int_list(args.residual_lengths)
    run_experiment(args)


if __name__ == "__main__":
    main()
