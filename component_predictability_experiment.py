"""Component predictability experiment for hourly client net-load data.

This script is intentionally standalone. It compares whether smoothed
season components are harder to forecast than trend components after an
Autoformer-style moving-average decomposition.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
DEFAULT_WINDOWS = [6, 12, 24, 48]
DEFAULT_MODELS = ["GRU", "DLinear", "TCN"]
DEFAULT_DECOMP_MODES = ["autoformer_same"]
SEQ_LEN = 48
HORIZON = 6
EPS = 1e-6


class ComponentDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class GRUForecaster(nn.Module):
    def __init__(
        self,
        horizon: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class DLinearForecaster(nn.Module):
    def __init__(self, seq_len: int, horizon: int) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.squeeze(-1))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
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
        self.projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return F.relu(out + residual)


class TCNForecaster(nn.Module):
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
            blocks.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.network(x)
        return self.head(x[:, :, -1])


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


def normalize_model_names(values: Sequence[str]) -> List[str]:
    canonical = {"gru": "GRU", "dlinear": "DLinear", "tcn": "TCN"}
    normalized: List[str] = []
    for value in values:
        key = value.lower()
        if key not in canonical:
            raise ValueError(f"Unsupported model '{value}'. Choose from GRU, DLinear, TCN.")
        normalized.append(canonical[key])
    return normalized


def normalize_decomp_modes(values: Sequence[str]) -> List[str]:
    valid = {"autoformer_same", "causal"}
    modes = [value.lower() for value in values]
    invalid = [value for value in modes if value not in valid]
    if invalid:
        raise ValueError(f"Unsupported decomp mode(s): {invalid}. Choose from {sorted(valid)}.")
    return modes


def resolve_device(device_name: str) -> torch.device:
    if device_name.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is unavailable. Falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(device_name)


def get_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "none":
        return None
    raise ValueError("Unsupported scaler. Choose from standard, minmax, none.")


def transform_series(scaler, series: np.ndarray) -> np.ndarray:
    if scaler is None:
        return series.astype(np.float32)
    return scaler.transform(series.reshape(-1, 1)).reshape(-1).astype(np.float32)


def inverse_transform_array(scaler, values: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(values, dtype=np.float64)
    values = np.asarray(values)
    original_shape = values.shape
    restored = scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)
    return restored.astype(np.float64)


def client_csv_path(data_dir: Path, client_id: int) -> Path:
    return data_dir / f"client_{client_id}_load_weather_1h.csv"


def read_client_series(
    data_path: Path,
    start_time: str,
    end_time: str,
    timestamp_col: str = "local_aest_time",
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Client CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    if timestamp_col not in df.columns:
        raise ValueError(f"{data_path} is missing required timestamp column '{timestamp_col}'.")

    if "net_load" in df.columns:
        target = pd.to_numeric(df["net_load"], errors="coerce")
    elif {"gc", "gg"}.issubset(df.columns):
        target = pd.to_numeric(df["gc"], errors="coerce") - pd.to_numeric(df["gg"], errors="coerce")
    else:
        raise ValueError(f"{data_path} must contain net_load or both gc and gg.")

    out = pd.DataFrame(
        {
            timestamp_col: pd.to_datetime(df[timestamp_col], errors="coerce"),
            "net_load": target.astype(float),
        }
    )
    out = out.dropna(subset=[timestamp_col, "net_load"])
    out = out.sort_values(timestamp_col).drop_duplicates(subset=[timestamp_col], keep="first")
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    out = out[(out[timestamp_col] >= start) & (out[timestamp_col] <= end)]
    out = out.reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No usable rows in requested time range for {data_path}.")
    return out


def moving_average_autoformer_same(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("smoothing window must be positive.")
    x = np.asarray(x, dtype=np.float64)
    total_pad = window - 1
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    trend = np.convolve(padded, kernel, mode="valid")
    if len(trend) != len(x):
        raise RuntimeError("autoformer_same decomposition changed sequence length.")
    return trend


def moving_average_causal(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("smoothing window must be positive.")
    x = np.asarray(x, dtype=np.float64)
    cumulative = np.cumsum(np.insert(x, 0, 0.0))
    trend = np.empty_like(x, dtype=np.float64)
    for idx in range(len(x)):
        start = max(0, idx - window + 1)
        trend[idx] = (cumulative[idx + 1] - cumulative[start]) / float(idx - start + 1)
    return trend


def decompose_series(x: np.ndarray, window: int, mode: str) -> Dict[str, np.ndarray]:
    if mode == "autoformer_same":
        trend = moving_average_autoformer_same(x, window)
    elif mode == "causal":
        trend = moving_average_causal(x, window)
    else:
        raise ValueError(f"Unsupported decomp mode: {mode}")
    season = np.asarray(x, dtype=np.float64) - trend
    return {"trend": trend, "season": season}


def build_component_samples(
    component: np.ndarray,
    seq_len: int,
    horizon: int,
    train_ratio: float,
    scaler_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, int]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    component = np.asarray(component, dtype=np.float64)
    n = len(component)
    split_idx = int(math.floor(n * train_ratio))
    min_required = seq_len + horizon + 1
    if split_idx < min_required or n - split_idx < horizon:
        raise ValueError(
            f"Not enough samples after split: n={n}, split_idx={split_idx}, "
            f"seq_len={seq_len}, horizon={horizon}."
        )

    scaler = get_scaler(scaler_name)
    if scaler is not None:
        scaler.fit(component[:split_idx].reshape(-1, 1))
    scaled = transform_series(scaler, component)

    train_starts = np.arange(seq_len, split_idx - horizon + 1)
    test_starts = np.arange(split_idx, n - horizon + 1)

    def make_arrays(starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.stack([scaled[start - seq_len : start] for start in starts], axis=0)
        y = np.stack([scaled[start : start + horizon] for start in starts], axis=0)
        return x[..., None].astype(np.float32), y.astype(np.float32)

    train_x, train_y = make_arrays(train_starts)
    test_x, test_y = make_arrays(test_starts)
    return train_x, train_y, test_x, test_y, scaler, split_idx


def create_model(name: str, args: argparse.Namespace) -> nn.Module:
    if name == "GRU":
        return GRUForecaster(
            horizon=args.horizon,
            hidden_size=args.gru_hidden_size,
            dropout=args.dropout,
            num_layers=2,
        )
    if name == "DLinear":
        return DLinearForecaster(seq_len=args.seq_len, horizon=args.horizon)
    if name == "TCN":
        return TCNForecaster(
            horizon=args.horizon,
            hidden_channels=args.tcn_hidden_channels,
            kernel_size=args.tcn_kernel_size,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model: {name}")


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    dataset = ComponentDataset(train_x, train_y)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    final_loss = float("nan")

    for _ in range(args.epochs):
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            batch_count = len(batch_x)
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count
        final_loss = total_loss / max(total_count, 1)

    return final_loss


def predict_model(
    model: nn.Module,
    test_x: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    dataset = ComponentDataset(test_x, np.zeros((len(test_x), args.horizon), dtype=np.float32))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            pred = model(batch_x).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = true - pred
    mae = np.mean(np.abs(err))
    rmse = math.sqrt(np.mean(err**2))
    smape = np.mean(2.0 * np.abs(err) / (np.abs(true) + np.abs(pred) + eps)) * 100.0
    wape = np.sum(np.abs(err)) / (np.sum(np.abs(true)) + eps) * 100.0
    nrmse_mean_abs = rmse / (np.mean(np.abs(true)) + eps) * 100.0
    ss_res = np.sum(err**2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot <= eps:
        r2 = 1.0 if ss_res <= eps else float("nan")
    else:
        r2 = 1.0 - ss_res / ss_tot
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "sMAPE": float(smape),
        "WAPE": float(wape),
        "NRMSE_mean_abs": float(nrmse_mean_abs),
        "R2": float(r2),
    }


def stable_run_seed(seed: int, client_id: int, window: int, mode_idx: int, model_idx: int) -> int:
    return seed + client_id * 10_000 + window * 100 + mode_idx * 10 + model_idx


def plot_prediction_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str,
    step_index: int = 0,
    show_n: int = 200,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(show_n, len(y_true))
    if n <= 0:
        return
    step_no = step_index + 1
    plt.figure(figsize=(12, 4.8))
    plt.plot(y_true[:n, step_index], label=f"true step_{step_no}", linewidth=1.7)
    plt.plot(y_pred[:n, step_index], label=f"pred step_{step_no}", linewidth=1.4)
    plt.title(title)
    plt.xlabel("test sample index")
    plt.ylabel("component value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_multistep_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str,
    show_n: int = 200,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(show_n, len(y_true))
    if n <= 0:
        return
    horizon = y_true.shape[1]
    ncols = 3
    nrows = int(math.ceil(horizon / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.0 * nrows), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for step in range(horizon):
        ax = axes_arr[step]
        step_no = step + 1
        ax.plot(y_true[:n, step], label=f"true step_{step_no}", linewidth=1.5)
        ax.plot(y_pred[:n, step], label=f"pred step_{step_no}", linewidth=1.25)
        ax.set_title(f"step_{step_no}")
        ax.set_ylabel("component value")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=8)
    for idx in range(horizon, len(axes_arr)):
        axes_arr[idx].axis("off")
    for ax in axes_arr[:horizon]:
        ax.set_xlabel("test sample index")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_prediction_csv(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {"sample_index": np.arange(len(y_true))}
    for step in range(y_true.shape[1]):
        step_no = step + 1
        payload[f"true_step_{step_no}"] = y_true[:, step]
        payload[f"pred_step_{step_no}"] = y_pred[:, step]
    pd.DataFrame(payload).to_csv(save_path, index=False, encoding="utf-8-sig")


def serialize_scaler(scaler) -> Dict[str, object]:
    if scaler is None:
        return {"class": "none"}
    payload: Dict[str, object] = {"class": scaler.__class__.__name__}
    for attr in [
        "mean_",
        "scale_",
        "var_",
        "n_samples_seen_",
        "min_",
        "data_min_",
        "data_max_",
        "data_range_",
        "feature_range",
    ]:
        if not hasattr(scaler, attr):
            continue
        value = getattr(scaler, attr)
        if isinstance(value, np.ndarray):
            payload[attr] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            payload[attr] = value.item()
        elif isinstance(value, tuple):
            payload[attr] = list(value)
        else:
            payload[attr] = value
    return payload


def save_model_checkpoint(
    model: nn.Module,
    scaler,
    save_path: Path,
    args: argparse.Namespace,
    client_id: int,
    model_name: str,
    component_name: str,
    smoothing_window: int,
    decomp_mode: str,
    train_final_loss: float,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "client_id": client_id,
            "component": component_name,
            "smoothing_window": smoothing_window,
            "decomp_mode": decomp_mode,
            "seq_len": args.seq_len,
            "horizon": args.horizon,
            "gru_hidden_size": args.gru_hidden_size,
            "tcn_hidden_channels": args.tcn_hidden_channels,
            "tcn_kernel_size": args.tcn_kernel_size,
            "dropout": args.dropout,
            "scaler_name": args.scaler,
            "scaler_state": serialize_scaler(scaler),
            "train_final_loss": float(train_final_loss),
        },
        save_path,
    )


def plot_metric_boxplot(summary_df: pd.DataFrame, metric: str, save_path: Path) -> None:
    if summary_df.empty:
        return
    models = list(dict.fromkeys(summary_df["model_name"].tolist()))
    components = ["trend", "season"]
    data: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    palette = {"trend": "#4c78a8", "season": "#f58518"}

    for model in models:
        for component in components:
            values = summary_df[
                (summary_df["model_name"] == model) & (summary_df["component"] == component)
            ][metric].dropna()
            if len(values) == 0:
                continue
            data.append(values.to_numpy())
            labels.append(f"{model}\n{component}")
            colors.append(palette[component])

    width = max(8.0, len(labels) * 0.85)
    plt.figure(figsize=(width, 5.4))
    box = plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
    clients = ", ".join(f"client_{cid}" for cid in sorted(summary_df["client_id"].unique()))
    windows = ", ".join(str(w) for w in sorted(summary_df["smoothing_window"].unique()))
    modes = ", ".join(sorted(summary_df["decomp_mode"].unique()))
    plt.title(f"{metric} by component and model\nclients: {clients}; windows: {windows}; modes: {modes}")
    plt.ylabel(metric)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_pair_scatter(pair_df: pd.DataFrame, metric: str, save_path: Path) -> None:
    if pair_df.empty:
        return
    x_col = f"trend_{metric}"
    y_col = f"season_{metric}"
    plt.figure(figsize=(8.2, 7.0))
    marker_map = {"GRU": "o", "DLinear": "s", "TCN": "^"}
    windows = sorted(pair_df["smoothing_window"].unique())
    cmap = plt.get_cmap("tab10")
    window_colors = {window: cmap(idx % 10) for idx, window in enumerate(windows)}

    for (model, window), group in pair_df.groupby(["model_name", "smoothing_window"]):
        plt.scatter(
            group[x_col],
            group[y_col],
            label=f"{model}, window={window}",
            marker=marker_map.get(model, "o"),
            color=window_colors[window],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            s=48,
        )

    all_values = np.concatenate([pair_df[x_col].to_numpy(), pair_df[y_col].to_numpy()])
    finite = all_values[np.isfinite(all_values)]
    if len(finite) > 0:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        pad = (hi - lo) * 0.06 if hi > lo else 1.0
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1.0)
        plt.xlim(lo - pad, hi + pad)
        plt.ylim(lo - pad, hi + pad)

    modes = ", ".join(sorted(pair_df["decomp_mode"].unique()))
    clients = ", ".join(f"client_{cid}" for cid in sorted(pair_df["client_id"].unique()))
    plt.xlabel(f"trend {metric}")
    plt.ylabel(f"season {metric}")
    plt.title(f"Season vs trend {metric}\neach point: client/model/window; clients: {clients}; modes: {modes}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_delta_heatmap(pair_df: pd.DataFrame, save_path: Path) -> None:
    if pair_df.empty:
        return
    plot_df = pair_df.copy()
    plot_df["row_label"] = plot_df.apply(
        lambda row: f"{row['decomp_mode']} | client_{int(row['client_id'])} | {row['model_name']}",
        axis=1,
    )
    heat = plot_df.pivot_table(
        index="row_label",
        columns="smoothing_window",
        values="season_minus_trend_sMAPE",
        aggfunc="mean",
    )
    heat = heat.sort_index()
    fig_height = max(6.0, len(heat.index) * 0.28)
    fig_width = max(7.5, len(heat.columns) * 1.4)
    plt.figure(figsize=(fig_width, fig_height))
    matrix = heat.to_numpy(dtype=float)
    finite = matrix[np.isfinite(matrix)]
    if len(finite) == 0:
        vmin, vmax = -1.0, 1.0
    else:
        max_abs = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1e-6)
        vmin, vmax = -max_abs, max_abs
    im = plt.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="season_sMAPE - trend_sMAPE")
    plt.xticks(np.arange(len(heat.columns)), [str(col) for col in heat.columns])
    plt.yticks(np.arange(len(heat.index)), heat.index, fontsize=8)
    plt.xlabel("smoothing window")
    plt.ylabel("decomp mode | client | model")
    plt.title("Heatmap of season minus trend sMAPE")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                plt.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7, color="black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def build_pair_compare(summary_df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["client_id", "model_name", "smoothing_window", "decomp_mode"]
    for keys, group in summary_df.groupby(group_cols):
        trend = group[group["component"] == "trend"]
        season = group[group["component"] == "season"]
        if trend.empty or season.empty:
            continue
        trend_row = trend.iloc[0]
        season_row = season.iloc[0]
        trend_smape = float(trend_row["sMAPE"])
        season_smape = float(season_row["sMAPE"])
        trend_wape = float(trend_row["WAPE"])
        season_wape = float(season_row["WAPE"])
        client_id, model_name, window, mode = keys
        rows.append(
            {
                "client_id": client_id,
                "model_name": model_name,
                "smoothing_window": window,
                "decomp_mode": mode,
                "trend_sMAPE": trend_smape,
                "season_sMAPE": season_smape,
                "season_minus_trend_sMAPE": season_smape - trend_smape,
                "season_div_trend_sMAPE": season_smape / (trend_smape + eps),
                "trend_WAPE": trend_wape,
                "season_WAPE": season_wape,
                "season_minus_trend_WAPE": season_wape - trend_wape,
                "season_div_trend_WAPE": season_wape / (trend_wape + eps),
                "is_season_harder_by_sMAPE": bool(season_smape > trend_smape),
                "is_season_harder_by_WAPE": bool(season_wape > trend_wape),
            }
        )
    return pd.DataFrame(rows)


def percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def format_group_verdict(pair_df: pd.DataFrame, group_col: str, title: str) -> List[str]:
    lines = [title]
    if pair_df.empty:
        lines.append("  no matched pairs")
        return lines
    for value, group in pair_df.groupby(group_col):
        total = len(group)
        smape_count = int(group["is_season_harder_by_sMAPE"].sum())
        wape_count = int(group["is_season_harder_by_WAPE"].sum())
        lines.append(
            f"  {value}: sMAPE {smape_count}/{total} ({percentage(smape_count, total):.2f}%), "
            f"WAPE {wape_count}/{total} ({percentage(wape_count, total):.2f}%)"
        )
    return lines


def write_overall_verdict(pair_df: pd.DataFrame, save_path: Path) -> None:
    lines: List[str] = ["Component predictability verdict", ""]
    total = len(pair_df)
    if total == 0:
        lines.append("No matched trend/season pairs were produced.")
        save_path.write_text("\n".join(lines), encoding="utf-8")
        return

    smape_count = int(pair_df["is_season_harder_by_sMAPE"].sum())
    wape_count = int(pair_df["is_season_harder_by_WAPE"].sum())
    smape_pct = percentage(smape_count, total)
    wape_pct = percentage(wape_count, total)
    lines.append(
        f"Overall, season sMAPE is higher than trend sMAPE in "
        f"{smape_count}/{total} matched experiments ({smape_pct:.2f}%)."
    )
    lines.append(
        f"Overall, season WAPE is higher than trend WAPE in "
        f"{wape_count}/{total} matched experiments ({wape_pct:.2f}%)."
    )
    if smape_pct > 50.0:
        lines.append(
            "Based on sMAPE, the results support the conclusion that the season component "
            "is generally harder to predict under the current decomposition and model settings."
        )
    elif smape_pct < 50.0:
        lines.append(
            "Based on sMAPE, the results do not support a general conclusion that the season "
            "component is harder to predict under the current settings."
        )
    else:
        lines.append(
            "Based on sMAPE, the paired results are evenly split between season and trend."
        )
    lines.append("")
    lines.extend(format_group_verdict(pair_df, "model_name", "By model:"))
    lines.append("")
    lines.extend(format_group_verdict(pair_df, "smoothing_window", "By smoothing window:"))
    lines.append("")
    lines.extend(format_group_verdict(pair_df, "client_id", "By client:"))
    lines.append("")
    lines.extend(format_group_verdict(pair_df, "decomp_mode", "By decomposition mode:"))
    save_path.write_text("\n".join(lines), encoding="utf-8")


def save_config_snapshot(args: argparse.Namespace, data_paths: Sequence[Path], output_dir: Path) -> None:
    payload = {
        "data_paths": [str(path) for path in data_paths],
        "start_time": args.start_time,
        "end_time": args.end_time,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "smoothing_windows": args.smoothing_windows,
        "models": args.models,
        "device": str(args.device),
        "train_ratio": args.train_ratio,
        "test_ratio": 1.0 - args.train_ratio,
        "decomp_modes": args.decomp_modes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scaler": args.scaler,
        "seed": args.seed,
        "save_models": args.save_models,
        "save_predictions": args.save_predictions,
        "plot_all_steps": args.plot_all_steps,
        "plot_multistep_panel": args.plot_multistep_panel,
        "plot_client_ids": args.plot_client_ids,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_experiment_config(args: argparse.Namespace, data_paths: Sequence[Path]) -> None:
    print("\n=== Component Predictability Experiment Config ===")
    print("data_paths:")
    for path in data_paths:
        print(f"  - {path}")
    print(f"time_range: {args.start_time} -> {args.end_time}")
    print(f"seq_len: {args.seq_len}")
    print(f"horizon: {args.horizon}")
    print(f"smoothing_windows: {args.smoothing_windows}")
    print(f"models: {args.models}")
    print(f"device: {args.device}")
    print(f"train_ratio: {args.train_ratio:.4f}")
    print(f"test_ratio: {1.0 - args.train_ratio:.4f}")
    print(f"decomp_modes: {args.decomp_modes}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"save_models: {args.save_models}")
    print(f"save_predictions: {args.save_predictions}")
    print(f"plot_all_steps: {args.plot_all_steps}")
    print(f"plot_multistep_panel: {args.plot_multistep_panel}")
    print(f"plot_client_ids: {args.plot_client_ids}")
    print(f"output_dir: {args.output_dir}")
    print("=================================================\n")


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "prediction_curves"
    panels_dir = output_dir / "prediction_curves_multistep_panels"
    predictions_dir = output_dir / "predictions"
    models_dir = output_dir / "model_checkpoints"
    data_dir = Path(args.data_dir)
    data_paths = [client_csv_path(data_dir, client_id) for client_id in args.client_ids]
    args.device = resolve_device(args.device)

    set_seed(args.seed)
    print_experiment_config(args, data_paths)
    save_config_snapshot(args, data_paths, output_dir)

    summary_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []

    for client_id, data_path in zip(args.client_ids, data_paths):
        client_name = f"client_{client_id}"
        df = read_client_series(data_path, args.start_time, args.end_time)
        net_load = df["net_load"].to_numpy(dtype=np.float64)
        print(f"[Client] {client_name}: rows={len(df)}, path={data_path}")

        for mode_idx, decomp_mode in enumerate(args.decomp_modes):
            for window in args.smoothing_windows:
                components = decompose_series(net_load, window=window, mode=decomp_mode)
                for model_idx, model_name in enumerate(args.models):
                    run_seed = stable_run_seed(args.seed, client_id, window, mode_idx, model_idx)
                    for component_name in ["trend", "season"]:
                        print(
                            f"[Run] {client_name} | mode={decomp_mode} | window={window} | "
                            f"model={model_name} | component={component_name}"
                        )
                        component = components[component_name]
                        train_x, train_y, test_x, test_y, scaler, split_idx = build_component_samples(
                            component=component,
                            seq_len=args.seq_len,
                            horizon=args.horizon,
                            train_ratio=args.train_ratio,
                            scaler_name=args.scaler,
                        )

                        set_seed(run_seed)
                        model = create_model(model_name, args)
                        final_loss = train_model(model, train_x, train_y, args, args.device)
                        pred_scaled = predict_model(model, test_x, args, args.device)

                        y_true = inverse_transform_array(scaler, test_y)
                        y_pred = inverse_transform_array(scaler, pred_scaled)
                        metrics = compute_metrics(y_true, y_pred, eps=args.eps)
                        plot_base_name = (
                            f"{client_name}_{model_name}_window{window}_{component_name}_{decomp_mode}"
                        )

                        if args.save_predictions:
                            save_prediction_csv(
                                y_true,
                                y_pred,
                                predictions_dir / f"{plot_base_name}_predictions.csv",
                            )
                        if args.save_models:
                            save_model_checkpoint(
                                model,
                                scaler,
                                models_dir / f"{plot_base_name}_model.pth",
                                args,
                                client_id=client_id,
                                model_name=model_name,
                                component_name=component_name,
                                smoothing_window=window,
                                decomp_mode=decomp_mode,
                                train_final_loss=final_loss,
                            )

                        summary_row: Dict[str, object] = {
                            "client_id": client_id,
                            "client_name": client_name,
                            "data_path": str(data_path),
                            "start_time": args.start_time,
                            "end_time": args.end_time,
                            "decomp_mode": decomp_mode,
                            "smoothing_window": window,
                            "model_name": model_name,
                            "component": component_name,
                            "train_samples": len(train_x),
                            "test_samples": len(test_x),
                            **metrics,
                            "target_abs_mean": float(np.mean(np.abs(y_true.reshape(-1)))),
                            "target_std": float(np.std(y_true.reshape(-1))),
                            "epochs": args.epochs,
                            "train_final_loss": float(final_loss),
                            "split_index": split_idx,
                            "scaler": args.scaler,
                        }
                        summary_rows.append(summary_row)

                        for step in range(args.horizon):
                            step_metrics = compute_metrics(y_true[:, step], y_pred[:, step], eps=args.eps)
                            step_rows.append(
                                {
                                    "client_id": client_id,
                                    "model_name": model_name,
                                    "component": component_name,
                                    "smoothing_window": window,
                                    "decomp_mode": decomp_mode,
                                    "step": f"step_{step + 1}",
                                    **step_metrics,
                                }
                            )

                        if client_id in args.plot_client_ids:
                            steps_to_plot = range(args.horizon) if args.plot_all_steps else range(1)
                            for step_index in steps_to_plot:
                                step_no = step_index + 1
                                plot_prediction_curve(
                                    y_true,
                                    y_pred,
                                    curves_dir / f"{plot_base_name}_step{step_no}_true_vs_pred.png",
                                    title=(
                                        f"{client_name} | {model_name} | window={window} | "
                                        f"{component_name} | {decomp_mode} | step_{step_no}"
                                    ),
                                    step_index=step_index,
                                    show_n=args.prediction_plot_samples,
                                )
                            if args.plot_multistep_panel:
                                plot_multistep_panel(
                                    y_true,
                                    y_pred,
                                    panels_dir
                                    / f"{plot_base_name}_steps1_to_{args.horizon}_true_vs_pred_panel.png",
                                    title=(
                                        f"{client_name} | {model_name} | window={window} | "
                                        f"{component_name} | {decomp_mode}"
                                    ),
                                    show_n=args.prediction_plot_samples,
                                )

                        print(
                            f"[Done] train_samples={len(train_x)}, test_samples={len(test_x)}, "
                            f"final_loss={final_loss:.6f}, sMAPE={metrics['sMAPE']:.3f}, "
                            f"WAPE={metrics['WAPE']:.3f}"
                        )

    summary_df = pd.DataFrame(summary_rows)
    step_df = pd.DataFrame(step_rows)
    pair_df = build_pair_compare(summary_df, eps=args.eps)

    summary_path = output_dir / "component_predictability_summary.csv"
    step_path = output_dir / "component_predictability_step_metrics.csv"
    pair_path = output_dir / "component_predictability_pair_compare.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    step_df.to_csv(step_path, index=False, encoding="utf-8-sig")
    pair_df.to_csv(pair_path, index=False, encoding="utf-8-sig")

    write_overall_verdict(pair_df, output_dir / "overall_verdict.txt")
    plot_metric_boxplot(summary_df, "sMAPE", output_dir / "summary_boxplot_sMAPE_by_component.png")
    plot_metric_boxplot(summary_df, "WAPE", output_dir / "summary_boxplot_WAPE_by_component.png")
    plot_pair_scatter(pair_df, "sMAPE", output_dir / "season_vs_trend_sMAPE_scatter.png")
    plot_pair_scatter(pair_df, "WAPE", output_dir / "season_vs_trend_WAPE_scatter.png")
    plot_delta_heatmap(pair_df, output_dir / "heatmap_season_minus_trend_sMAPE.png")

    print("\n=== Outputs saved ===")
    print(summary_path)
    print(step_path)
    print(pair_path)
    print(output_dir / "overall_verdict.txt")
    print(output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether season components are harder to predict than trend components "
            "after moving-average decomposition of client net_load."
        )
    )
    parser.add_argument("--data_dir", type=str, default="per_client_merged_1h")
    parser.add_argument("--output_dir", type=str, default="runs/component_predictability")
    parser.add_argument("--start_time", type=str, default="2011-06-01 00:00:00")
    parser.add_argument("--end_time", type=str, default="2013-07-31 23:59:59")
    parser.add_argument("--client_ids", nargs="+", default=[str(v) for v in DEFAULT_CLIENT_IDS])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--smoothing_windows", nargs="+", default=[str(v) for v in DEFAULT_WINDOWS])
    parser.add_argument("--decomp_modes", nargs="+", default=DEFAULT_DECOMP_MODES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax", "none"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gru_hidden_size", type=int, default=64)
    parser.add_argument("--tcn_hidden_channels", type=int, default=32)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--plot_client_ids", nargs="+", default=["1", "5"])
    parser.add_argument("--prediction_plot_samples", type=int, default=200)
    parser.add_argument(
        "--no_save_models",
        dest="save_models",
        action="store_false",
        help="Do not save per-run model checkpoints.",
    )
    parser.add_argument(
        "--no_save_predictions",
        dest="save_predictions",
        action="store_false",
        help="Do not save per-sample true/pred CSV files.",
    )
    parser.add_argument(
        "--no_plot_all_steps",
        dest="plot_all_steps",
        action="store_false",
        help="Only plot step_1 curves instead of step_1 through step_horizon.",
    )
    parser.add_argument(
        "--no_plot_multistep_panel",
        dest="plot_multistep_panel",
        action="store_false",
        help="Do not save the 2x3 multi-step true-vs-pred panel.",
    )
    parser.set_defaults(
        save_models=True,
        save_predictions=True,
        plot_all_steps=True,
        plot_multistep_panel=True,
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.client_ids = parse_int_list(args.client_ids)
    args.plot_client_ids = parse_int_list(args.plot_client_ids)
    args.smoothing_windows = parse_int_list(args.smoothing_windows)
    args.models = normalize_model_names(parse_str_list(args.models))
    args.decomp_modes = normalize_decomp_modes(parse_str_list(args.decomp_modes))
    run_experiment(args)


if __name__ == "__main__":
    main()
