import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    MinMaxScaler = None


class SimpleMinMaxScaler:
    """sklearn 不可用时的轻量 MinMaxScaler fallback。"""

    def fit(self, x):
        arr = np.asarray(x, dtype=np.float64)
        self.data_min_ = np.min(arr, axis=0)
        self.data_max_ = np.max(arr, axis=0)
        data_range = self.data_max_ - self.data_min_
        self.safe_range_ = np.where(data_range == 0.0, 1.0, data_range)
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=np.float64)
        return (arr - self.data_min_) / self.safe_range_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        arr = np.asarray(x, dtype=np.float64)
        return arr * self.safe_range_ + self.data_min_


@dataclass
class DataConfig:
    data_path: str = "per_client_merged/client_2_load_weather_30min.csv"
    save_dir: str = "runs/iter_dual_expert_client2-动态自反馈-raw-每轮更新"
    datetime_col: str = "timestamp"
    gc_col: str = "gc"
    gg_col: str = "gg"
    net_load_col: str = "net_load"
    ghi_col: str = "ghi_wm2"
    temp_c_col: str = "temp2m_c"
    temp_k_col: str = "temp2m_k"
    wind_col: str = "wind10m_ms"
    seq_len: int = 48
    horizon: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    dropna: bool = True
    sort_by_time: bool = True


@dataclass
class FeatureConfig:
    use_ghi: bool = True
    use_temp: bool = True
    use_wind: bool = True
    use_slot_sin_cos: bool = True
    use_weekday_sin_cos: bool = True
    use_month_sin_cos: bool = True
    use_is_workday: bool = True
    recent_short_points: int = 4
    recent_long_points: int = 12
    ghi_day_threshold: float = 10.0
    ghi_delta_threshold: float = 50.0


@dataclass
class BaseConfig:
    period_hidden: int = 32
    trend_hidden: int = 48
    trend_num_layers: int = 1
    fusion_hidden: int = 32
    dropout: float = 0.0
    target_mode: str = "raw"
    smooth_window: int = 6
    smoothness_lambda: float = 0.0
    freeze_after_train: bool = True


@dataclass
class PVExpertConfig:
    backbone: str = "cnn_lstm_attention"
    seq_len_plus_future: bool = True
    use_future_weather: bool = True
    use_future_time: bool = True
    conv1_channels: int = 32
    conv2_channels: int = 64
    conv1_kernel: int = 3
    conv2_kernel: int = 3
    pool1_kernel: int = 2
    pool2_kernel: int = 3
    lstm_hidden1: int = 48
    lstm_hidden2: int = 24
    attn_units: int = 24
    fc_hidden: int = 24
    dropout: float = 0.0
    weibull_hidden: int = 48
    weibull_layers: int = 1
    weibull_eps: float = 1e-8
    output_tanh: bool = True
    max_correction: float = 0.7


@dataclass
class LoadExpertConfig:
    hidden1: int = 64
    hidden2: int = 64
    hidden3: int = 32
    dropout: float = 0.1
    use_residual_mlp: bool = True
    output_tanh: bool = True
    max_correction: float = 1.0


@dataclass
class GateConfig:
    hidden: int = 32
    dropout: float = 0.0
    use_prior_bias: bool = True
    prior_beta: float = 1
    prior_eps: float = 1e-6
    night_pv_prior: float = 0.05
    ghi_change_pv_prior: float = 0.75
    evening_mix_pv_prior: float = 0.45
    default_pv_prior: float = 0.5


@dataclass
class FeedbackConfig:
    use_self_feedback: bool = True
    use_feedback_controller: bool = True
    dynamic_stop: bool = True
    feedback_steps: int = 2
    safe_max_feedback_steps: int = 15
    convergence_eps: float = 0.001
    gamma: float = 0.3
    stopgrad_feedback: bool = True
    feedback_hidden: int = 32
    feedback_dropout: float = 0.0


@dataclass
class LossConfig:
    mse_loss_weight: float = 1.0
    night_loss_weight: float = 0.1
    conv_loss_weight: float = 0.02


@dataclass
class TrainConfig:
    batch_size: int = 256
    base_epochs: int = 40
    expert_epochs: int = 40
    lr: float = 1e-3
    expert_lr: Optional[float] = 5e-4
    weight_decay: float = 0.0
    patience: int = 8
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    scaler_x: str = "minmax"
    scaler_y: str = "minmax"


@dataclass
class FullConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    base: BaseConfig = field(default_factory=BaseConfig)
    pv_expert: PVExpertConfig = field(default_factory=PVExpertConfig)
    load_expert: LoadExpertConfig = field(default_factory=LoadExpertConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


TIME_BASE_COLS = [
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_workday",
]
PV_TIME_COLS = ["slot_sin", "slot_cos", "month_sin", "month_cos"]
LOAD_TIME_COLS = [
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_workday",
]
GATE_TIME_COLS = ["slot_sin", "slot_cos", "month_sin", "month_cos", "is_workday"]


@dataclass
class PreparedSplit:
    name: str
    df: pd.DataFrame
    feature_scaled: pd.DataFrame
    y_scaled: np.ndarray
    y_target_scaled: np.ndarray


@dataclass
class PreparedData:
    train: PreparedSplit
    val: PreparedSplit
    test: PreparedSplit
    feature_cols: List[str]
    time_cols: List[str]
    x_scaler: object
    y_scaler: object


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scaler(kind: str):
    if kind.lower() != "minmax":
        raise ValueError(f"Only minmax scaler is supported in this self-contained script, got {kind!r}.")
    if MinMaxScaler is not None:
        return MinMaxScaler()
    return SimpleMinMaxScaler()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def inverse_transform_1d(scaler, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    mape = float(np.mean(np.abs(err / (np.abs(y_true) + eps))) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if ss_tot <= eps else float(1.0 - ss_res / ss_tot)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE_percent": mape, "R2": r2}


def add_time_features(df: pd.DataFrame, cfg: FullConfig) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[cfg.data.datetime_col])
    out[cfg.data.datetime_col] = dt

    out["slot"] = (dt.dt.hour * 2 + (dt.dt.minute // 30)).astype(int)
    out["slot_sin"] = np.sin(2.0 * np.pi * out["slot"] / 48.0)
    out["slot_cos"] = np.cos(2.0 * np.pi * out["slot"] / 48.0)

    weekday = dt.dt.weekday.astype(int)
    out["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    out["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
    out["is_workday"] = (weekday < 5).astype(float)

    month_zero_based = dt.dt.month.astype(int) - 1
    out["month_sin"] = np.sin(2.0 * np.pi * month_zero_based / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month_zero_based / 12.0)
    return out


def selected_time_cols(cfg: FullConfig) -> List[str]:
    cols: List[str] = []
    if cfg.features.use_slot_sin_cos:
        cols += ["slot_sin", "slot_cos"]
    if cfg.features.use_weekday_sin_cos:
        cols += ["weekday_sin", "weekday_cos"]
    if cfg.features.use_month_sin_cos:
        cols += ["month_sin", "month_cos"]
    if cfg.features.use_is_workday:
        cols += ["is_workday"]
    return cols


def selected_weather_feature_cols(cfg: FullConfig) -> List[str]:
    cols: List[str] = []
    if cfg.features.use_ghi:
        cols.append(cfg.data.ghi_col)
    if cfg.features.use_temp:
        cols.append(cfg.data.temp_c_col)
    if cfg.features.use_wind:
        cols.append(cfg.data.wind_col)
    return cols


def _legacy_selected_feature_cols_disabled(cfg: FullConfig, time_cols: List[str]) -> List[str]:
    # GHI/temp/wind 即使未进入 PV 序列，也会被 gate/load 分支或物理先验使用。
    cols: List[str] = [cfg.data.ghi_col, cfg.data.temp_c_col, cfg.data.wind_col]
    cols += time_cols
    return list(dict.fromkeys(cols))


def selected_feature_cols(cfg: FullConfig, time_cols: List[str]) -> List[str]:
    cols = selected_weather_feature_cols(cfg)
    cols += time_cols
    return list(dict.fromkeys(cols))


def selected_load_time_cols(cfg: FullConfig) -> List[str]:
    cols: List[str] = []
    if cfg.features.use_slot_sin_cos:
        cols += ["slot_sin", "slot_cos"]
    if cfg.features.use_weekday_sin_cos:
        cols += ["weekday_sin", "weekday_cos"]
    if cfg.features.use_month_sin_cos:
        cols += ["month_sin", "month_cos"]
    if cfg.features.use_is_workday:
        cols += ["is_workday"]
    return cols


def selected_gate_time_cols(cfg: FullConfig) -> List[str]:
    cols: List[str] = []
    if cfg.features.use_slot_sin_cos:
        cols += ["slot_sin", "slot_cos"]
    if cfg.features.use_month_sin_cos:
        cols += ["month_sin", "month_cos"]
    if cfg.features.use_is_workday:
        cols += ["is_workday"]
    return cols


def smooth_target(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return y.copy()
    return (
        pd.Series(np.asarray(y, dtype=np.float32))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )


def load_and_prepare_data(cfg: FullConfig) -> PreparedData:
    df = pd.read_csv(cfg.data.data_path)
    required = [cfg.data.datetime_col, cfg.data.gc_col, cfg.data.gg_col]
    if cfg.features.use_ghi:
        required.append(cfg.data.ghi_col)
    if cfg.features.use_wind:
        required.append(cfg.data.wind_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if cfg.features.use_temp and cfg.data.temp_c_col not in df.columns:
        if cfg.data.temp_k_col not in df.columns:
            raise ValueError(f"CSV must contain either {cfg.data.temp_c_col!r} or {cfg.data.temp_k_col!r}.")
        df[cfg.data.temp_c_col] = df[cfg.data.temp_k_col].astype(float) - 273.15

    df = add_time_features(df, cfg)
    df[cfg.data.net_load_col] = df[cfg.data.gc_col].astype(float) - df[cfg.data.gg_col].astype(float)
    if cfg.data.sort_by_time:
        df = df.sort_values(cfg.data.datetime_col)

    time_cols = selected_time_cols(cfg)
    weather_cols = selected_weather_feature_cols(cfg)
    feature_cols = selected_feature_cols(cfg, time_cols)
    needed = [cfg.data.datetime_col, cfg.data.net_load_col]
    if cfg.features.use_ghi:
        needed.append(cfg.data.ghi_col)
    if cfg.features.use_temp:
        needed.append(cfg.data.temp_c_col)
    if cfg.features.use_wind:
        needed.append(cfg.data.wind_col)
    needed += list(dict.fromkeys(time_cols + ["slot"]))
    if cfg.data.dropna:
        df = df.dropna(subset=list(dict.fromkeys(needed + feature_cols))).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if len(df) < cfg.data.seq_len * 4:
        raise ValueError(
            f"Not enough rows after preprocessing: {len(df)}. "
            f"Need substantially more than 4 * seq_len={4 * cfg.data.seq_len}."
        )

    n = len(df)
    train_end = int(n * cfg.data.train_ratio)
    val_end = train_end + int(n * cfg.data.val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid train/val/test split sizes. Please adjust train_ratio and val_ratio.")

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    x_scaler = make_scaler(cfg.train.scaler_x) if weather_cols else None
    y_scaler = make_scaler(cfg.train.scaler_y)
    if weather_cols:
        x_scaler.fit(train_df[weather_cols].to_numpy(dtype=np.float64))
    y_scaler.fit(train_df[[cfg.data.net_load_col]].to_numpy(dtype=np.float64))

    def make_split(name: str, split_df: pd.DataFrame) -> PreparedSplit:
        feature_scaled = pd.DataFrame(index=split_df.index)
        if weather_cols:
            x_scaled = x_scaler.transform(split_df[weather_cols].to_numpy(dtype=np.float64))
            for i, col in enumerate(weather_cols):
                feature_scaled[col] = x_scaled[:, i]
        for col in time_cols:
            feature_scaled[col] = split_df[col].to_numpy(dtype=np.float32)
        feature_scaled = feature_scaled[feature_cols]
        y_scaled = y_scaler.transform(split_df[[cfg.data.net_load_col]].to_numpy(dtype=np.float64)).reshape(-1)
        y_scaled = y_scaled.astype(np.float32)
        if cfg.base.target_mode == "smooth":
            y_target = smooth_target(y_scaled, cfg.base.smooth_window)
        elif cfg.base.target_mode == "raw":
            y_target = y_scaled.copy()
        else:
            raise ValueError("BaseConfig.target_mode must be 'raw' or 'smooth'.")
        return PreparedSplit(name, split_df, feature_scaled.astype(np.float32), y_scaled, y_target.astype(np.float32))

    return PreparedData(
        train=make_split("train", train_df),
        val=make_split("val", val_df),
        test=make_split("test", test_df),
        feature_cols=feature_cols,
        time_cols=time_cols,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


class BaseWindowDataset(Dataset):
    def __init__(self, split: PreparedSplit, time_cols: List[str], cfg: FullConfig):
        self.split = split
        self.time_cols = time_cols
        self.seq_len = int(cfg.data.seq_len)
        self.horizon = int(cfg.data.horizon)
        if self.horizon != 1:
            raise ValueError("This script implements one-step horizon=1 forecasting.")
        self.time_values = split.feature_scaled[time_cols].to_numpy(dtype=np.float32)
        self.y_scaled = np.asarray(split.y_scaled, dtype=np.float32)
        self.y_target_scaled = np.asarray(split.y_target_scaled, dtype=np.float32)

    def __len__(self):
        return max(0, len(self.y_scaled) - self.seq_len)

    def __getitem__(self, idx):
        j = idx + self.seq_len
        net_seq = self.y_scaled[idx:j].reshape(self.seq_len, 1)
        time_seq = self.time_values[idx:j]
        time_future = self.time_values[j]
        target = np.array([self.y_target_scaled[j]], dtype=np.float32)
        return {
            "net_seq": torch.from_numpy(net_seq.copy()),
            "time_seq": torch.from_numpy(time_seq.copy()),
            "time_future": torch.from_numpy(time_future.copy()),
            "target": torch.from_numpy(target),
            "target_index": torch.tensor(j, dtype=torch.long),
        }


class TrendPeriodBaseModel(nn.Module):
    def __init__(self, time_dim: int, cfg: FullConfig):
        super().__init__()
        self.cfg = cfg
        self.recent_short = max(1, int(cfg.features.recent_short_points))
        self.recent_long = max(1, int(cfg.features.recent_long_points))

        self.period_mlp = nn.Sequential(
            nn.Linear(time_dim, cfg.base.period_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.base.dropout),
            nn.Linear(cfg.base.period_hidden, 1),
        )
        self.trend_gru = nn.GRU(
            input_size=1,
            hidden_size=cfg.base.trend_hidden,
            num_layers=cfg.base.trend_num_layers,
            dropout=cfg.base.dropout if cfg.base.trend_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.trend_head = nn.Sequential(
            nn.Linear(cfg.base.trend_hidden, cfg.base.trend_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.base.dropout),
            nn.Linear(cfg.base.trend_hidden, 1),
        )
        state_dim = 5
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 + state_dim, cfg.base.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.base.dropout),
            nn.Linear(cfg.base.fusion_hidden, 2),
        )

    def _state_features(self, net_seq: torch.Tensor) -> torch.Tensor:
        x = net_seq.squeeze(-1)
        n_t = x[:, -1]
        short = min(self.recent_short, x.shape[1])
        long = min(self.recent_long, x.shape[1])
        mean_short = x[:, -short:].mean(dim=1)
        long_window = x[:, -long:]
        mean_long = long_window.mean(dim=1)
        std_long = long_window.std(dim=1, unbiased=False)
        lag = min(self.recent_short, x.shape[1] - 1)
        diff = n_t - x[:, -1 - lag]
        return torch.stack([n_t, mean_short, mean_long, std_long, diff], dim=1)

    def forward(self, net_seq: torch.Tensor, time_seq: torch.Tensor, time_future: torch.Tensor, return_components=False):
        period_pred = self.period_mlp(time_future)
        _, h_n = self.trend_gru(net_seq)
        trend_state = h_n[-1]
        trend_pred = self.trend_head(trend_state)
        state = self._state_features(net_seq)
        fusion_input = torch.cat([period_pred, trend_pred, state], dim=1)
        weights = torch.softmax(self.fusion_mlp(fusion_input), dim=1)
        pred = weights[:, 0:1] * period_pred + weights[:, 1:2] * trend_pred
        if return_components:
            return {
                "pred": pred,
                "period_pred": period_pred,
                "trend_pred": trend_pred,
                "period_weight": weights[:, 0:1],
                "trend_weight": weights[:, 1:2],
            }
        return pred


class SamePadMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

    def forward(self, x):
        total_pad = self.kernel_size - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = F.pad(x, (pad_left, pad_right), mode="constant", value=float("-inf"))
        return F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)


class FallbackAttention(nn.Module):
    def __init__(self, input_dim: int, attn_units: int):
        super().__init__()
        self.score_vec = nn.Linear(input_dim, input_dim, bias=False)
        self.attn_out = nn.Linear(input_dim * 2, attn_units, bias=False)

    def forward(self, x):
        score_first = self.score_vec(x)
        h_t = x[:, -1, :]
        score = torch.bmm(score_first, h_t.unsqueeze(2)).squeeze(2)
        weights = torch.softmax(score, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return torch.tanh(self.attn_out(torch.cat([context, h_t], dim=1)))


class FallbackCNNLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cfg.conv1_channels, cfg.conv1_kernel, padding=cfg.conv1_kernel // 2)
        self.pool1 = SamePadMaxPool1d(cfg.pool1_kernel)
        self.conv2 = nn.Conv1d(cfg.conv1_channels, cfg.conv2_channels, cfg.conv2_kernel, padding=cfg.conv2_kernel // 2)
        self.pool2 = SamePadMaxPool1d(cfg.pool2_kernel)
        self.dropout = nn.Dropout(cfg.dropout)
        self.lstm1 = nn.LSTM(cfg.conv2_channels, cfg.lstm_hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(cfg.lstm_hidden1, cfg.lstm_hidden2, batch_first=True)
        self.attention = FallbackAttention(cfg.lstm_hidden2, cfg.attn_units)
        self.fc1 = nn.Linear(cfg.attn_units, cfg.fc_hidden)
        self.fc2 = nn.Linear(cfg.fc_hidden, output_dim)

    def forward(self, x):
        z = x.permute(0, 2, 1)
        z = self.dropout(self.pool1(F.relu(self.conv1(z))))
        z = self.dropout(self.pool2(F.relu(self.conv2(z))))
        z = z.permute(0, 2, 1)
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        z = self.attention(z)
        return self.fc2(F.relu(self.fc1(z)))


class MultiFeatureWeibullAttentionLSTM(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden: int, layers: int, eps: float):
        super().__init__()
        self.seq_len = int(seq_len)
        self.eps = float(eps)
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True)
        self.refine = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("tau", torch.arange(1, self.seq_len + 1, dtype=torch.float32))
        init_kappa = self._inverse_softplus(1.5)
        init_lambda = self._inverse_softplus(max(1.0, self.seq_len / 2.0))
        self.raw_kappa = nn.Parameter(torch.full((self.seq_len,), float(init_kappa)))
        self.raw_lambda = nn.Parameter(torch.full((self.seq_len,), float(init_lambda)))

    @staticmethod
    def _inverse_softplus(value: float) -> torch.Tensor:
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        refined = self.refine(h_seq)
        tau = self.tau.to(device=x.device, dtype=refined.dtype)
        kappa = F.softplus(self.raw_kappa).to(device=x.device, dtype=refined.dtype) + self.eps
        lambda_ = F.softplus(self.raw_lambda).to(device=x.device, dtype=refined.dtype) + self.eps
        scaled_tau = torch.clamp(tau / lambda_, min=self.eps)
        alpha = (kappa / lambda_) * torch.pow(scaled_tau, kappa - 1.0) * torch.exp(-torch.pow(scaled_tau, kappa))
        alpha = alpha / (alpha.sum() + self.eps)
        context = torch.sum(refined * alpha.view(1, self.seq_len, 1), dim=1)
        last = refined[:, -1, :]
        return self.head(torch.cat([context, last], dim=1))


class PVExpert(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, cfg: FullConfig):
        super().__init__()
        self.cfg = cfg
        pv_cfg = cfg.pv_expert
        if pv_cfg.backbone == "cnn_lstm_attention":
            model_cfg = SimpleNamespace(
                use_attention=True,
                conv1_channels=pv_cfg.conv1_channels,
                conv2_channels=pv_cfg.conv2_channels,
                conv1_kernel=pv_cfg.conv1_kernel,
                conv2_kernel=pv_cfg.conv2_kernel,
                pool1_kernel=pv_cfg.pool1_kernel,
                pool2_kernel=pv_cfg.pool2_kernel,
                lstm_hidden1=pv_cfg.lstm_hidden1,
                lstm_hidden2=pv_cfg.lstm_hidden2,
                attn_units=pv_cfg.attn_units,
                fc_hidden=pv_cfg.fc_hidden,
                dropout=pv_cfg.dropout,
            )
            try:
                from models.cnn_lstm import CNNLSTMModel

                self.backbone = CNNLSTMModel(input_dim=input_dim, output_dim=1, cfg=model_cfg)
            except Exception:
                self.backbone = FallbackCNNLSTMAttention(input_dim=input_dim, output_dim=1, cfg=model_cfg)
        elif pv_cfg.backbone == "weibull_lstm":
            self.backbone = MultiFeatureWeibullAttentionLSTM(
                input_dim=input_dim,
                seq_len=seq_len,
                hidden=pv_cfg.weibull_hidden,
                layers=pv_cfg.weibull_layers,
                eps=pv_cfg.weibull_eps,
            )
        else:
            raise ValueError("PV backbone must be 'cnn_lstm_attention' or 'weibull_lstm'.")

    def forward(self, x):
        out = self.backbone(x)
        if self.cfg.pv_expert.output_tanh:
            out = self.cfg.pv_expert.max_correction * torch.tanh(out)
        return out


class LoadExpertResidualMLP(nn.Module):
    def __init__(self, input_dim: int, cfg: FullConfig):
        super().__init__()
        lc = cfg.load_expert
        self.cfg = cfg
        self.fc1 = nn.Linear(input_dim, lc.hidden1)
        self.fc2 = nn.Linear(lc.hidden1, lc.hidden2)
        self.proj = nn.Identity() if lc.hidden1 == lc.hidden2 else nn.Linear(lc.hidden1, lc.hidden2)
        self.dropout = nn.Dropout(lc.dropout)
        self.fc3 = nn.Linear(lc.hidden2, lc.hidden3)
        self.out = nn.Linear(lc.hidden3, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = self.dropout(F.relu(self.fc2(h1)))
        if self.cfg.load_expert.use_residual_mlp:
            h = self.proj(h1) + h2
        else:
            h = h2
        h = F.relu(self.fc3(h))
        out = self.out(h)
        if self.cfg.load_expert.output_tanh:
            out = self.cfg.load_expert.max_correction * torch.tanh(out)
        return out


class PhysicsPriorGate(nn.Module):
    def __init__(self, input_dim: int, cfg: FullConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.gate.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.gate.dropout),
            nn.Linear(cfg.gate.hidden, 2),
        )

    def _prior(self, future_ghi_raw: torch.Tensor, delta_ghi_raw: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        gc = self.cfg.gate
        fc = self.cfg.features
        pv_prior = torch.full_like(future_ghi_raw, float(gc.default_pv_prior))

        change_mask = torch.abs(delta_ghi_raw) > fc.ghi_delta_threshold
        pv_prior = torch.where(change_mask, torch.full_like(pv_prior, gc.ghi_change_pv_prior), pv_prior)

        evening_mask = (slot >= 32.0) & (slot < 40.0) & (delta_ghi_raw < 0.0)
        pv_prior = torch.where(evening_mask, torch.full_like(pv_prior, gc.evening_mix_pv_prior), pv_prior)

        night_mask = future_ghi_raw < fc.ghi_day_threshold
        pv_prior = torch.where(night_mask, torch.full_like(pv_prior, gc.night_pv_prior), pv_prior)

        load_prior = 1.0 - pv_prior
        return torch.cat([pv_prior, load_prior], dim=1).clamp(min=gc.prior_eps, max=1.0)

    def forward(
        self,
        x: torch.Tensor,
        future_ghi_raw: torch.Tensor,
        delta_ghi_raw: torch.Tensor,
        hour_or_slot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        prior = self._prior(future_ghi_raw, delta_ghi_raw, hour_or_slot)
        if self.cfg.gate.use_prior_bias:
            logits = logits + self.cfg.gate.prior_beta * torch.log(prior + self.cfg.gate.prior_eps)
        weights = torch.softmax(logits, dim=1)
        return weights, prior


class FeedbackController(nn.Module):
    def __init__(self, input_dim: int, cfg: FullConfig):
        super().__init__()
        hidden = int(cfg.feedback.feedback_hidden)
        dropout = float(cfg.feedback.feedback_dropout)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IterativeDualExpertNetLoadModel(nn.Module):
    def __init__(
        self,
        base_model: TrendPeriodBaseModel,
        pv_input_dim: int,
        pv_seq_len: int,
        load_input_dim: int,
        gate_input_dim: int,
        cfg: FullConfig,
        feedback_input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.cfg = cfg
        self.pv_expert = PVExpert(pv_input_dim, pv_seq_len, cfg)
        self.load_expert = LoadExpertResidualMLP(load_input_dim, cfg)
        self.gate = PhysicsPriorGate(gate_input_dim, cfg)
        if feedback_input_dim is None:
            feedback_input_dim = gate_input_dim + 4
        self.feedback_controller = FeedbackController(feedback_input_dim, cfg)

    def _forward_legacy_unused(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pv_hist = batch["pv_hist_seq_base"]
        pv_future_features = batch["pv_future_features"]
        load_features = batch["load_features"]
        gate_features = batch["gate_features_base"]
        base_pred = batch["base_pred_future"]
        future_ghi_raw = batch["future_ghi_raw"]
        delta_ghi_raw = batch["delta_ghi_raw"]
        hour_or_slot = batch["hour_or_slot"]

        if not self.cfg.feedback.use_self_feedback:
            steps = 1
        elif self.cfg.feedback.dynamic_stop:
            steps = int(self.cfg.feedback.safe_max_feedback_steps)
        else:
            steps = int(self.cfg.feedback.feedback_steps)
        steps = max(1, steps)
        y_iter = base_pred
        active_mask = torch.ones_like(base_pred, dtype=torch.bool)
        iteration_count = torch.zeros_like(base_pred)
        converged = torch.zeros_like(base_pred, dtype=torch.bool)
        prev_abs_actual_correction = torch.zeros_like(base_pred)
        prev_pv_weight = torch.full_like(base_pred, 0.5)
        prev_load_weight = torch.full_like(base_pred, 0.5)

        iter_preds = []
        pv_deltas = []
        load_deltas = []
        pv_weights = []
        load_weights = []
        feedback_rhos = []
        pv_contribs = []
        load_contribs = []
        total_corrections = []
        priors = []

        for _ in range(steps):
            # 自反馈关键点：未来残差不能使用真实 N_j，只能来自当前预测状态 y^(s)-base_pred。
            # stopgrad_feedback=True 时，后一轮专家输入状态不回传到前一轮，但本轮 correction 仍有梯度。
            y_input = y_iter.detach() if self.cfg.feedback.stopgrad_feedback else y_iter
            y_input = y_iter.detach() if self.cfg.feedback.stopgrad_feedback else y_iter
            base_input = base_pred.detach() if self.cfg.feedback.stopgrad_feedback else base_pred
            e_future = y_input - base_input

            future_step = torch.cat([e_future, pv_future_features], dim=1).unsqueeze(1)
            pv_seq = torch.cat([pv_hist, future_step], dim=1)
            pv_delta = self.pv_expert(pv_seq)

            load_input = torch.cat([load_features, y_input], dim=1)
            load_delta = self.load_expert(load_input)

            gate_input = torch.cat([gate_features, y_input], dim=1)
            weights, prior = self.gate(gate_input, future_ghi_raw, delta_ghi_raw, hour_or_slot)
            pv_weight = weights[:, 0:1]
            load_weight = weights[:, 1:2]

            feedback_input = torch.cat(
                [
                    gate_features,
                    y_input,
                    e_future,
                    prev_abs_actual_correction,
                    prev_pv_weight,
                    prev_load_weight,
                ],
                dim=1,
            )
            if self.cfg.feedback.use_feedback_controller:
                rho = self.feedback_controller(feedback_input)
            else:
                rho_value = min(max(float(self.cfg.feedback.gamma), 0.0), 1.0)
                rho = torch.full_like(base_pred, rho_value)

            raw_pv_contrib = pv_weight * pv_delta
            raw_load_contrib = load_weight * load_delta
            actual_pv_contrib = rho * raw_pv_contrib
            actual_load_contrib = rho * raw_load_contrib
            actual_correction = actual_pv_contrib + actual_load_contrib

            actual_pv_contrib = torch.where(active_mask, actual_pv_contrib, torch.zeros_like(actual_pv_contrib))
            actual_load_contrib = torch.where(active_mask, actual_load_contrib, torch.zeros_like(actual_load_contrib))
            actual_correction = torch.where(active_mask, actual_correction, torch.zeros_like(actual_correction))
            rho = torch.where(active_mask, rho, torch.zeros_like(rho))

            y_next = y_iter + actual_correction
            iteration_count = iteration_count + active_mask.float()
            step_converged = torch.abs(actual_correction) < float(self.cfg.feedback.convergence_eps)
            if self.cfg.feedback.dynamic_stop:
                active_mask_next = active_mask & (~step_converged)
            else:
                active_mask_next = active_mask
            converged = converged | (active_mask & step_converged)

            iter_preds.append(y_next)
            pv_deltas.append(pv_delta)
            load_deltas.append(load_delta)
            pv_weights.append(pv_weight)
            load_weights.append(load_weight)
            feedback_rhos.append(rho)
            pv_contribs.append(actual_pv_contrib)
            load_contribs.append(actual_load_contrib)
            total_corrections.append(actual_correction)
            priors.append(prior)

            if self.cfg.feedback.stopgrad_feedback:
                prev_abs_actual_correction = torch.abs(actual_correction.detach())
                prev_pv_weight = pv_weight.detach()
                prev_load_weight = load_weight.detach()
            else:
                prev_abs_actual_correction = torch.abs(actual_correction)
                prev_pv_weight = pv_weight
                prev_load_weight = load_weight
            y_iter = y_next
            active_mask = active_mask_next

        return {
            "base_pred": base_pred,
            "final_pred": iter_preds[-1],
            "iter_preds": torch.stack(iter_preds, dim=1),
            "pv_deltas": torch.stack(pv_deltas, dim=1),
            "load_deltas": torch.stack(load_deltas, dim=1),
            "pv_weights": torch.stack(pv_weights, dim=1),
            "load_weights": torch.stack(load_weights, dim=1),
            "feedback_rhos": torch.stack(feedback_rhos, dim=1),
            "pv_contribs": torch.stack(pv_contribs, dim=1),
            "load_contribs": torch.stack(load_contribs, dim=1),
            "total_corrections": torch.stack(total_corrections, dim=1),
            "iteration_count": iteration_count,
            "converged": converged,
            "prior_probs": torch.stack(priors, dim=1),
            "loss_aux_info": {},
        }

    def forward_one_step(
        self,
        batch: Dict[str, torch.Tensor],
        y_iter: torch.Tensor,
        prev_abs_actual_correction: torch.Tensor,
        prev_pv_weight: torch.Tensor,
        prev_load_weight: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pv_hist = batch["pv_hist_seq_base"]
        pv_future_features = batch["pv_future_features"]
        load_features = batch["load_features"]
        gate_features = batch["gate_features_base"]
        base_pred = batch["base_pred_future"]
        future_ghi_raw = batch["future_ghi_raw"]
        delta_ghi_raw = batch["delta_ghi_raw"]
        hour_or_slot = batch["hour_or_slot"]

        y_input = y_iter.detach() if self.cfg.feedback.stopgrad_feedback else y_iter
        base_input = base_pred.detach() if self.cfg.feedback.stopgrad_feedback else base_pred
        e_future = y_input - base_input

        future_step = torch.cat([e_future, pv_future_features], dim=1).unsqueeze(1)
        pv_seq = torch.cat([pv_hist, future_step], dim=1)
        pv_delta = self.pv_expert(pv_seq)

        load_input = torch.cat([load_features, y_input], dim=1)
        load_delta = self.load_expert(load_input)

        gate_input = torch.cat([gate_features, y_input], dim=1)
        weights, prior = self.gate(gate_input, future_ghi_raw, delta_ghi_raw, hour_or_slot)
        pv_weight = weights[:, 0:1]
        load_weight = weights[:, 1:2]

        feedback_input = torch.cat(
            [
                gate_features,
                y_input,
                e_future,
                prev_abs_actual_correction,
                prev_pv_weight,
                prev_load_weight,
            ],
            dim=1,
        )
        if self.cfg.feedback.use_feedback_controller:
            rho = self.feedback_controller(feedback_input)
        else:
            rho_value = min(max(float(self.cfg.feedback.gamma), 0.0), 1.0)
            rho = torch.full_like(base_pred, rho_value)

        raw_pv_contrib = pv_weight * pv_delta
        raw_load_contrib = load_weight * load_delta
        actual_pv_contrib = rho * raw_pv_contrib
        actual_load_contrib = rho * raw_load_contrib
        actual_total_correction = actual_pv_contrib + actual_load_contrib

        actual_pv_contrib = torch.where(active_mask, actual_pv_contrib, torch.zeros_like(actual_pv_contrib))
        actual_load_contrib = torch.where(active_mask, actual_load_contrib, torch.zeros_like(actual_load_contrib))
        actual_total_correction = torch.where(
            active_mask,
            actual_total_correction,
            torch.zeros_like(actual_total_correction),
        )
        rho = torch.where(active_mask, rho, torch.zeros_like(rho))

        y_next = y_iter + actual_total_correction
        return {
            "y_input": y_input,
            "y_next": y_next,
            "pv_delta": pv_delta,
            "load_delta": load_delta,
            "pv_weight": pv_weight,
            "load_weight": load_weight,
            "feedback_rho": rho,
            "actual_pv_contrib": actual_pv_contrib,
            "actual_load_contrib": actual_load_contrib,
            "actual_total_correction": actual_total_correction,
            "prior": prior,
            "active_mask": active_mask,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        base_pred = batch["base_pred_future"]
        if not self.cfg.feedback.use_self_feedback:
            steps = 1
        elif self.cfg.feedback.dynamic_stop:
            steps = int(self.cfg.feedback.safe_max_feedback_steps)
        else:
            steps = int(self.cfg.feedback.feedback_steps)
        steps = max(1, steps)

        y_iter = base_pred
        active_mask = torch.ones_like(base_pred, dtype=torch.bool)
        iteration_count = torch.zeros_like(base_pred)
        converged = torch.zeros_like(base_pred, dtype=torch.bool)
        prev_abs_actual_correction = torch.zeros_like(base_pred)
        prev_pv_weight = torch.full_like(base_pred, 0.5)
        prev_load_weight = torch.full_like(base_pred, 0.5)

        iter_preds = []
        pv_deltas = []
        load_deltas = []
        pv_weights = []
        load_weights = []
        feedback_rhos = []
        pv_contribs = []
        load_contribs = []
        total_corrections = []
        priors = []

        for _ in range(steps):
            step_outputs = self.forward_one_step(
                batch=batch,
                y_iter=y_iter,
                prev_abs_actual_correction=prev_abs_actual_correction,
                prev_pv_weight=prev_pv_weight,
                prev_load_weight=prev_load_weight,
                active_mask=active_mask,
            )
            y_next = step_outputs["y_next"]
            actual_total_correction = step_outputs["actual_total_correction"]
            iteration_count = iteration_count + active_mask.float()
            step_converged = torch.abs(actual_total_correction) < float(self.cfg.feedback.convergence_eps)
            if self.cfg.feedback.dynamic_stop:
                active_mask_next = active_mask & (~step_converged)
            else:
                active_mask_next = active_mask
            converged = converged | (active_mask & step_converged)

            iter_preds.append(y_next)
            pv_deltas.append(step_outputs["pv_delta"])
            load_deltas.append(step_outputs["load_delta"])
            pv_weights.append(step_outputs["pv_weight"])
            load_weights.append(step_outputs["load_weight"])
            feedback_rhos.append(step_outputs["feedback_rho"])
            pv_contribs.append(step_outputs["actual_pv_contrib"])
            load_contribs.append(step_outputs["actual_load_contrib"])
            total_corrections.append(actual_total_correction)
            priors.append(step_outputs["prior"])

            if self.cfg.feedback.stopgrad_feedback:
                prev_abs_actual_correction = torch.abs(actual_total_correction.detach())
                prev_pv_weight = step_outputs["pv_weight"].detach()
                prev_load_weight = step_outputs["load_weight"].detach()
            else:
                prev_abs_actual_correction = torch.abs(actual_total_correction)
                prev_pv_weight = step_outputs["pv_weight"]
                prev_load_weight = step_outputs["load_weight"]
            y_iter = y_next
            active_mask = active_mask_next

        return {
            "base_pred": base_pred,
            "final_pred": iter_preds[-1],
            "iter_preds": torch.stack(iter_preds, dim=1),
            "pv_deltas": torch.stack(pv_deltas, dim=1),
            "load_deltas": torch.stack(load_deltas, dim=1),
            "pv_weights": torch.stack(pv_weights, dim=1),
            "load_weights": torch.stack(load_weights, dim=1),
            "feedback_rhos": torch.stack(feedback_rhos, dim=1),
            "pv_contribs": torch.stack(pv_contribs, dim=1),
            "load_contribs": torch.stack(load_contribs, dim=1),
            "total_corrections": torch.stack(total_corrections, dim=1),
            "iteration_count": iteration_count,
            "converged": converged,
            "prior_probs": torch.stack(priors, dim=1),
            "loss_aux_info": {},
        }


class ExpertRefinementDataset(Dataset):
    def __init__(self, split: PreparedSplit, base_pred_series: np.ndarray, cfg: FullConfig):
        self.split = split
        self.cfg = cfg
        self.seq_len = int(cfg.data.seq_len)
        self.y = np.asarray(split.y_scaled, dtype=np.float32)
        self.base_pred = np.asarray(base_pred_series, dtype=np.float32)
        if len(self.y) != len(self.base_pred):
            raise ValueError("base_pred_series length must match split length.")

        self.e_base = self.y - self.base_pred
        self.pv_future_cols = self._pv_future_cols()
        self.pv_hist_cols = self.pv_future_cols
        self.load_time_cols = selected_load_time_cols(cfg)
        self.gate_time_cols = selected_gate_time_cols(cfg)
        self.indices = self._valid_indices()
        if not self.indices:
            raise ValueError(
                f"No valid expert samples for split={split.name!r}. "
                f"Need target index j >= 2*seq_len={2 * self.seq_len} and finite base predictions "
                "for every historical residual in [j-L, j-1]."
            )

    def _pv_future_cols(self) -> List[str]:
        cols = []
        if self.cfg.features.use_ghi:
            cols.append(self.cfg.data.ghi_col)
        if self.cfg.features.use_temp:
            cols.append(self.cfg.data.temp_c_col)
        if self.cfg.features.use_wind:
            cols.append(self.cfg.data.wind_col)
        if self.cfg.features.use_slot_sin_cos:
            cols += ["slot_sin", "slot_cos"]
        if self.cfg.features.use_month_sin_cos:
            cols += ["month_sin", "month_cos"]
        return cols

    def _valid_indices(self) -> List[int]:
        indices = []
        start = 2 * self.seq_len
        for j in range(start, len(self.y)):
            hist_base = self.base_pred[j - self.seq_len : j]
            if np.isfinite(self.base_pred[j]) and np.all(np.isfinite(hist_base)):
                indices.append(j)
        return indices

    def __len__(self):
        return len(self.indices)

    def _scaled(self, col: str, idx) -> np.ndarray:
        return self.split.feature_scaled[col].to_numpy(dtype=np.float32)[idx]

    def __getitem__(self, item):
        j = self.indices[item]
        L = self.seq_len
        hist_slice = slice(j - L, j)

        hist_parts = [self.e_base[hist_slice].reshape(L, 1)]
        for col in self.pv_hist_cols:
            hist_parts.append(self._scaled(col, hist_slice).reshape(L, 1))
        pv_hist_seq = np.concatenate(hist_parts, axis=1).astype(np.float32)

        pv_future = np.array([self._scaled(col, j) for col in self.pv_future_cols], dtype=np.float32)

        short = min(self.cfg.features.recent_short_points, j)
        long = min(self.cfg.features.recent_long_points, j)
        y_hist_short = self.y[j - short : j]
        y_hist_long = self.y[j - long : j]
        lag = min(self.cfg.features.recent_short_points, j)
        n_last = self.y[j - 1]
        n_lag = self.y[j - 1 - lag] if j - 1 - lag >= 0 else self.y[0]

        load_values = []
        for col in self.load_time_cols:
            load_values.append(float(self._scaled(col, j)))
        if self.cfg.features.use_temp:
            load_values.append(float(self._scaled(self.cfg.data.temp_c_col, j)))
        load_values.extend(
            [
                float(n_last),
                float(np.mean(y_hist_short)),
                float(np.mean(y_hist_long)),
                float(np.std(y_hist_long)),
                float(n_last - n_lag),
            ]
        )

        gate_values = []
        if self.cfg.features.use_ghi:
            ghi_j_scaled = float(self._scaled(self.cfg.data.ghi_col, j))
            ghi_prev_scaled = float(self._scaled(self.cfg.data.ghi_col, j - 1))
            gate_values.extend(
                [
                    ghi_j_scaled,
                    ghi_j_scaled - ghi_prev_scaled,
                    abs(ghi_j_scaled - ghi_prev_scaled),
                ]
            )
        for col in self.gate_time_cols:
            gate_values.append(float(self._scaled(col, j)))
        gate_values.extend([float(np.std(y_hist_long)), float(n_last - n_lag)])

        if self.cfg.features.use_ghi:
            raw_ghi = float(self.split.df.iloc[j][self.cfg.data.ghi_col])
            raw_ghi_prev = float(self.split.df.iloc[j - 1][self.cfg.data.ghi_col])
        else:
            raw_ghi = float(self.cfg.features.ghi_day_threshold + 1.0)
            raw_ghi_prev = raw_ghi
        slot = float(self.split.df.iloc[j]["slot"])

        return {
            "pv_hist_seq_base": torch.from_numpy(pv_hist_seq),
            "pv_future_features": torch.from_numpy(pv_future),
            "load_features": torch.tensor(load_values, dtype=torch.float32),
            "gate_features_base": torch.tensor(gate_values, dtype=torch.float32),
            "base_pred_future": torch.tensor([self.base_pred[j]], dtype=torch.float32),
            "target": torch.tensor([self.y[j]], dtype=torch.float32),
            "future_ghi_raw": torch.tensor([raw_ghi], dtype=torch.float32),
            "delta_ghi_raw": torch.tensor([raw_ghi - raw_ghi_prev], dtype=torch.float32),
            "hour_or_slot": torch.tensor([slot], dtype=torch.float32),
            "target_index": torch.tensor(j, dtype=torch.long),
        }


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_base_model(
    model: TrendPeriodBaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: FullConfig,
    save_dir: str,
) -> pd.DataFrame:
    device = cfg.train.device
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    best_val = float("inf")
    bad_epochs = 0
    rows = []
    best_path = os.path.join(save_dir, "best_base_model.pth")

    for epoch in range(1, cfg.train.base_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            pred = model(batch["net_seq"], batch["time_seq"], batch["time_future"])
            loss = F.mse_loss(pred, batch["target"])
            if cfg.base.smoothness_lambda > 0.0:
                loss = loss + cfg.base.smoothness_lambda * F.mse_loss(pred, batch["net_seq"][:, -1, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss = evaluate_base_loss(model, val_loader, cfg)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[base] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                print(f"[base] early stopping at epoch {epoch}.")
                break

    log_df = pd.DataFrame(rows)
    log_df.to_csv(os.path.join(save_dir, "base_training_log.csv"), index=False, encoding="utf-8-sig")
    return log_df


@torch.no_grad()
def evaluate_base_loss(model: TrendPeriodBaseModel, loader: DataLoader, cfg: FullConfig) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = move_batch(batch, cfg.train.device)
        pred = model(batch["net_seq"], batch["time_seq"], batch["time_future"])
        loss = F.mse_loss(pred, batch["target"])
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def predict_base_series(
    model: TrendPeriodBaseModel,
    split: PreparedSplit,
    time_cols: List[str],
    cfg: FullConfig,
) -> np.ndarray:
    dataset = BaseWindowDataset(split, time_cols, cfg)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False)
    preds = np.full(len(split.y_scaled), np.nan, dtype=np.float32)
    model.eval()
    for batch in loader:
        batch_dev = move_batch(batch, cfg.train.device)
        pred = model(batch_dev["net_seq"], batch_dev["time_seq"], batch_dev["time_future"])
        idx = batch["target_index"].cpu().numpy()
        preds[idx] = pred.detach().cpu().numpy().reshape(-1)
    return preds


def gather_last_valid_step_torch(values: torch.Tensor, iteration_count: torch.Tensor) -> torch.Tensor:
    steps = values.shape[1]
    last_idx = torch.clamp(iteration_count.long().view(-1) - 1, min=0, max=steps - 1)
    gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, values.shape[-1])
    return torch.gather(values, dim=1, index=gather_idx).squeeze(1)


def compute_refinement_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    future_ghi_raw: torch.Tensor,
    cfg: FullConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    final_loss = F.mse_loss(outputs["final_pred"], target)
    night_mask = (future_ghi_raw < cfg.features.ghi_day_threshold).float().unsqueeze(1)
    pv_contribs = outputs["pv_contribs"]
    load_contribs = outputs["load_contribs"]
    night_loss = torch.mean(night_mask * (pv_contribs ** 2))

    total_corrections = outputs["total_corrections"]
    if total_corrections.shape[1] > 1:
        prev_abs = torch.abs(total_corrections[:, :-1, :])
        next_abs = torch.abs(total_corrections[:, 1:, :])
        conv_loss = torch.mean(F.relu(next_abs - prev_abs) ** 2)
    else:
        conv_loss = torch.zeros((), device=target.device)

    total = (
        cfg.loss.mse_loss_weight * final_loss
        + cfg.loss.night_loss_weight * night_loss
        + cfg.loss.conv_loss_weight * conv_loss
    )
    iteration_count = outputs["iteration_count"]
    converged = outputs["converged"]
    last_total_correction = gather_last_valid_step_torch(total_corrections, iteration_count)
    last_pv_contrib = gather_last_valid_step_torch(pv_contribs, iteration_count)
    last_load_contrib = gather_last_valid_step_torch(load_contribs, iteration_count)
    last_feedback_rho = gather_last_valid_step_torch(outputs["feedback_rhos"], iteration_count)
    parts = {
        "total_loss": float(total.detach().cpu().item()),
        "final_mse": float(final_loss.detach().cpu().item()),
        "night_loss": float(night_loss.detach().cpu().item()),
        "conv_loss": float(conv_loss.detach().cpu().item()),
        "avg_iteration_count": float(iteration_count.detach().float().mean().cpu().item()),
        "converged_ratio": float(converged.detach().float().mean().cpu().item()),
        "mean_abs_last_total_correction": float(torch.abs(last_total_correction).mean().detach().cpu().item()),
        "mean_abs_last_pv_contrib": float(torch.abs(last_pv_contrib).mean().detach().cpu().item()),
        "mean_abs_last_load_contrib": float(torch.abs(last_load_contrib).mean().detach().cpu().item()),
        "mean_feedback_rho_last": float(last_feedback_rho.mean().detach().cpu().item()),
    }
    return total, parts


def compute_step_refinement_loss(
    step_outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    future_ghi_raw: torch.Tensor,
    prev_abs_actual_correction: torch.Tensor,
    active_mask: torch.Tensor,
    cfg: FullConfig,
    step_index: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    active_float = active_mask.float()
    active_count = active_float.sum()
    active_ratio = float(active_float.mean().detach().cpu().item())
    if float(active_count.detach().cpu().item()) <= 0.0:
        zero = target.sum() * 0.0
        return zero, {
            "step_total_loss": 0.0,
            "step_mse": 0.0,
            "step_night_loss": 0.0,
            "step_conv_loss": 0.0,
            "step_active_ratio": 0.0,
            "step_mean_abs_total_correction": 0.0,
            "step_mean_abs_pv_actual_contrib": 0.0,
            "step_mean_abs_load_actual_contrib": 0.0,
            "step_mean_feedback_rho": 0.0,
            "step_converged_ratio_active": 0.0,
        }

    active_count = active_count.clamp(min=1.0)

    def active_mean(values: torch.Tensor) -> torch.Tensor:
        return (active_float * values).sum() / active_count

    pred_err = (step_outputs["y_next"] - target) ** 2
    step_mse = active_mean(pred_err)

    night_mask = (future_ghi_raw < cfg.features.ghi_day_threshold).float()
    step_night_loss = active_mean(night_mask * (step_outputs["actual_pv_contrib"] ** 2))

    curr_abs = torch.abs(step_outputs["actual_total_correction"])
    if step_index == 0:
        step_conv_loss = torch.zeros((), device=target.device, dtype=target.dtype)
    else:
        step_conv_loss = active_mean(F.relu(curr_abs - prev_abs_actual_correction) ** 2)

    step_total_loss = (
        cfg.loss.mse_loss_weight * step_mse
        + cfg.loss.night_loss_weight * step_night_loss
        + cfg.loss.conv_loss_weight * step_conv_loss
    )
    step_converged = curr_abs < float(cfg.feedback.convergence_eps)
    parts = {
        "step_total_loss": float(step_total_loss.detach().cpu().item()),
        "step_mse": float(step_mse.detach().cpu().item()),
        "step_night_loss": float(step_night_loss.detach().cpu().item()),
        "step_conv_loss": float(step_conv_loss.detach().cpu().item()),
        "step_active_ratio": active_ratio,
        "step_mean_abs_total_correction": float(active_mean(curr_abs).detach().cpu().item()),
        "step_mean_abs_pv_actual_contrib": float(
            active_mean(torch.abs(step_outputs["actual_pv_contrib"])).detach().cpu().item()
        ),
        "step_mean_abs_load_actual_contrib": float(
            active_mean(torch.abs(step_outputs["actual_load_contrib"])).detach().cpu().item()
        ),
        "step_mean_feedback_rho": float(active_mean(step_outputs["feedback_rho"]).detach().cpu().item()),
        "step_converged_ratio_active": float(active_mean(step_converged.float()).detach().cpu().item()),
    }
    return step_total_loss, parts


def train_refinement_model(
    model: IterativeDualExpertNetLoadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: FullConfig,
    save_dir: str,
) -> pd.DataFrame:
    params = [p for p in model.parameters() if p.requires_grad]
    expert_lr = cfg.train.expert_lr if cfg.train.expert_lr is not None else cfg.train.lr
    optimizer = torch.optim.Adam(params, lr=float(expert_lr), weight_decay=cfg.train.weight_decay)
    best_val = float("inf")
    bad_epochs = 0
    rows = []
    step_rows = []
    best_path = os.path.join(save_dir, "best_refinement_model.pth")

    for epoch in range(1, cfg.train.expert_epochs + 1):
        model.train()
        train_parts: Dict[str, List[float]] = {}
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, cfg.train.device)
            base_pred = batch["base_pred_future"]
            y_iter = base_pred.detach()
            active_mask = torch.ones_like(base_pred, dtype=torch.bool)
            iteration_count = torch.zeros_like(base_pred)
            converged = torch.zeros_like(base_pred, dtype=torch.bool)
            prev_abs_actual_correction = torch.zeros_like(base_pred)
            prev_pv_weight = torch.full_like(base_pred, 0.5)
            prev_load_weight = torch.full_like(base_pred, 0.5)

            if not cfg.feedback.use_self_feedback:
                max_steps = 1
            elif cfg.feedback.dynamic_stop:
                max_steps = int(cfg.feedback.safe_max_feedback_steps)
            else:
                max_steps = int(cfg.feedback.feedback_steps)
            max_steps = max(1, max_steps)

            for step_index in range(max_steps):
                active_before = active_mask.clone()
                if int(active_before.sum().detach().cpu().item()) == 0:
                    break

                step_outputs = model.forward_one_step(
                    batch=batch,
                    y_iter=y_iter,
                    prev_abs_actual_correction=prev_abs_actual_correction,
                    prev_pv_weight=prev_pv_weight,
                    prev_load_weight=prev_load_weight,
                    active_mask=active_before,
                )
                step_loss, step_parts = compute_step_refinement_loss(
                    step_outputs=step_outputs,
                    target=batch["target"],
                    future_ghi_raw=batch["future_ghi_raw"],
                    prev_abs_actual_correction=prev_abs_actual_correction,
                    active_mask=active_before,
                    cfg=cfg,
                    step_index=step_index,
                )

                optimizer.zero_grad()
                step_loss.backward()
                optimizer.step()

                for k, v in step_parts.items():
                    train_parts.setdefault(k, []).append(v)
                step_rows.append(
                    {
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "step_index": step_index,
                        **step_parts,
                    }
                )

                with torch.no_grad():
                    actual_total_correction = step_outputs["actual_total_correction"].detach()
                    step_converged = torch.abs(actual_total_correction) < float(cfg.feedback.convergence_eps)
                    iteration_count = iteration_count + active_before.float()
                    if cfg.feedback.dynamic_stop:
                        active_mask = active_before & (~step_converged)
                    else:
                        active_mask = active_before
                    converged = converged | (active_before & step_converged)
                    y_iter = step_outputs["y_next"].detach()
                    prev_abs_actual_correction = torch.abs(actual_total_correction).detach()
                    prev_pv_weight = step_outputs["pv_weight"].detach()
                    prev_load_weight = step_outputs["load_weight"].detach()

            train_parts.setdefault("avg_iteration_count", []).append(
                float(iteration_count.detach().float().mean().cpu().item())
            )
            train_parts.setdefault("converged_ratio", []).append(float(converged.detach().float().mean().cpu().item()))

        val_parts = evaluate_refinement_loss(model, val_loader, cfg)
        row = {"epoch": epoch}
        for k, values in train_parts.items():
            row[f"train_{k}"] = float(np.mean(values))
        for k, v in val_parts.items():
            row[f"val_{k}"] = v
        rows.append(row)
        print(
            f"[refine] epoch={epoch:03d} "
            f"train_loss={row.get('train_step_total_loss', float('nan')):.6f} "
            f"val_loss={row.get('val_total_loss', float('nan')):.6f} "
            f"train_iter={row.get('train_avg_iteration_count', float('nan')):.2f} "
            f"val_iter={row.get('val_avg_iteration_count', float('nan')):.2f} "
            f"train_converged={row.get('train_converged_ratio', float('nan')):.2%} "
            f"val_converged={row.get('val_converged_ratio', float('nan')):.2%}"
        )

        val_loss = row["val_total_loss"]
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                print(f"[refine] early stopping at epoch {epoch}.")
                break

    log_df = pd.DataFrame(rows)
    log_df.to_csv(os.path.join(save_dir, "refinement_training_log.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(step_rows).to_csv(
        os.path.join(save_dir, "refinement_step_training_log.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return log_df


@torch.no_grad()
def evaluate_refinement_loss(model: IterativeDualExpertNetLoadModel, loader: DataLoader, cfg: FullConfig) -> Dict[str, float]:
    model.eval()
    parts_all: Dict[str, List[float]] = {}
    for batch in loader:
        batch = move_batch(batch, cfg.train.device)
        outputs = model(batch)
        _, parts = compute_refinement_loss(outputs, batch["target"], batch["future_ghi_raw"], cfg)
        for k, v in parts.items():
            parts_all.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in parts_all.items()}


@torch.no_grad()
def predict_refinement(
    model: IterativeDualExpertNetLoadModel,
    loader: DataLoader,
    cfg: FullConfig,
) -> Dict[str, np.ndarray]:
    model.eval()
    cols: Dict[str, List[np.ndarray]] = {
        "target_index": [],
        "target": [],
        "base_pred": [],
        "final_pred": [],
        "future_ghi_raw": [],
        "delta_ghi_raw": [],
        "hour_or_slot": [],
        "iter_preds": [],
        "pv_weights": [],
        "load_weights": [],
        "feedback_rhos": [],
        "pv_deltas": [],
        "load_deltas": [],
        "pv_contribs": [],
        "load_contribs": [],
        "total_corrections": [],
        "iteration_count": [],
        "converged": [],
    }
    for batch in loader:
        batch_dev = move_batch(batch, cfg.train.device)
        outputs = model(batch_dev)
        cols["target_index"].append(batch["target_index"].cpu().numpy().reshape(-1))
        cols["target"].append(batch["target"].cpu().numpy().reshape(-1))
        cols["base_pred"].append(outputs["base_pred"].cpu().numpy().reshape(-1))
        cols["final_pred"].append(outputs["final_pred"].cpu().numpy().reshape(-1))
        cols["future_ghi_raw"].append(batch["future_ghi_raw"].cpu().numpy().reshape(-1))
        cols["delta_ghi_raw"].append(batch["delta_ghi_raw"].cpu().numpy().reshape(-1))
        cols["hour_or_slot"].append(batch["hour_or_slot"].cpu().numpy().reshape(-1))
        cols["iteration_count"].append(outputs["iteration_count"].cpu().numpy().reshape(-1))
        cols["converged"].append(outputs["converged"].cpu().numpy().reshape(-1))
        for k in [
            "iter_preds",
            "pv_weights",
            "load_weights",
            "feedback_rhos",
            "pv_deltas",
            "load_deltas",
            "pv_contribs",
            "load_contribs",
            "total_corrections",
        ]:
            cols[k].append(outputs[k].cpu().numpy().squeeze(-1))

    out: Dict[str, np.ndarray] = {}
    for k, chunks in cols.items():
        if not chunks:
            out[k] = np.array([])
        elif k in [
            "iter_preds",
            "pv_weights",
            "load_weights",
            "feedback_rhos",
            "pv_deltas",
            "load_deltas",
            "pv_contribs",
            "load_contribs",
            "total_corrections",
        ]:
            out[k] = np.concatenate(chunks, axis=0)
        else:
            out[k] = np.concatenate(chunks, axis=0)
    return out


def make_base_loaders(prepared: PreparedData, cfg: FullConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    kwargs = {"batch_size": cfg.train.batch_size}
    train_ds = BaseWindowDataset(prepared.train, prepared.time_cols, cfg)
    val_ds = BaseWindowDataset(prepared.val, prepared.time_cols, cfg)
    test_ds = BaseWindowDataset(prepared.test, prepared.time_cols, cfg)
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError("One of base datasets is empty. Check split sizes and seq_len.")
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


def make_expert_loaders(
    prepared: PreparedData,
    base_train: np.ndarray,
    base_val: np.ndarray,
    base_test: np.ndarray,
    cfg: FullConfig,
) -> Tuple[ExpertRefinementDataset, ExpertRefinementDataset, ExpertRefinementDataset, DataLoader, DataLoader, DataLoader]:
    train_ds = ExpertRefinementDataset(prepared.train, base_train, cfg)
    val_ds = ExpertRefinementDataset(prepared.val, base_val, cfg)
    test_ds = ExpertRefinementDataset(prepared.test, base_test, cfg)
    kwargs = {"batch_size": cfg.train.batch_size}
    return (
        train_ds,
        val_ds,
        test_ds,
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


def last_valid_step_indices(iteration_count: np.ndarray, steps: int) -> np.ndarray:
    idx = np.asarray(iteration_count).astype(np.int64) - 1
    return np.clip(idx, 0, steps - 1)


def take_last_step(values: np.ndarray, last_idx: np.ndarray) -> np.ndarray:
    rows = np.arange(values.shape[0])
    return values[rows, last_idx]


def save_prediction_outputs(
    arrays: Dict[str, np.ndarray],
    test_ds: ExpertRefinementDataset,
    prepared: PreparedData,
    cfg: FullConfig,
    save_dir: str,
):
    idx = arrays["target_index"].astype(int)
    timestamps = test_ds.split.df.iloc[idx][cfg.data.datetime_col].astype(str).to_numpy()
    y_true_scaled = arrays["target"]
    base_scaled = arrays["base_pred"]
    final_scaled = arrays["final_pred"]
    y_true = inverse_transform_1d(prepared.y_scaler, y_true_scaled)
    base_pred = inverse_transform_1d(prepared.y_scaler, base_scaled)
    final_pred = inverse_transform_1d(prepared.y_scaler, final_scaled)

    pred_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "slot": arrays["hour_or_slot"].astype(int),
            "y_true_scaled": y_true_scaled,
            "base_pred_scaled": base_scaled,
            "final_pred_scaled": final_scaled,
            "y_true": y_true,
            "base_pred": base_pred,
            "final_pred": final_pred,
            "future_ghi_raw": arrays["future_ghi_raw"],
            "delta_ghi_raw": arrays["delta_ghi_raw"],
            "iteration_count": arrays["iteration_count"].astype(np.int64),
            "converged": arrays["converged"].astype(bool),
        }
    )

    steps = arrays["iter_preds"].shape[1]
    step_cols = {}
    for s in range(steps):
        step = s + 1
        step_cols[f"iter_pred_scaled_s{step}"] = arrays["iter_preds"][:, s]
        step_cols[f"iter_pred_s{step}"] = inverse_transform_1d(prepared.y_scaler, arrays["iter_preds"][:, s])
        step_cols[f"feedback_rho_s{step}"] = arrays["feedback_rhos"][:, s]
        step_cols[f"pv_weight_s{step}"] = arrays["pv_weights"][:, s]
        step_cols[f"load_weight_s{step}"] = arrays["load_weights"][:, s]
        step_cols[f"pv_delta_s{step}"] = arrays["pv_deltas"][:, s]
        step_cols[f"load_delta_s{step}"] = arrays["load_deltas"][:, s]
        step_cols[f"pv_actual_contrib_s{step}"] = arrays["pv_contribs"][:, s]
        step_cols[f"load_actual_contrib_s{step}"] = arrays["load_contribs"][:, s]
        step_cols[f"actual_total_correction_s{step}"] = arrays["total_corrections"][:, s]

    last_idx = last_valid_step_indices(arrays["iteration_count"], steps)
    step_cols.update(
        {
            "feedback_rho_last": take_last_step(arrays["feedback_rhos"], last_idx),
            "pv_weight_last": take_last_step(arrays["pv_weights"], last_idx),
            "load_weight_last": take_last_step(arrays["load_weights"], last_idx),
            "pv_delta_last": take_last_step(arrays["pv_deltas"], last_idx),
            "load_delta_last": take_last_step(arrays["load_deltas"], last_idx),
            "pv_actual_contrib_last": take_last_step(arrays["pv_contribs"], last_idx),
            "load_actual_contrib_last": take_last_step(arrays["load_contribs"], last_idx),
            "actual_total_correction_last": take_last_step(arrays["total_corrections"], last_idx),
        }
    )
    pred_df = pd.concat([pred_df, pd.DataFrame(step_cols, index=pred_df.index)], axis=1).copy()
    pred_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False, encoding="utf-8-sig")

    base_metrics = compute_metrics(y_true, base_pred)
    final_metrics = compute_metrics(y_true, final_pred)
    improvement = (base_metrics["RMSE"] - final_metrics["RMSE"]) / (base_metrics["RMSE"] + 1e-8) * 100.0
    metrics_row = {}
    for k, v in base_metrics.items():
        metrics_row[f"base_{k}"] = v
    for k, v in final_metrics.items():
        metrics_row[f"final_{k}"] = v
    metrics_row["RMSE_improvement_percent"] = float(improvement)
    pd.DataFrame([metrics_row]).to_csv(os.path.join(save_dir, "test_metrics.csv"), index=False, encoding="utf-8-sig")
    print(
        f"[test] avg_iteration_count={pred_df['iteration_count'].mean():.2f} "
        f"max_iteration_count={pred_df['iteration_count'].max():.0f} "
        f"converged_ratio={pred_df['converged'].mean():.2%}"
    )

    hourly = (
        pred_df.groupby("slot")
        .agg(
            sample_count=("timestamp", "size"),
            mean_iteration_count=("iteration_count", "mean"),
            converged_ratio=("converged", "mean"),
            mean_pv_weight_last=("pv_weight_last", "mean"),
            mean_load_weight_last=("load_weight_last", "mean"),
            mean_feedback_rho_last=("feedback_rho_last", "mean"),
            mean_abs_total_correction_last=(
                "actual_total_correction_last",
                lambda x: float(np.mean(np.abs(x))),
            ),
            mean_abs_pv_actual_contrib_last=(
                "pv_actual_contrib_last",
                lambda x: float(np.mean(np.abs(x))),
            ),
            mean_abs_load_actual_contrib_last=(
                "load_actual_contrib_last",
                lambda x: float(np.mean(np.abs(x))),
            ),
            mean_pv_actual_contrib_last=("pv_actual_contrib_last", "mean"),
            mean_load_actual_contrib_last=("load_actual_contrib_last", "mean"),
            mean_abs_delta_ghi=("delta_ghi_raw", lambda x: float(np.mean(np.abs(x)))),
            mean_ghi=("future_ghi_raw", "mean"),
        )
        .reset_index()
    )
    hourly.to_csv(os.path.join(save_dir, "hourly_gate_summary.csv"), index=False, encoding="utf-8-sig")
    save_plots(pred_df, hourly, save_dir)


def _plot_prediction(path: str, pred_df: pd.DataFrame, cols: List[Tuple[str, str]], title: str, max_points: int = 300):
    n = min(max_points, len(pred_df))
    plt.figure(figsize=(12, 5))
    x = np.arange(n)
    for col, label in cols:
        plt.plot(x, pred_df[col].to_numpy()[:n], label=label)
    plt.title(title)
    plt.xlabel("sample")
    plt.ylabel("net load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_plots(pred_df: pd.DataFrame, hourly: pd.DataFrame, save_dir: str):
    _plot_prediction(
        os.path.join(save_dir, "test_prediction.png"),
        pred_df,
        [("y_true", "true"), ("final_pred", "final")],
        "Test prediction",
    )
    _plot_prediction(
        os.path.join(save_dir, "base_vs_final_prediction.png"),
        pred_df,
        [("y_true", "true"), ("base_pred", "base"), ("final_pred", "final")],
        "Base vs final prediction",
    )

    plt.figure(figsize=(10, 4))
    plt.plot(hourly["slot"], hourly["mean_pv_weight_last"], label="pv weight")
    plt.plot(hourly["slot"], hourly["mean_load_weight_last"], label="load weight")
    plt.xlabel("half-hour slot")
    plt.ylabel("weight")
    plt.title("Hourly gate weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hourly_gate_weight.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(hourly["slot"], hourly["mean_pv_actual_contrib_last"], label="pv actual contribution")
    plt.plot(hourly["slot"], hourly["mean_load_actual_contrib_last"], label="load actual contribution")
    plt.xlabel("half-hour slot")
    plt.ylabel("scaled correction")
    plt.title("Hourly contribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hourly_contribution.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(hourly["slot"], hourly["mean_iteration_count"], label="iteration count")
    plt.xlabel("half-hour slot")
    plt.ylabel("mean iterations")
    plt.title("Hourly feedback iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hourly_feedback_iteration.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(hourly["slot"], hourly["mean_feedback_rho_last"], label="feedback rho")
    plt.xlabel("half-hour slot")
    plt.ylabel("rho")
    plt.title("Hourly feedback rho")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hourly_feedback_rho.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(hourly["slot"], hourly["mean_pv_actual_contrib_last"], label="pv actual contribution")
    plt.plot(hourly["slot"], hourly["mean_load_actual_contrib_last"], label="load actual contribution")
    plt.xlabel("half-hour slot")
    plt.ylabel("scaled correction")
    plt.title("Hourly actual contribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hourly_actual_contribution.png"), dpi=200)
    plt.close()


def save_loss_curve(base_log: pd.DataFrame, refine_log: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not base_log.empty:
        axes[0].plot(base_log["epoch"], base_log["train_loss"], label="train")
        axes[0].plot(base_log["epoch"], base_log["val_loss"], label="val")
    axes[0].set_title("Base loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    if not refine_log.empty:
        refine_train_col = "train_total_loss"
        if refine_train_col not in refine_log.columns and "train_step_total_loss" in refine_log.columns:
            refine_train_col = "train_step_total_loss"
        if refine_train_col in refine_log.columns:
            axes[1].plot(refine_log["epoch"], refine_log[refine_train_col], label="train")
        if "val_total_loss" in refine_log.columns:
            axes[1].plot(refine_log["epoch"], refine_log["val_total_loss"], label="val")
    axes[1].set_title("Refinement loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_val_loss_curve.png"), dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative dual-expert net-load forecasting: trend-period baseline + PV/load residual correction."
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--pv-backbone", type=str, choices=["cnn_lstm_attention", "weibull_lstm"], default=None)
    parser.add_argument("--use-self-feedback", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-feedback-controller", type=int, choices=[0, 1], default=None)
    parser.add_argument("--dynamic-stop", type=int, choices=[0, 1], default=None)
    parser.add_argument("--feedback-steps", type=int, default=None)
    parser.add_argument("--safe-max-feedback-steps", type=int, default=None)
    parser.add_argument("--convergence-eps", type=float, default=None)
    parser.add_argument("--conv-loss-weight", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override both base_epochs and expert_epochs.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--expert-lr", type=float, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg: FullConfig, args: argparse.Namespace) -> FullConfig:
    if args.data_path is not None:
        cfg.data.data_path = args.data_path
    if args.save_dir is not None:
        cfg.data.save_dir = args.save_dir
    if args.pv_backbone is not None:
        cfg.pv_expert.backbone = args.pv_backbone
    if args.use_self_feedback is not None:
        cfg.feedback.use_self_feedback = bool(args.use_self_feedback)
    if args.use_feedback_controller is not None:
        cfg.feedback.use_feedback_controller = bool(args.use_feedback_controller)
    if args.dynamic_stop is not None:
        cfg.feedback.dynamic_stop = bool(args.dynamic_stop)
    if args.feedback_steps is not None:
        cfg.feedback.feedback_steps = int(args.feedback_steps)
    if args.safe_max_feedback_steps is not None:
        cfg.feedback.safe_max_feedback_steps = int(args.safe_max_feedback_steps)
    if args.convergence_eps is not None:
        cfg.feedback.convergence_eps = float(args.convergence_eps)
    if args.conv_loss_weight is not None:
        cfg.loss.conv_loss_weight = float(args.conv_loss_weight)
    if args.epochs is not None:
        cfg.train.base_epochs = int(args.epochs)
        cfg.train.expert_epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    if args.expert_lr is not None:
        cfg.train.expert_lr = float(args.expert_lr)
    if not cfg.feedback.use_self_feedback:
        cfg.feedback.feedback_steps = 1
    cfg.feedback.feedback_steps = max(1, int(cfg.feedback.feedback_steps))
    cfg.feedback.safe_max_feedback_steps = max(1, int(cfg.feedback.safe_max_feedback_steps))
    cfg.feedback.convergence_eps = max(0.0, float(cfg.feedback.convergence_eps))
    return cfg


def main():
    args = parse_args()
    cfg = apply_cli_overrides(FullConfig(), args)
    set_seed(cfg.train.seed)
    ensure_dir(cfg.data.save_dir)
    save_json(os.path.join(cfg.data.save_dir, "config.json"), asdict(cfg))

    print(f"Using device: {cfg.train.device}")
    print(f"Data path: {cfg.data.data_path}")
    print(f"Save dir: {cfg.data.save_dir}")
    prepared = load_and_prepare_data(cfg)

    base_train_loader, base_val_loader, _ = make_base_loaders(prepared, cfg)
    base_model = TrendPeriodBaseModel(time_dim=len(prepared.time_cols), cfg=cfg).to(cfg.train.device)
    base_log = train_base_model(base_model, base_train_loader, base_val_loader, cfg, cfg.data.save_dir)
    best_base_path = os.path.join(cfg.data.save_dir, "best_base_model.pth")
    base_model.load_state_dict(torch.load(best_base_path, map_location=cfg.train.device))
    if cfg.base.freeze_after_train:
        for p in base_model.parameters():
            p.requires_grad = False
    base_model.eval()

    print("Generating base prediction series for train/val/test splits...")
    base_train = predict_base_series(base_model, prepared.train, prepared.time_cols, cfg)
    base_val = predict_base_series(base_model, prepared.val, prepared.time_cols, cfg)
    base_test = predict_base_series(base_model, prepared.test, prepared.time_cols, cfg)

    train_ds, val_ds, test_ds, expert_train_loader, expert_val_loader, expert_test_loader = make_expert_loaders(
        prepared, base_train, base_val, base_test, cfg
    )
    sample = train_ds[0]
    pv_input_dim = sample["pv_hist_seq_base"].shape[-1]
    pv_seq_len = sample["pv_hist_seq_base"].shape[0] + 1
    load_input_dim = sample["load_features"].numel() + 1
    gate_base_dim = sample["gate_features_base"].numel()
    gate_input_dim = gate_base_dim + 1
    feedback_input_dim = gate_base_dim + 5

    refine_model = IterativeDualExpertNetLoadModel(
        base_model=base_model,
        pv_input_dim=pv_input_dim,
        pv_seq_len=pv_seq_len,
        load_input_dim=load_input_dim,
        gate_input_dim=gate_input_dim,
        cfg=cfg,
        feedback_input_dim=feedback_input_dim,
    ).to(cfg.train.device)
    refine_log = train_refinement_model(refine_model, expert_train_loader, expert_val_loader, cfg, cfg.data.save_dir)
    best_refine_path = os.path.join(cfg.data.save_dir, "best_refinement_model.pth")
    refine_model.load_state_dict(torch.load(best_refine_path, map_location=cfg.train.device))
    refine_model.eval()

    arrays = predict_refinement(refine_model, expert_test_loader, cfg)
    save_prediction_outputs(arrays, test_ds, prepared, cfg, cfg.data.save_dir)
    save_loss_curve(base_log, refine_log, cfg.data.save_dir)
    print("Done.")
    print(f"Outputs saved to: {cfg.data.save_dir}")


if __name__ == "__main__":
    main()
