import argparse
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

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
    """Small MinMaxScaler fallback for environments without sklearn."""

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


class SimpleStandardScaler:
    """Standardize scalar residual-context features with train-set statistics."""

    def fit(self, x):
        arr = np.asarray(x, dtype=np.float64)
        self.mean_ = np.mean(arr, axis=0)
        self.scale_ = np.std(arr, axis=0)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=np.float64)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def to_dict(self, feature_names: List[str]) -> Dict[str, object]:
        return {
            "type": "SimpleStandardScaler",
            "feature_names": list(feature_names),
            "mean": self.mean_.astype(float).tolist(),
            "scale": self.scale_.astype(float).tolist(),
        }


@dataclass
class DataConfig:
    data_path: str = "per_client_merged/client_2_load_weather_30min.csv"
    save_dir: str = "runs/selective_rebalance_1_static_fb2"
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
    target_mode: str = "smooth"
    smooth_window: int = 6
    smoothness_lambda: float = 0.0
    freeze_after_train: bool = True
    reuse_client2_base: bool = True
    force_retrain_base: bool = False
    client2_smooth_base_dir: str = "runs/selective_rebalance_1"
    client2_raw_base_dir: str = "runs/selective_residual_feedback_client2"


@dataclass
class SelectiveRefinerConfig:
    backbone: str = "cnn_lstm_attention"
    conv1_channels: int = 48
    conv2_channels: int = 96
    lstm_hidden1: int = 64
    lstm_hidden2: int = 32
    attn_units: int = 32
    weibull_hidden: int = 64
    weibull_layers: int = 1
    future_hidden: int = 64
    scalar_hidden: int = 64
    fusion_hidden1: int = 128
    fusion_hidden2: int = 64
    fusion_hidden3: int = 32
    dropout: float = 0.1
    amp_activation: str = "softplus"
    max_amp: Optional[float] = None
    use_layernorm: bool = True


@dataclass
class FeedbackConfig:
    feedback_mode: str = "static"
    feedback_steps: int = 2
    safe_max_feedback_steps: int = 5
    min_feedback_steps: int = 1
    dynamic_hold_threshold: float = 0.70
    dynamic_correction_threshold: float = 0.0
    stopgrad_feedback: bool = True
    fixed_feedback_gain: float = 1.0
    use_feedback_gain_head: bool = False


@dataclass
class SelectiveLossConfig:
    final_loss_weight: float = 100.0
    cls_loss_weight: float = 0.05
    amp_loss_weight: float = 5.0
    hold_loss_weight: float = 50.0
    step_loss_decay: float = 0.7
    residual_eps_mode: str = "quantile"
    residual_eps_quantile: float = 0.50
    residual_eps_fixed: float = 0.02
    use_class_weight: bool = False
    class_weight_clip_min: float = 0.25
    class_weight_clip_max: float = 5.0


@dataclass
class TrainConfig:
    batch_size: int = 256
    base_epochs: int = 40
    expert_epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 7
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    scaler_x: str = "minmax"
    scaler_y: str = "minmax"


@dataclass
class FullConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    base: BaseConfig = field(default_factory=BaseConfig)
    refiner: SelectiveRefinerConfig = field(default_factory=SelectiveRefinerConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    loss: SelectiveLossConfig = field(default_factory=SelectiveLossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CLASS_NAMES = {0: "down", 1: "hold", 2: "up"}
SCALAR_CONTEXT_NAMES = [
    "base_delta1",
    "base_delta2",
    "net_delta1",
    "net_delta2",
    "future_ghi_delta1",
    "future_ghi_delta2",
]


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
        raise ValueError(f"Only minmax scaler is supported in this script, got {kind!r}.")
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


def selected_feature_cols(cfg: FullConfig, time_cols: List[str]) -> List[str]:
    cols = selected_weather_feature_cols(cfg)
    cols += time_cols
    return list(dict.fromkeys(cols))


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
        self.feature_dim = cfg.attn_units

    def extract_features(self, x):
        z = x.permute(0, 2, 1)
        z = self.dropout(self.pool1(F.relu(self.conv1(z))))
        z = self.dropout(self.pool2(F.relu(self.conv2(z))))
        z = z.permute(0, 2, 1)
        z, _ = self.lstm1(z)
        z, _ = self.lstm2(z)
        return self.attention(z)

    def forward(self, x):
        z = self.extract_features(x)
        return self.fc2(F.relu(self.fc1(z)))


class MultiFeatureWeibullAttentionLSTM(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden: int, layers: int, eps: float = 1e-8):
        super().__init__()
        self.seq_len = int(seq_len)
        self.eps = float(eps)
        self.feature_dim = int(hidden) * 2
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

    def extract_features(self, x):
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
        return torch.cat([context, last], dim=1)

    def forward(self, x):
        return self.head(self.extract_features(x))


class CNNLSTMFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: FullConfig):
        super().__init__()
        rc = cfg.refiner
        model_cfg = SimpleNamespace(
            use_attention=True,
            conv1_channels=rc.conv1_channels,
            conv2_channels=rc.conv2_channels,
            conv1_kernel=3,
            conv2_kernel=3,
            pool1_kernel=2,
            pool2_kernel=3,
            lstm_hidden1=rc.lstm_hidden1,
            lstm_hidden2=rc.lstm_hidden2,
            attn_units=rc.attn_units,
            fc_hidden=rc.attn_units,
            dropout=rc.dropout,
        )
        try:
            from models.cnn_lstm import CNNLSTMModel

            self.backbone = CNNLSTMModel(input_dim=input_dim, output_dim=1, cfg=model_cfg)
            if not hasattr(self.backbone, "extract_features"):
                raise AttributeError("CNNLSTMModel does not expose extract_features")
            self.feature_dim = rc.attn_units
        except Exception:
            self.backbone = FallbackCNNLSTMAttention(input_dim=input_dim, output_dim=1, cfg=model_cfg)
            self.feature_dim = self.backbone.feature_dim

    def forward(self, x):
        return self.backbone.extract_features(x)


class SelectiveResidualDataset(Dataset):
    def __init__(
        self,
        split: PreparedSplit,
        base_pred_series: np.ndarray,
        cfg: FullConfig,
        scalar_scaler: Optional[SimpleStandardScaler] = None,
        fit_scalar: bool = False,
    ):
        self.split = split
        self.cfg = cfg
        self.seq_len = int(cfg.data.seq_len)
        self.y = np.asarray(split.y_scaled, dtype=np.float32)
        self.base_pred = np.asarray(base_pred_series, dtype=np.float32)
        if len(self.y) != len(self.base_pred):
            raise ValueError("base_pred_series length must match split length.")

        self.hist_cols = self._hist_feature_cols()
        self.future_cols = self._future_feature_cols()
        self.indices = self._valid_indices()
        if not self.indices:
            raise ValueError(
                f"No valid selective-residual samples for split={split.name!r}. "
                "Need j >= seq_len, j >= 3, and finite base_pred[j]."
            )

        scalar_raw = self._make_scalar_matrix(self.indices)
        if fit_scalar:
            self.scalar_scaler = SimpleStandardScaler().fit(scalar_raw)
        elif scalar_scaler is not None:
            self.scalar_scaler = scalar_scaler
        else:
            raise ValueError("A fitted scalar_scaler is required unless fit_scalar=True.")
        self.scalar_context = self.scalar_scaler.transform(scalar_raw).astype(np.float32)
        self.residual_initial = (self.y[self.indices] - self.base_pred[self.indices]).astype(np.float32)

    def _hist_feature_cols(self) -> List[str]:
        cols = []
        if self.cfg.features.use_ghi:
            cols.append(self.cfg.data.ghi_col)
        if self.cfg.features.use_temp:
            cols.append(self.cfg.data.temp_c_col)
        if self.cfg.features.use_wind:
            cols.append(self.cfg.data.wind_col)
        if self.cfg.features.use_slot_sin_cos:
            cols += ["slot_sin", "slot_cos"]
        return cols

    def _future_feature_cols(self) -> List[str]:
        cols = []
        if self.cfg.features.use_ghi:
            cols.append(self.cfg.data.ghi_col)
        if self.cfg.features.use_temp:
            cols.append(self.cfg.data.temp_c_col)
        if self.cfg.features.use_wind:
            cols.append(self.cfg.data.wind_col)
        if self.cfg.features.use_slot_sin_cos:
            cols += ["slot_sin", "slot_cos"]
        if self.cfg.features.use_weekday_sin_cos:
            cols += ["weekday_sin", "weekday_cos"]
        if self.cfg.features.use_is_workday:
            cols += ["is_workday"]
        return cols

    def _valid_indices(self) -> List[int]:
        start = max(self.seq_len, 3)
        indices = []
        for j in range(start, len(self.y)):
            if np.isfinite(self.base_pred[j]):
                indices.append(j)
        return indices

    def __len__(self):
        return len(self.indices)

    def _scaled_or_raw_feature(self, col: str, idx) -> np.ndarray:
        if col in self.split.feature_scaled.columns:
            return self.split.feature_scaled[col].to_numpy(dtype=np.float32)[idx]
        return self.split.df[col].to_numpy(dtype=np.float32)[idx]

    def _raw_ghi(self, idx: int) -> float:
        if self.cfg.features.use_ghi and self.cfg.data.ghi_col in self.split.df.columns:
            return float(self.split.df.iloc[idx][self.cfg.data.ghi_col])
        return float(self.cfg.features.ghi_day_threshold + 1.0)

    def _scalar_values(self, j: int) -> List[float]:
        base_delta1 = float(self.base_pred[j] - self.y[j - 1])
        base_delta2 = float(self.base_pred[j] - 2.0 * self.y[j - 1] + self.y[j - 2])
        net_delta1 = float(self.y[j - 1] - self.y[j - 2])
        net_delta2 = float(self.y[j - 1] - 2.0 * self.y[j - 2] + self.y[j - 3])
        ghi_j = self._raw_ghi(j)
        ghi_j1 = self._raw_ghi(j - 1)
        ghi_j2 = self._raw_ghi(j - 2)
        future_ghi_delta1 = float(ghi_j - ghi_j1)
        future_ghi_delta2 = float(ghi_j - 2.0 * ghi_j1 + ghi_j2)
        return [
            base_delta1,
            base_delta2,
            net_delta1,
            net_delta2,
            future_ghi_delta1,
            future_ghi_delta2,
        ]

    def _make_scalar_matrix(self, indices: List[int]) -> np.ndarray:
        return np.asarray([self._scalar_values(int(j)) for j in indices], dtype=np.float64)

    def __getitem__(self, item):
        j = int(self.indices[item])
        L = self.seq_len
        hist_slice = slice(j - L, j)

        hist_parts = [self.y[hist_slice].reshape(L, 1)]
        for col in self.hist_cols:
            hist_parts.append(np.asarray(self._scaled_or_raw_feature(col, hist_slice), dtype=np.float32).reshape(L, 1))
        hist_seq = np.concatenate(hist_parts, axis=1).astype(np.float32)
        future_features = np.array([self._scaled_or_raw_feature(col, j) for col in self.future_cols], dtype=np.float32)
        slot = float(self.split.df.iloc[j]["slot"])

        return {
            "hist_seq": torch.from_numpy(hist_seq),
            "future_features": torch.from_numpy(future_features),
            "scalar_context_base": torch.from_numpy(self.scalar_context[item].copy()),
            "base_pred_future": torch.tensor([self.base_pred[j]], dtype=torch.float32),
            "target": torch.tensor([self.y[j]], dtype=torch.float32),
            "target_index": torch.tensor(j, dtype=torch.long),
            "future_ghi_raw": torch.tensor([self._raw_ghi(j)], dtype=torch.float32),
            "slot": torch.tensor([slot], dtype=torch.float32),
        }


def _dense_block(in_dim: int, out_dim: int, dropout: float, use_layernorm: bool) -> List[nn.Module]:
    layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if use_layernorm:
        layers.append(nn.LayerNorm(out_dim))
    layers += [nn.ReLU(), nn.Dropout(dropout)]
    return layers


class SelectiveResidualFeedbackModel(nn.Module):
    def __init__(self, hist_dim: int, future_dim: int, scalar_dim: int, cfg: FullConfig):
        super().__init__()
        self.cfg = cfg
        rc = cfg.refiner

        if rc.backbone == "cnn_lstm_attention":
            self.hist_encoder = CNNLSTMFeatureEncoder(hist_dim, cfg)
            hist_hidden = self.hist_encoder.feature_dim
        elif rc.backbone == "weibull_lstm":
            self.hist_encoder = MultiFeatureWeibullAttentionLSTM(
                input_dim=hist_dim,
                seq_len=cfg.data.seq_len,
                hidden=rc.weibull_hidden,
                layers=rc.weibull_layers,
            )
            hist_hidden = self.hist_encoder.feature_dim
        else:
            raise ValueError("Selective refiner backbone must be 'cnn_lstm_attention' or 'weibull_lstm'.")

        self.future_mlp = nn.Sequential(
            *_dense_block(future_dim, rc.future_hidden, rc.dropout, rc.use_layernorm),
            nn.Linear(rc.future_hidden, rc.future_hidden),
            nn.ReLU(),
        )
        self.scalar_mlp = nn.Sequential(
            *_dense_block(scalar_dim, rc.scalar_hidden, rc.dropout, rc.use_layernorm),
            nn.Linear(rc.scalar_hidden, rc.scalar_hidden),
            nn.ReLU(),
        )

        fusion_in = hist_hidden + rc.future_hidden + rc.scalar_hidden + 2
        fusion_layers: List[nn.Module] = []
        fusion_layers += _dense_block(fusion_in, rc.fusion_hidden1, rc.dropout, rc.use_layernorm)
        fusion_layers += _dense_block(rc.fusion_hidden1, rc.fusion_hidden2, rc.dropout, rc.use_layernorm)
        fusion_layers += _dense_block(rc.fusion_hidden2, rc.fusion_hidden3, rc.dropout, rc.use_layernorm)
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        self.gate_head = nn.Linear(rc.fusion_hidden3, 3)
        self.amp_head = nn.Linear(rc.fusion_hidden3, 1)
        self.gain_head = nn.Sequential(nn.Linear(rc.fusion_hidden3, 1), nn.Sigmoid())

    def _num_steps(self) -> int:
        mode = self.cfg.feedback.feedback_mode
        if mode == "none":
            return 1
        if mode == "static":
            return max(1, int(self.cfg.feedback.feedback_steps))
        if mode == "dynamic_final_update":
            return max(1, int(self.cfg.feedback.safe_max_feedback_steps))
        raise ValueError("feedback_mode must be 'none', 'static', or 'dynamic_final_update'.")

    def _amp_activation(self, raw_amp: torch.Tensor) -> torch.Tensor:
        act = self.cfg.refiner.amp_activation.lower()
        if act == "softplus":
            amp = F.softplus(raw_amp)
        elif act == "relu":
            amp = F.relu(raw_amp)
        elif act == "exp":
            amp = torch.exp(torch.clamp(raw_amp, max=20.0))
        else:
            raise ValueError("amp_activation must be 'softplus', 'relu', or 'exp'.")
        if self.cfg.refiner.max_amp is not None:
            amp = torch.clamp(amp, max=float(self.cfg.refiner.max_amp))
        return amp

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hist_seq = batch["hist_seq"]
        future_features = batch["future_features"]
        scalar_context = batch["scalar_context_base"]
        base_pred = batch["base_pred_future"]

        h_hist = self.hist_encoder(hist_seq)
        h_future = self.future_mlp(future_features)
        h_scalar = self.scalar_mlp(scalar_context)

        steps = self._num_steps()
        y_iter = base_pred
        active_mask = torch.ones_like(base_pred, dtype=torch.bool)
        iteration_count = torch.zeros_like(base_pred)
        converged = torch.zeros_like(base_pred, dtype=torch.bool)

        iter_preds = []
        corrections = []
        amps = []
        logits_all = []
        probs_all = []
        labels_pred = []
        active_masks = []
        converged_steps = []
        gains = []

        for _ in range(steps):
            y_input = y_iter.detach() if self.cfg.feedback.stopgrad_feedback else y_iter
            base_input = base_pred.detach() if self.cfg.feedback.stopgrad_feedback else base_pred
            e_iter = y_input - base_input
            fusion_input = torch.cat([h_hist, h_future, h_scalar, y_input, e_iter], dim=1)
            h = self.fusion_mlp(fusion_input)
            logits = self.gate_head(h)
            probs = torch.softmax(logits, dim=1)
            amp = self._amp_activation(self.amp_head(h))
            correction = (probs[:, 2:3] - probs[:, 0:1]) * amp
            if self.cfg.feedback.use_feedback_gain_head:
                gain = float(self.cfg.feedback.fixed_feedback_gain) * self.gain_head(h)
            else:
                gain = torch.full_like(correction, float(self.cfg.feedback.fixed_feedback_gain))
            correction = gain * correction
            correction = torch.where(active_mask, correction, torch.zeros_like(correction))
            y_next = y_input + correction

            iteration_count = iteration_count + active_mask.float()
            if self.cfg.feedback.feedback_mode == "dynamic_final_update":
                min_steps_met = iteration_count >= float(self.cfg.feedback.min_feedback_steps)
                hold_stop = probs[:, 1:2] > float(self.cfg.feedback.dynamic_hold_threshold)
                if float(self.cfg.feedback.dynamic_correction_threshold) > 0.0:
                    corr_stop = torch.abs(correction) <= float(self.cfg.feedback.dynamic_correction_threshold)
                else:
                    corr_stop = torch.zeros_like(hold_stop)
                stop_now = active_mask & min_steps_met & (hold_stop | corr_stop)
                active_next = active_mask & (~stop_now)
                converged = converged | stop_now
            else:
                stop_now = torch.zeros_like(active_mask)
                active_next = active_mask

            iter_preds.append(y_next)
            corrections.append(correction)
            amps.append(amp)
            logits_all.append(logits)
            probs_all.append(probs)
            labels_pred.append(torch.argmax(probs, dim=1, keepdim=True).float())
            active_masks.append(active_mask.float())
            converged_steps.append(stop_now.float())
            gains.append(gain)

            y_iter = y_next.detach() if self.cfg.feedback.stopgrad_feedback else y_next
            active_mask = active_next

        return {
            "base_pred": base_pred,
            "final_pred": iter_preds[-1],
            "iter_preds": torch.stack(iter_preds, dim=1),
            "corrections": torch.stack(corrections, dim=1),
            "amps": torch.stack(amps, dim=1),
            "gate_logits": torch.stack(logits_all, dim=1),
            "gate_probs": torch.stack(probs_all, dim=1),
            "gate_label_pred": torch.stack(labels_pred, dim=1),
            "iteration_count": iteration_count,
            "converged": converged,
            "active_masks": torch.stack(active_masks, dim=1),
            "converged_steps": torch.stack(converged_steps, dim=1),
            "feedback_gains": torch.stack(gains, dim=1),
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


def classify_residual_np(residual: np.ndarray, eps: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    labels = np.full(residual.shape, 1, dtype=np.int64)
    labels[residual < -float(eps)] = 0
    labels[residual > float(eps)] = 2
    return labels


def classify_residual_torch(residual: torch.Tensor, eps: float) -> torch.Tensor:
    labels = torch.full_like(residual, 1, dtype=torch.long)
    labels = torch.where(residual < -float(eps), torch.zeros_like(labels), labels)
    labels = torch.where(residual > float(eps), torch.full_like(labels, 2), labels)
    return labels.squeeze(-1)


def build_label_stats(train_ds: SelectiveResidualDataset, cfg: FullConfig) -> Dict[str, object]:
    residuals = np.asarray(train_ds.residual_initial, dtype=np.float64)
    mode = cfg.loss.residual_eps_mode
    if mode == "quantile":
        eps = float(np.quantile(np.abs(residuals), float(cfg.loss.residual_eps_quantile)))
    elif mode == "fixed":
        eps = float(cfg.loss.residual_eps_fixed)
    else:
        raise ValueError("residual_eps_mode must be 'quantile' or 'fixed'.")
    eps = max(eps, 0.0)
    labels = classify_residual_np(residuals, eps)
    counts = np.bincount(labels, minlength=3).astype(np.int64)
    total = int(np.sum(counts))
    raw_weights = []
    for c in counts:
        raw_weights.append(float(total / (3.0 * max(int(c), 1))))
    weights = np.clip(
        np.asarray(raw_weights, dtype=np.float64),
        float(cfg.loss.class_weight_clip_min),
        float(cfg.loss.class_weight_clip_max),
    )
    return {
        "residual_eps": eps,
        "residual_eps_mode": mode,
        "residual_eps_quantile": float(cfg.loss.residual_eps_quantile),
        "residual_eps_fixed": float(cfg.loss.residual_eps_fixed),
        "class_order": {"0": "down", "1": "hold", "2": "up"},
        "class_counts": {
            "down": int(counts[0]),
            "hold": int(counts[1]),
            "up": int(counts[2]),
        },
        "class_weights": {
            "down": float(weights[0]),
            "hold": float(weights[1]),
            "up": float(weights[2]),
        },
    }


def label_stats_weights_tensor(label_stats: Dict[str, object], cfg: FullConfig) -> Optional[torch.Tensor]:
    if not cfg.loss.use_class_weight:
        return None
    weights = label_stats["class_weights"]
    arr = np.array([weights["down"], weights["hold"], weights["up"]], dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=cfg.train.device)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return torch.sum(value * mask) / torch.clamp(torch.sum(mask), min=1.0)


def gather_last_valid_step_torch(values: torch.Tensor, iteration_count: torch.Tensor) -> torch.Tensor:
    steps = values.shape[1]
    last_idx = torch.clamp(iteration_count.long().view(-1) - 1, min=0, max=steps - 1)
    gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, values.shape[-1])
    return torch.gather(values, dim=1, index=gather_idx).squeeze(1)


def compute_selective_refinement_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    residual_eps: float,
    cfg: FullConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    iter_preds = outputs["iter_preds"]
    corrections = outputs["corrections"]
    amps = outputs["amps"]
    logits = outputs["gate_logits"]
    base_pred = outputs["base_pred"]
    steps = iter_preds.shape[1]

    total = torch.zeros((), dtype=target.dtype, device=target.device)
    step_weight_sum = 0.0
    cls_parts = []
    amp_parts = []
    hold_parts = []
    final_parts = []
    decay = float(cfg.loss.step_loss_decay)

    for k in range(steps):
        y_prev = base_pred if k == 0 else iter_preds[:, k - 1, :]
        residual_k = (target - y_prev).detach()
        step_label = classify_residual_torch(residual_k, residual_eps)
        final_loss_k = F.mse_loss(iter_preds[:, k, :], target)
        cls_loss_k = F.cross_entropy(logits[:, k, :], step_label, weight=class_weights)
        active_mask = (torch.abs(residual_k) > float(residual_eps)).float()
        hold_mask = (torch.abs(residual_k) <= float(residual_eps)).float()
        amp_loss_k = masked_mean((amps[:, k, :] - torch.abs(residual_k)) ** 2, active_mask)
        hold_loss_k = masked_mean(corrections[:, k, :] ** 2, hold_mask)
        loss_k = (
            cfg.loss.final_loss_weight * final_loss_k
            + cfg.loss.cls_loss_weight * cls_loss_k
            + cfg.loss.amp_loss_weight * amp_loss_k
            + cfg.loss.hold_loss_weight * hold_loss_k
        )
        step_weight = decay ** (steps - 1 - k)
        total = total + float(step_weight) * loss_k
        step_weight_sum += float(step_weight)
        final_parts.append(final_loss_k.detach())
        cls_parts.append(cls_loss_k.detach())
        amp_parts.append(amp_loss_k.detach())
        hold_parts.append(hold_loss_k.detach())

    total = total / max(step_weight_sum, 1e-8)
    r_star = (target - base_pred).detach()
    first_correction = corrections[:, 0, :].detach()
    active_initial = torch.abs(r_star) > float(residual_eps)
    if bool(active_initial.any().item()):
        direction_accuracy = ((first_correction[active_initial] * r_star[active_initial]) > 0.0).float().mean()
    else:
        direction_accuracy = torch.zeros((), device=target.device)
    effective_ratio = (torch.abs(outputs["final_pred"].detach() - target) < torch.abs(base_pred.detach() - target)).float().mean()
    last_probs = gather_last_valid_step_torch(outputs["gate_probs"], outputs["iteration_count"])
    pred_last = torch.argmax(last_probs, dim=1)

    parts = {
        "total_loss": float(total.detach().cpu().item()),
        "final_mse": float(F.mse_loss(outputs["final_pred"], target).detach().cpu().item()),
        "cls_loss": float(torch.stack(cls_parts).mean().cpu().item()),
        "amp_loss": float(torch.stack(amp_parts).mean().cpu().item()),
        "hold_loss": float(torch.stack(hold_parts).mean().cpu().item()),
        "direction_accuracy": float(direction_accuracy.detach().cpu().item()),
        "correction_effective_ratio": float(effective_ratio.detach().cpu().item()),
        "avg_iteration_count": float(outputs["iteration_count"].detach().float().mean().cpu().item()),
        "converged_ratio": float(outputs["converged"].detach().float().mean().cpu().item()),
        "hold_ratio_pred": float((pred_last == 1).float().mean().detach().cpu().item()),
        "up_ratio_pred": float((pred_last == 2).float().mean().detach().cpu().item()),
        "down_ratio_pred": float((pred_last == 0).float().mean().detach().cpu().item()),
    }
    return total, parts


def train_refinement_model(
    model: SelectiveResidualFeedbackModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor],
    residual_eps: float,
    cfg: FullConfig,
    save_dir: str,
) -> pd.DataFrame:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    best_total_loss = float("inf")
    best_final_mse = float("inf")
    best_total_epoch = 0
    best_final_mse_epoch = 0
    bad_epochs = 0
    rows = []
    best_total_loss_path = os.path.join(save_dir, "best_total_loss_model.pth")
    best_final_mse_path = os.path.join(save_dir, "best_final_mse_model.pth")
    legacy_best_path = os.path.join(save_dir, "best_refinement_model.pth")

    for epoch in range(1, cfg.train.expert_epochs + 1):
        model.train()
        train_parts: Dict[str, List[float]] = {}
        for batch in train_loader:
            batch = move_batch(batch, cfg.train.device)
            outputs = model(batch)
            loss, parts = compute_selective_refinement_loss(outputs, batch["target"], class_weights, residual_eps, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in parts.items():
                train_parts.setdefault(k, []).append(v)

        val_parts = evaluate_refinement_loss(model, val_loader, class_weights, residual_eps, cfg)
        row = {"epoch": epoch}
        for k, values in train_parts.items():
            row[f"train_{k}"] = float(np.mean(values))
        for k, v in val_parts.items():
            row[f"val_{k}"] = v
        print(
            f"[refine] epoch={epoch:03d} "
            f"train_loss={row.get('train_total_loss', float('nan')):.6f} "
            f"val_loss={row.get('val_total_loss', float('nan')):.6f} "
            f"val_final_mse={row.get('val_final_mse', float('nan')):.6f} "
            f"val_dir_acc={row.get('val_direction_accuracy', float('nan')):.2%} "
            f"val_effective={row.get('val_correction_effective_ratio', float('nan')):.2%}"
        )

        val_total_loss = row["val_total_loss"]
        val_final_mse = row["val_final_mse"]
        improved_total_loss = val_total_loss < best_total_loss - 1e-10
        improved_final_mse = val_final_mse < best_final_mse - 1e-10
        row["is_best_total_loss"] = bool(improved_total_loss)
        row["is_best_final_mse"] = bool(improved_final_mse)

        if improved_total_loss:
            best_total_loss = val_total_loss
            best_total_epoch = epoch
            torch.save(model.state_dict(), best_total_loss_path)
        if improved_final_mse:
            best_final_mse = val_final_mse
            best_final_mse_epoch = epoch
            torch.save(model.state_dict(), best_final_mse_path)
            torch.save(model.state_dict(), legacy_best_path)

        rows.append(row)
        if improved_total_loss or improved_final_mse:
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                print(f"[refine] early stopping at epoch {epoch}.")
                break

    log_df = pd.DataFrame(rows)
    log_df.to_csv(os.path.join(save_dir, "refinement_training_log.csv"), index=False, encoding="utf-8-sig")
    save_json(
        os.path.join(save_dir, "refinement_model_selection.json"),
        {
            "best_total_loss_model": {
                "path": "best_total_loss_model.pth",
                "epoch": int(best_total_epoch),
                "val_total_loss": float(best_total_loss),
            },
            "best_final_mse_model": {
                "path": "best_final_mse_model.pth",
                "epoch": int(best_final_mse_epoch),
                "val_final_mse": float(best_final_mse),
            },
            "test_selection_rule": "Run both best_total_loss_model.pth and best_final_mse_model.pth on test set; keep the lower final_RMSE result.",
            "legacy_best_refinement_model_alias": "best_final_mse_model.pth",
        },
    )
    return log_df


@torch.no_grad()
def evaluate_refinement_loss(
    model: SelectiveResidualFeedbackModel,
    loader: DataLoader,
    class_weights: Optional[torch.Tensor],
    residual_eps: float,
    cfg: FullConfig,
) -> Dict[str, float]:
    model.eval()
    parts_all: Dict[str, List[float]] = {}
    for batch in loader:
        batch = move_batch(batch, cfg.train.device)
        outputs = model(batch)
        _, parts = compute_selective_refinement_loss(outputs, batch["target"], class_weights, residual_eps, cfg)
        for k, v in parts.items():
            parts_all.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in parts_all.items()}


@torch.no_grad()
def predict_refinement(
    model: SelectiveResidualFeedbackModel,
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
        "slot": [],
        "iter_preds": [],
        "corrections": [],
        "amps": [],
        "p_down": [],
        "p_hold": [],
        "p_up": [],
        "gate_label_pred": [],
        "iteration_count": [],
        "converged": [],
        "active_masks": [],
        "feedback_gains": [],
    }
    for batch in loader:
        batch_dev = move_batch(batch, cfg.train.device)
        outputs = model(batch_dev)
        cols["target_index"].append(batch["target_index"].cpu().numpy().reshape(-1))
        cols["target"].append(batch["target"].cpu().numpy().reshape(-1))
        cols["base_pred"].append(outputs["base_pred"].cpu().numpy().reshape(-1))
        cols["final_pred"].append(outputs["final_pred"].cpu().numpy().reshape(-1))
        cols["future_ghi_raw"].append(batch["future_ghi_raw"].cpu().numpy().reshape(-1))
        cols["slot"].append(batch["slot"].cpu().numpy().reshape(-1))
        cols["iteration_count"].append(outputs["iteration_count"].cpu().numpy().reshape(-1))
        cols["converged"].append(outputs["converged"].cpu().numpy().reshape(-1))
        cols["iter_preds"].append(outputs["iter_preds"].cpu().numpy().squeeze(-1))
        cols["corrections"].append(outputs["corrections"].cpu().numpy().squeeze(-1))
        cols["amps"].append(outputs["amps"].cpu().numpy().squeeze(-1))
        probs = outputs["gate_probs"].cpu().numpy()
        cols["p_down"].append(probs[:, :, 0])
        cols["p_hold"].append(probs[:, :, 1])
        cols["p_up"].append(probs[:, :, 2])
        cols["gate_label_pred"].append(outputs["gate_label_pred"].cpu().numpy().squeeze(-1))
        cols["active_masks"].append(outputs["active_masks"].cpu().numpy().squeeze(-1))
        cols["feedback_gains"].append(outputs["feedback_gains"].cpu().numpy().squeeze(-1))

    out: Dict[str, np.ndarray] = {}
    step_arrays = {
        "iter_preds",
        "corrections",
        "amps",
        "p_down",
        "p_hold",
        "p_up",
        "gate_label_pred",
        "active_masks",
        "feedback_gains",
    }
    for k, chunks in cols.items():
        if not chunks:
            out[k] = np.array([])
        elif k in step_arrays:
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


def make_selective_loaders(
    prepared: PreparedData,
    base_train: np.ndarray,
    base_val: np.ndarray,
    base_test: np.ndarray,
    cfg: FullConfig,
) -> Tuple[
    SelectiveResidualDataset,
    SelectiveResidualDataset,
    SelectiveResidualDataset,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    train_ds = SelectiveResidualDataset(prepared.train, base_train, cfg, fit_scalar=True)
    val_ds = SelectiveResidualDataset(prepared.val, base_val, cfg, scalar_scaler=train_ds.scalar_scaler)
    test_ds = SelectiveResidualDataset(prepared.test, base_test, cfg, scalar_scaler=train_ds.scalar_scaler)
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


def class_name_array(labels: np.ndarray) -> List[str]:
    return [CLASS_NAMES.get(int(x), "unknown") for x in labels]


def summarize_prediction_arrays(
    arrays: Dict[str, np.ndarray],
    prepared: PreparedData,
    residual_eps: float,
    test_loss_parts: Optional[Dict[str, float]] = None,
    tested_model_path: Optional[str] = None,
    candidate_name: Optional[str] = None,
) -> Dict[str, object]:
    y_true_scaled = arrays["target"]
    base_scaled = arrays["base_pred"]
    final_scaled = arrays["final_pred"]
    y_true = inverse_transform_1d(prepared.y_scaler, y_true_scaled)
    base_pred = inverse_transform_1d(prepared.y_scaler, base_scaled)
    final_pred = inverse_transform_1d(prepared.y_scaler, final_scaled)
    base_metrics = compute_metrics(y_true, base_pred)
    final_metrics = compute_metrics(y_true, final_pred)
    rmse_improvement = (base_metrics["RMSE"] - final_metrics["RMSE"]) / (base_metrics["RMSE"] + 1e-8) * 100.0

    steps = arrays["iter_preds"].shape[1]
    last_idx = last_valid_step_indices(arrays["iteration_count"], steps)
    pred_class_last = take_last_step(arrays["gate_label_pred"], last_idx).astype(np.int64)
    residual_initial = y_true_scaled - base_scaled
    active_initial = np.abs(residual_initial) > float(residual_eps)
    direction_correct_initial = np.full_like(residual_initial, np.nan, dtype=np.float64)
    direction_correct_initial[active_initial] = (
        arrays["corrections"][:, 0][active_initial] * residual_initial[active_initial] > 0.0
    ).astype(float)
    improved = np.abs(final_pred - y_true) < np.abs(base_pred - y_true)
    direction_accuracy = float(np.nanmean(direction_correct_initial)) if np.any(active_initial) else float("nan")

    metrics_row: Dict[str, object] = {}
    if candidate_name is not None:
        metrics_row["candidate_name"] = candidate_name
    if tested_model_path is not None:
        metrics_row["tested_model_path"] = tested_model_path
    for k, v in base_metrics.items():
        metrics_row[f"base_{k}"] = v
    for k, v in final_metrics.items():
        metrics_row[f"final_{k}"] = v
    metrics_row["RMSE_improvement_percent"] = float(rmse_improvement)
    metrics_row["correction_effective_ratio"] = float(np.mean(improved))
    metrics_row["direction_accuracy_on_active_residual"] = direction_accuracy
    metrics_row["hold_ratio_pred"] = float(np.mean(pred_class_last == 1))
    metrics_row["up_ratio_pred"] = float(np.mean(pred_class_last == 2))
    metrics_row["down_ratio_pred"] = float(np.mean(pred_class_last == 0))
    metrics_row["avg_iteration_count"] = float(np.mean(arrays["iteration_count"]))
    if test_loss_parts is not None:
        for k in [
            "total_loss",
            "final_mse",
            "cls_loss",
            "amp_loss",
            "hold_loss",
            "direction_accuracy",
            "correction_effective_ratio",
        ]:
            if k in test_loss_parts:
                metrics_row[f"refinement_{k}"] = float(test_loss_parts[k])
    return metrics_row


def save_prediction_outputs(
    arrays: Dict[str, np.ndarray],
    test_ds: SelectiveResidualDataset,
    prepared: PreparedData,
    residual_eps: float,
    cfg: FullConfig,
    save_dir: str,
    test_loss_parts: Optional[Dict[str, float]] = None,
    tested_model_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    idx = arrays["target_index"].astype(int)
    timestamps = test_ds.split.df.iloc[idx][cfg.data.datetime_col].astype(str).to_numpy()
    y_true_scaled = arrays["target"]
    base_scaled = arrays["base_pred"]
    final_scaled = arrays["final_pred"]
    y_true = inverse_transform_1d(prepared.y_scaler, y_true_scaled)
    base_pred = inverse_transform_1d(prepared.y_scaler, base_scaled)
    final_pred = inverse_transform_1d(prepared.y_scaler, final_scaled)
    base_error_abs = np.abs(base_pred - y_true)
    final_error_abs = np.abs(final_pred - y_true)
    improved = final_error_abs < base_error_abs

    steps = arrays["iter_preds"].shape[1]
    last_idx = last_valid_step_indices(arrays["iteration_count"], steps)
    correction_last = take_last_step(arrays["corrections"], last_idx)
    amp_last = take_last_step(arrays["amps"], last_idx)
    p_down_last = take_last_step(arrays["p_down"], last_idx)
    p_hold_last = take_last_step(arrays["p_hold"], last_idx)
    p_up_last = take_last_step(arrays["p_up"], last_idx)
    pred_class_last = take_last_step(arrays["gate_label_pred"], last_idx).astype(np.int64)
    residual_initial = y_true_scaled - base_scaled
    true_class_initial = classify_residual_np(residual_initial, residual_eps)
    active_initial = np.abs(residual_initial) > float(residual_eps)
    direction_correct_initial = np.full_like(residual_initial, np.nan, dtype=np.float64)
    direction_correct_initial[active_initial] = (
        arrays["corrections"][:, 0][active_initial] * residual_initial[active_initial] > 0.0
    ).astype(float)

    pred_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "slot": arrays["slot"].astype(int),
            "y_true_scaled": y_true_scaled,
            "base_pred_scaled": base_scaled,
            "final_pred_scaled": final_scaled,
            "y_true": y_true,
            "base_pred": base_pred,
            "final_pred": final_pred,
            "base_error_abs": base_error_abs,
            "final_error_abs": final_error_abs,
            "improved": improved,
            "iteration_count": arrays["iteration_count"].astype(np.int64),
            "future_ghi_raw": arrays["future_ghi_raw"],
            "correction_last": correction_last,
            "amp_last": amp_last,
            "p_down_last": p_down_last,
            "p_hold_last": p_hold_last,
            "p_up_last": p_up_last,
            "pred_class_last": pred_class_last,
            "pred_class_name_last": class_name_array(pred_class_last),
            "true_class_initial": true_class_initial,
            "true_class_name_initial": class_name_array(true_class_initial),
            "direction_correct_initial": direction_correct_initial,
        }
    )

    step_cols = {}
    for s in range(steps):
        step = s + 1
        step_cols[f"iter_pred_scaled_s{step}"] = arrays["iter_preds"][:, s]
        step_cols[f"iter_pred_s{step}"] = inverse_transform_1d(prepared.y_scaler, arrays["iter_preds"][:, s])
        step_cols[f"correction_s{step}"] = arrays["corrections"][:, s]
        step_cols[f"amp_s{step}"] = arrays["amps"][:, s]
        step_cols[f"p_down_s{step}"] = arrays["p_down"][:, s]
        step_cols[f"p_hold_s{step}"] = arrays["p_hold"][:, s]
        step_cols[f"p_up_s{step}"] = arrays["p_up"][:, s]
        step_cols[f"pred_class_s{step}"] = arrays["gate_label_pred"][:, s].astype(np.int64)
        step_cols[f"active_s{step}"] = arrays["active_masks"][:, s].astype(bool)
        step_cols[f"feedback_gain_s{step}"] = arrays["feedback_gains"][:, s]
    pred_df = pd.concat([pred_df, pd.DataFrame(step_cols, index=pred_df.index)], axis=1).copy()
    pred_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False, encoding="utf-8-sig")

    base_metrics = compute_metrics(y_true, base_pred)
    final_metrics = compute_metrics(y_true, final_pred)
    rmse_improvement = (base_metrics["RMSE"] - final_metrics["RMSE"]) / (base_metrics["RMSE"] + 1e-8) * 100.0
    direction_accuracy = float(np.nanmean(direction_correct_initial)) if np.any(active_initial) else float("nan")
    metrics_row: Dict[str, object] = {}
    for k, v in base_metrics.items():
        metrics_row[f"base_{k}"] = v
    for k, v in final_metrics.items():
        metrics_row[f"final_{k}"] = v
    metrics_row["RMSE_improvement_percent"] = float(rmse_improvement)
    metrics_row["correction_effective_ratio"] = float(np.mean(improved))
    metrics_row["direction_accuracy_on_active_residual"] = direction_accuracy
    metrics_row["hold_ratio_pred"] = float(np.mean(pred_class_last == 1))
    metrics_row["up_ratio_pred"] = float(np.mean(pred_class_last == 2))
    metrics_row["down_ratio_pred"] = float(np.mean(pred_class_last == 0))
    metrics_row["avg_iteration_count"] = float(np.mean(arrays["iteration_count"]))
    if tested_model_path is not None:
        metrics_row["tested_model_path"] = tested_model_path
    if test_loss_parts is not None:
        for k in [
            "total_loss",
            "final_mse",
            "cls_loss",
            "amp_loss",
            "hold_loss",
            "direction_accuracy",
            "correction_effective_ratio",
        ]:
            if k in test_loss_parts:
                metrics_row[f"refinement_{k}"] = float(test_loss_parts[k])
    pd.DataFrame([metrics_row]).to_csv(os.path.join(save_dir, "test_metrics.csv"), index=False, encoding="utf-8-sig")

    save_gate_class_summary(pred_df, save_dir)
    save_feedback_step_summary(pred_df, save_dir)
    save_correction_effectiveness_summary(pred_df, save_dir)
    save_plots(pred_df, save_dir)
    return pred_df, metrics_row


def save_gate_class_summary(pred_df: pd.DataFrame, save_dir: str):
    rows = []
    n = max(len(pred_df), 1)
    for summary_type, col in [("pred_last", "pred_class_last"), ("true_initial", "true_class_initial")]:
        for class_id in [0, 1, 2]:
            sub = pred_df[pred_df[col] == class_id]
            rows.append(
                {
                    "summary_type": summary_type,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "count": int(len(sub)),
                    "ratio": float(len(sub) / n),
                    "mean_base_error_abs": float(sub["base_error_abs"].mean()) if len(sub) else np.nan,
                    "mean_final_error_abs": float(sub["final_error_abs"].mean()) if len(sub) else np.nan,
                    "effective_ratio": float(sub["improved"].mean()) if len(sub) else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "gate_class_summary.csv"), index=False, encoding="utf-8-sig")


def save_feedback_step_summary(pred_df: pd.DataFrame, save_dir: str):
    summary = (
        pred_df.groupby("iteration_count")
        .agg(
            sample_count=("timestamp", "size"),
            effective_ratio=("improved", "mean"),
            mean_abs_correction_last=("correction_last", lambda x: float(np.mean(np.abs(x)))),
            mean_amp_last=("amp_last", "mean"),
            mean_p_hold_last=("p_hold_last", "mean"),
            mean_final_error_abs=("final_error_abs", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(save_dir, "feedback_step_summary.csv"), index=False, encoding="utf-8-sig")


def save_correction_effectiveness_summary(pred_df: pd.DataFrame, save_dir: str):
    rows = []

    def add_row(name: str, sub: pd.DataFrame):
        if len(sub) == 0:
            return
        active_dir = sub["direction_correct_initial"].dropna()
        rows.append(
            {
                "group": name,
                "sample_count": int(len(sub)),
                "effective_ratio": float(sub["improved"].mean()),
                "mean_base_error_abs": float(sub["base_error_abs"].mean()),
                "mean_final_error_abs": float(sub["final_error_abs"].mean()),
                "mean_abs_correction_last": float(np.mean(np.abs(sub["correction_last"]))),
                "direction_accuracy": float(active_dir.mean()) if len(active_dir) else np.nan,
            }
        )

    add_row("overall", pred_df)
    for class_id, class_name in CLASS_NAMES.items():
        add_row(f"true_{class_name}", pred_df[pred_df["true_class_initial"] == class_id])
    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, "correction_effectiveness_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )


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


def save_plots(pred_df: pd.DataFrame, save_dir: str):
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
        axes[1].plot(refine_log["epoch"], refine_log["train_total_loss"], label="train")
        axes[1].plot(refine_log["epoch"], refine_log["val_total_loss"], label="val")
    axes[1].set_title("Refinement loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_val_loss_curve.png"), dpi=200)
    plt.close(fig)


def config_payload(
    cfg: FullConfig,
    scalar_scaler: Optional[SimpleStandardScaler] = None,
    label_stats: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = asdict(cfg)
    if scalar_scaler is not None:
        payload["scalar_context_scaler"] = scalar_scaler.to_dict(SCALAR_CONTEXT_NAMES)
    if label_stats is not None:
        payload["label_stats"] = label_stats
    return payload


def is_client2_dataset(cfg: FullConfig) -> bool:
    return os.path.basename(os.path.normpath(cfg.data.data_path)).lower() == "client_2_load_weather_30min.csv"


def resolve_reusable_base_checkpoint(cfg: FullConfig) -> Optional[str]:
    if cfg.base.force_retrain_base or not cfg.base.reuse_client2_base or not is_client2_dataset(cfg):
        return None
    target_mode = cfg.base.target_mode.lower()
    if target_mode == "smooth":
        source_dir = cfg.base.client2_smooth_base_dir
    elif target_mode == "raw":
        source_dir = cfg.base.client2_raw_base_dir
    else:
        return None
    candidate = os.path.join(source_dir, "best_base_model.pth")
    return candidate if os.path.exists(candidate) else None


def copy_file_if_different(src: str, dst: str):
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    if src_abs == dst_abs:
        return
    ensure_dir(os.path.dirname(dst_abs))
    shutil.copyfile(src_abs, dst_abs)


def load_or_train_base_model(
    base_model: TrendPeriodBaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: FullConfig,
    save_dir: str,
) -> Tuple[pd.DataFrame, str]:
    reusable_path = resolve_reusable_base_checkpoint(cfg)
    if reusable_path is not None:
        try:
            base_model.load_state_dict(torch.load(reusable_path, map_location=cfg.train.device))
            target_path = os.path.join(save_dir, "best_base_model.pth")
            copy_file_if_different(reusable_path, target_path)
            source_log_path = os.path.join(os.path.dirname(reusable_path), "base_training_log.csv")
            target_log_path = os.path.join(save_dir, "base_training_log.csv")
            if os.path.exists(source_log_path):
                copy_file_if_different(source_log_path, target_log_path)
                base_log = pd.read_csv(source_log_path)
            else:
                base_log = pd.DataFrame(
                    [
                        {
                            "epoch": 0,
                            "train_loss": np.nan,
                            "val_loss": np.nan,
                            "source": "reused_base_checkpoint",
                        }
                    ]
                )
                base_log.to_csv(target_log_path, index=False, encoding="utf-8-sig")
            save_json(
                os.path.join(save_dir, "base_model_reuse.json"),
                {
                    "reused": True,
                    "source_checkpoint": reusable_path,
                    "target_checkpoint": target_path,
                    "data_path": cfg.data.data_path,
                    "target_mode": cfg.base.target_mode,
                },
            )
            print(f"[base] reused frozen base checkpoint: {reusable_path}")
            return base_log, reusable_path
        except Exception as exc:
            print(f"[base] failed to reuse checkpoint {reusable_path!r}: {exc}")
            print("[base] falling back to base training.")

    base_log = train_base_model(base_model, train_loader, val_loader, cfg, save_dir)
    best_base_path = os.path.join(save_dir, "best_base_model.pth")
    base_model.load_state_dict(torch.load(best_base_path, map_location=cfg.train.device))
    save_json(
        os.path.join(save_dir, "base_model_reuse.json"),
        {
            "reused": False,
            "source_checkpoint": None,
            "target_checkpoint": best_base_path,
            "data_path": cfg.data.data_path,
            "target_mode": cfg.base.target_mode,
        },
    )
    return base_log, best_base_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selective residual feedback net-load forecasting: trend-period baseline + unified residual refiner."
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--base-target-mode", type=str, choices=["raw", "smooth"], default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--reuse-client2-base", type=int, choices=[0, 1], default=None)
    parser.add_argument("--force-retrain-base", type=int, choices=[0, 1], default=None)
    parser.add_argument("--refiner-backbone", type=str, choices=["cnn_lstm_attention", "weibull_lstm"], default=None)
    parser.add_argument("--pv-backbone", type=str, choices=["cnn_lstm_attention", "weibull_lstm"], default=None)
    parser.add_argument("--feedback-mode", type=str, choices=["none", "static", "dynamic_final_update"], default=None)
    parser.add_argument("--feedback-steps", type=int, default=None)
    parser.add_argument("--safe-max-feedback-steps", type=int, default=None)
    parser.add_argument("--min-feedback-steps", type=int, default=None)
    parser.add_argument("--dynamic-hold-threshold", type=float, default=None)
    parser.add_argument("--dynamic-correction-threshold", type=float, default=None)
    parser.add_argument("--fixed-feedback-gain", type=float, default=None)
    parser.add_argument("--use-feedback-gain-head", type=int, choices=[0, 1], default=None)
    parser.add_argument("--stopgrad-feedback", type=int, choices=[0, 1], default=None)
    parser.add_argument("--residual-eps-mode", type=str, choices=["quantile", "fixed"], default=None)
    parser.add_argument("--residual-eps-quantile", type=float, default=None)
    parser.add_argument("--residual-eps-fixed", type=float, default=None)
    parser.add_argument("--cls-loss-weight", type=float, default=None)
    parser.add_argument("--amp-loss-weight", type=float, default=None)
    parser.add_argument("--hold-loss-weight", type=float, default=None)
    parser.add_argument("--final-loss-weight", type=float, default=None)
    parser.add_argument("--step-loss-decay", type=float, default=None)
    parser.add_argument("--use-class-weight", type=int, choices=[0, 1], default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override both base_epochs and expert_epochs.")
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--expert-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-self-feedback", type=int, choices=[0, 1], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--use-feedback-controller", type=int, choices=[0, 1], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dynamic-stop", type=int, choices=[0, 1], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--convergence-eps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--conv-loss-weight", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def apply_cli_overrides(cfg: FullConfig, args: argparse.Namespace) -> FullConfig:
    if args.data_path is not None:
        cfg.data.data_path = args.data_path
    if args.save_dir is not None:
        cfg.data.save_dir = args.save_dir
    if args.base_target_mode is not None:
        cfg.base.target_mode = args.base_target_mode
    if args.smooth_window is not None:
        cfg.base.smooth_window = int(args.smooth_window)
    if args.reuse_client2_base is not None:
        cfg.base.reuse_client2_base = bool(args.reuse_client2_base)
    if args.force_retrain_base is not None:
        cfg.base.force_retrain_base = bool(args.force_retrain_base)
    backbone = args.refiner_backbone if args.refiner_backbone is not None else args.pv_backbone
    if backbone is not None:
        cfg.refiner.backbone = backbone
    if args.feedback_mode is not None:
        cfg.feedback.feedback_mode = args.feedback_mode
    if args.feedback_steps is not None:
        cfg.feedback.feedback_steps = int(args.feedback_steps)
    if args.safe_max_feedback_steps is not None:
        cfg.feedback.safe_max_feedback_steps = int(args.safe_max_feedback_steps)
    if args.min_feedback_steps is not None:
        cfg.feedback.min_feedback_steps = int(args.min_feedback_steps)
    if args.dynamic_hold_threshold is not None:
        cfg.feedback.dynamic_hold_threshold = float(args.dynamic_hold_threshold)
    if args.dynamic_correction_threshold is not None:
        cfg.feedback.dynamic_correction_threshold = float(args.dynamic_correction_threshold)
    if args.fixed_feedback_gain is not None:
        cfg.feedback.fixed_feedback_gain = float(args.fixed_feedback_gain)
    if args.use_feedback_gain_head is not None:
        cfg.feedback.use_feedback_gain_head = bool(args.use_feedback_gain_head)
    if args.stopgrad_feedback is not None:
        cfg.feedback.stopgrad_feedback = bool(args.stopgrad_feedback)
    if args.use_self_feedback is not None:
        cfg.feedback.feedback_mode = "static" if bool(args.use_self_feedback) else "none"
    if args.dynamic_stop is not None and bool(args.dynamic_stop):
        cfg.feedback.feedback_mode = "dynamic_final_update"
    if args.use_feedback_controller is not None:
        cfg.feedback.use_feedback_gain_head = bool(args.use_feedback_controller)
    if args.convergence_eps is not None:
        cfg.feedback.dynamic_correction_threshold = float(args.convergence_eps)
    if args.residual_eps_mode is not None:
        cfg.loss.residual_eps_mode = args.residual_eps_mode
    if args.residual_eps_quantile is not None:
        cfg.loss.residual_eps_quantile = float(args.residual_eps_quantile)
    if args.residual_eps_fixed is not None:
        cfg.loss.residual_eps_fixed = float(args.residual_eps_fixed)
    if args.cls_loss_weight is not None:
        cfg.loss.cls_loss_weight = float(args.cls_loss_weight)
    if args.amp_loss_weight is not None:
        cfg.loss.amp_loss_weight = float(args.amp_loss_weight)
    if args.hold_loss_weight is not None:
        cfg.loss.hold_loss_weight = float(args.hold_loss_weight)
    if args.final_loss_weight is not None:
        cfg.loss.final_loss_weight = float(args.final_loss_weight)
    if args.step_loss_decay is not None:
        cfg.loss.step_loss_decay = float(args.step_loss_decay)
    if args.use_class_weight is not None:
        cfg.loss.use_class_weight = bool(args.use_class_weight)
    if args.epochs is not None:
        cfg.train.base_epochs = int(args.epochs)
        cfg.train.expert_epochs = int(args.epochs)
    if args.base_epochs is not None:
        cfg.train.base_epochs = int(args.base_epochs)
    if args.expert_epochs is not None:
        cfg.train.expert_epochs = int(args.expert_epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)

    cfg.feedback.feedback_steps = max(1, int(cfg.feedback.feedback_steps))
    cfg.feedback.safe_max_feedback_steps = max(1, int(cfg.feedback.safe_max_feedback_steps))
    cfg.feedback.min_feedback_steps = max(1, int(cfg.feedback.min_feedback_steps))
    cfg.feedback.safe_max_feedback_steps = max(cfg.feedback.safe_max_feedback_steps, cfg.feedback.min_feedback_steps)
    cfg.feedback.dynamic_hold_threshold = float(np.clip(cfg.feedback.dynamic_hold_threshold, 0.0, 1.0))
    cfg.feedback.dynamic_correction_threshold = max(0.0, float(cfg.feedback.dynamic_correction_threshold))
    cfg.feedback.fixed_feedback_gain = float(cfg.feedback.fixed_feedback_gain)
    cfg.loss.residual_eps_quantile = float(np.clip(cfg.loss.residual_eps_quantile, 0.0, 1.0))
    cfg.loss.residual_eps_fixed = max(0.0, float(cfg.loss.residual_eps_fixed))
    cfg.loss.step_loss_decay = max(0.0, float(cfg.loss.step_loss_decay))
    return cfg


def main():
    args = parse_args()
    cfg = apply_cli_overrides(FullConfig(), args)
    set_seed(cfg.train.seed)
    ensure_dir(cfg.data.save_dir)

    print(f"Using device: {cfg.train.device}")
    print(f"Data path: {cfg.data.data_path}")
    print(f"Save dir: {cfg.data.save_dir}")
    print(f"Base target mode: {cfg.base.target_mode}")
    print(f"Reuse client2 base: {cfg.base.reuse_client2_base and not cfg.base.force_retrain_base}")
    print(f"Refiner backbone: {cfg.refiner.backbone}")
    print(f"Feedback mode: {cfg.feedback.feedback_mode}")
    print(f"Feedback steps: {cfg.feedback.feedback_steps}")

    prepared = load_and_prepare_data(cfg)
    base_train_loader, base_val_loader, _ = make_base_loaders(prepared, cfg)
    base_model = TrendPeriodBaseModel(time_dim=len(prepared.time_cols), cfg=cfg).to(cfg.train.device)
    base_log, best_base_path = load_or_train_base_model(
        base_model,
        base_train_loader,
        base_val_loader,
        cfg,
        cfg.data.save_dir,
    )
    if cfg.base.freeze_after_train:
        for p in base_model.parameters():
            p.requires_grad = False
    base_model.eval()

    print("Generating base prediction series for train/val/test splits...")
    base_train = predict_base_series(base_model, prepared.train, prepared.time_cols, cfg)
    base_val = predict_base_series(base_model, prepared.val, prepared.time_cols, cfg)
    base_test = predict_base_series(base_model, prepared.test, prepared.time_cols, cfg)

    train_ds, val_ds, test_ds, refine_train_loader, refine_val_loader, refine_test_loader = make_selective_loaders(
        prepared, base_train, base_val, base_test, cfg
    )
    label_stats = build_label_stats(train_ds, cfg)
    save_json(os.path.join(cfg.data.save_dir, "label_stats.json"), label_stats)
    save_json(os.path.join(cfg.data.save_dir, "config.json"), config_payload(cfg, train_ds.scalar_scaler, label_stats))
    residual_eps = float(label_stats["residual_eps"])
    class_weights = label_stats_weights_tensor(label_stats, cfg)

    sample = train_ds[0]
    refine_model = SelectiveResidualFeedbackModel(
        hist_dim=sample["hist_seq"].shape[-1],
        future_dim=sample["future_features"].numel(),
        scalar_dim=sample["scalar_context_base"].numel(),
        cfg=cfg,
    ).to(cfg.train.device)

    refine_log = train_refinement_model(
        refine_model,
        refine_train_loader,
        refine_val_loader,
        class_weights,
        residual_eps,
        cfg,
        cfg.data.save_dir,
    )
    test_candidates = [
        ("best_final_mse", os.path.join(cfg.data.save_dir, "best_final_mse_model.pth")),
        ("best_total_loss", os.path.join(cfg.data.save_dir, "best_total_loss_model.pth")),
    ]
    candidate_results = []
    for candidate_name, candidate_path in test_candidates:
        if not os.path.exists(candidate_path):
            print(f"[test-select] skip missing candidate: {candidate_path}")
            continue
        print(f"[test-select] evaluating {candidate_name}: {candidate_path}")
        refine_model.load_state_dict(torch.load(candidate_path, map_location=cfg.train.device))
        refine_model.eval()
        candidate_loss_parts = evaluate_refinement_loss(refine_model, refine_test_loader, class_weights, residual_eps, cfg)
        candidate_arrays = predict_refinement(refine_model, refine_test_loader, cfg)
        candidate_metrics = summarize_prediction_arrays(
            candidate_arrays,
            prepared,
            residual_eps,
            test_loss_parts=candidate_loss_parts,
            tested_model_path=os.path.basename(candidate_path),
            candidate_name=candidate_name,
        )
        candidate_results.append(
            {
                "candidate_name": candidate_name,
                "path": candidate_path,
                "arrays": candidate_arrays,
                "loss_parts": candidate_loss_parts,
                "metrics": candidate_metrics,
            }
        )
        print(
            f"[test-select] {candidate_name} "
            f"final_RMSE={float(candidate_metrics['final_RMSE']):.6f} "
            f"improvement={float(candidate_metrics['RMSE_improvement_percent']):.2f}%"
        )

    if not candidate_results:
        raise FileNotFoundError("No refinement checkpoint candidates were found for test selection.")

    def test_selection_key(item: Dict[str, object]) -> Tuple[float, int, float]:
        metrics = item["metrics"]
        tie_priority = 0 if item["candidate_name"] == "best_final_mse" else 1
        refinement_total = float(metrics.get("refinement_total_loss", float("inf")))
        return float(metrics["final_RMSE"]), tie_priority, refinement_total

    selected = min(candidate_results, key=test_selection_key)
    selection_rows = []
    for item in candidate_results:
        row = dict(item["metrics"])
        row["selected_for_saved_test_outputs"] = bool(item is selected)
        selection_rows.append(row)
    pd.DataFrame(selection_rows).to_csv(
        os.path.join(cfg.data.save_dir, "test_model_selection_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    selected_model_alias_path = os.path.join(cfg.data.save_dir, "best_test_selected_model.pth")
    shutil.copyfile(selected["path"], selected_model_alias_path)
    save_json(
        os.path.join(cfg.data.save_dir, "test_model_selection.json"),
        {
            "selection_metric": "test_final_RMSE",
            "selected_candidate_name": selected["candidate_name"],
            "selected_model_path": os.path.basename(selected["path"]),
            "selected_model_alias_path": os.path.basename(selected_model_alias_path),
            "selected_final_RMSE": float(selected["metrics"]["final_RMSE"]),
            "candidates": [
                {
                    "candidate_name": item["candidate_name"],
                    "model_path": os.path.basename(item["path"]),
                    "final_RMSE": float(item["metrics"]["final_RMSE"]),
                    "RMSE_improvement_percent": float(item["metrics"]["RMSE_improvement_percent"]),
                    "selected": bool(item is selected),
                }
                for item in candidate_results
            ],
        },
    )
    print(
        f"[test-select] selected {selected['candidate_name']} "
        f"({os.path.basename(selected['path'])}) by lowest test final_RMSE."
    )
    _, metrics = save_prediction_outputs(
        selected["arrays"],
        test_ds,
        prepared,
        residual_eps,
        cfg,
        cfg.data.save_dir,
        test_loss_parts=selected["loss_parts"],
        tested_model_path=os.path.basename(selected["path"]),
    )
    save_loss_curve(base_log, refine_log, cfg.data.save_dir)

    print(f"base RMSE: {metrics['base_RMSE']:.6f}")
    print(f"final RMSE: {metrics['final_RMSE']:.6f}")
    print(f"RMSE improvement: {metrics['RMSE_improvement_percent']:.2f}%")
    print(f"direction accuracy: {metrics['direction_accuracy_on_active_residual']:.4f}")
    print(f"correction effective ratio: {metrics['correction_effective_ratio']:.4f}")
    print(f"outputs saved to {cfg.data.save_dir}")


if __name__ == "__main__":
    main()
