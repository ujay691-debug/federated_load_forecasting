import argparse
import copy
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
    save_dir: str = "runs/selective_residual_feedback_client2-smooth"
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
    feedback_mode: str = "none"
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
    base_epochs: int = 60
    expert_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    scaler_x: str = "minmax"
    scaler_y: str = "minmax"


@dataclass
class FederatedSelectiveConfig:
    client_files: List[str] = field(
        default_factory=lambda: [
            os.path.join("per_client_merged", f"client_{i}_load_weather_30min.csv") for i in range(1, 10)
        ]
    )
    save_dir: str = "runs/federated_selective_residual"
    rounds: int = 20
    client_fraction: float = 1.0
    base_pretrain_epochs: int = 20
    local_epochs: int = 3
    final_local_finetune_epochs: int = 10
    final_local_train_base: Optional[bool] = None
    final_local_train_refiner: Optional[bool] = None
    freeze_base_before_final_local: bool = False
    aggregation_scope: str = "base_only"
    aggregation_weight: str = "num_samples"
    personalized_modules: List[str] = field(
        default_factory=lambda: ["future_mlp", "scalar_mlp", "fusion_mlp", "gate_head", "amp_head", "gain_head"]
    )
    share_base_model: bool = True
    share_hist_encoder: bool = False
    share_future_mlp: bool = False
    share_scalar_mlp: bool = False
    share_fusion_mlp: bool = False
    share_gate_head: bool = False
    share_amp_head: bool = False
    share_gain_head: bool = False
    freeze_base_after_pretrain: bool = False
    train_base_in_federated_rounds: bool = True
    train_refiner_in_federated_rounds: bool = True
    rebuild_label_every_round: bool = True
    evaluate_every_round: bool = True
    early_stop_patience: int = 0
    best_metric: str = "avg_client_val_final_RMSE"
    save_personalized_client_models: bool = True
    save_global_shared_state: bool = True


@dataclass
class FullConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    base: BaseConfig = field(default_factory=BaseConfig)
    refiner: SelectiveRefinerConfig = field(default_factory=SelectiveRefinerConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    loss: SelectiveLossConfig = field(default_factory=SelectiveLossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    federated: FederatedSelectiveConfig = field(default_factory=FederatedSelectiveConfig)


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


class FederatedSelectiveNetLoadModel(nn.Module):
    def __init__(self, base_model: TrendPeriodBaseModel, refiner: SelectiveResidualFeedbackModel):
        super().__init__()
        self.base_model = base_model
        self.refiner = refiner

    def forward(
        self,
        net_seq: torch.Tensor,
        time_seq: torch.Tensor,
        time_future: torch.Tensor,
        hist_seq: torch.Tensor,
        future_features: torch.Tensor,
        scalar_context_base: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_pred = self.base_model(net_seq, time_seq, time_future)
        refine_batch = {
            "hist_seq": hist_seq,
            "future_features": future_features,
            "scalar_context_base": scalar_context_base,
            "base_pred_future": base_pred,
        }
        return self.refiner(refine_batch)


def clone_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def apply_aggregation_scope(cfg: FullConfig) -> FullConfig:
    scope = cfg.federated.aggregation_scope
    valid = {"none", "base_only", "base_and_encoder", "encoder_only", "all_shared"}
    if scope not in valid:
        raise ValueError(f"Unknown aggregation_scope={scope!r}. Valid choices: {sorted(valid)}")

    fed = cfg.federated
    fed.share_base_model = False
    fed.share_hist_encoder = False
    fed.share_future_mlp = False
    fed.share_scalar_mlp = False
    fed.share_fusion_mlp = False
    fed.share_gate_head = False
    fed.share_amp_head = False
    fed.share_gain_head = False

    if scope == "base_only":
        fed.share_base_model = True
    elif scope == "base_and_encoder":
        fed.share_base_model = True
        fed.share_hist_encoder = True
    elif scope == "encoder_only":
        fed.share_hist_encoder = True
    elif scope == "all_shared":
        fed.share_base_model = True
        fed.share_hist_encoder = True
        fed.share_future_mlp = True
        fed.share_scalar_mlp = True
        fed.share_fusion_mlp = True
        fed.share_gate_head = True
        fed.share_amp_head = True
        fed.share_gain_head = True
    return cfg


def get_shared_prefixes(cfg: FullConfig) -> List[str]:
    fed = cfg.federated
    prefixes: List[str] = []
    if fed.share_base_model:
        prefixes.append("base_model.")
    if fed.share_hist_encoder:
        prefixes.append("refiner.hist_encoder.")
    if fed.share_future_mlp:
        prefixes.append("refiner.future_mlp.")
    if fed.share_scalar_mlp:
        prefixes.append("refiner.scalar_mlp.")
    if fed.share_fusion_mlp:
        prefixes.append("refiner.fusion_mlp.")
    if fed.share_gate_head:
        prefixes.append("refiner.gate_head.")
    if fed.share_amp_head:
        prefixes.append("refiner.amp_head.")
    if fed.share_gain_head:
        prefixes.append("refiner.gain_head.")
    return prefixes


def get_shared_param_names(model: nn.Module, cfg: FullConfig) -> List[str]:
    prefixes = get_shared_prefixes(cfg)
    if not prefixes:
        return []
    return [name for name in model.state_dict().keys() if any(name.startswith(prefix) for prefix in prefixes)]


def extract_shared_state(model: nn.Module, shared_param_names: List[str]) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    return {name: state[name].detach().cpu().clone() for name in shared_param_names if name in state}


def load_shared_state(model: nn.Module, shared_state: Optional[Dict[str, torch.Tensor]], strict: bool = False) -> List[str]:
    if not shared_state:
        return []
    current = model.state_dict()
    loaded: List[str] = []
    skipped: List[str] = []
    for name, tensor in shared_state.items():
        if name not in current:
            skipped.append(name)
            continue
        if tuple(current[name].shape) != tuple(tensor.shape):
            skipped.append(name)
            continue
        current[name] = tensor.detach().to(device=current[name].device, dtype=current[name].dtype).clone()
        loaded.append(name)
    if strict and skipped:
        raise KeyError(f"Could not load shared keys: {skipped[:10]}")
    model.load_state_dict(current, strict=False)
    return loaded


def fedavg_shared_states(
    client_updates: List[Dict[str, torch.Tensor]],
    sample_counts: List[int],
) -> Dict[str, torch.Tensor]:
    nonempty = [(state, int(count)) for state, count in zip(client_updates, sample_counts) if state]
    if not nonempty:
        return {}
    keys = set(nonempty[0][0].keys())
    for state, _ in nonempty[1:]:
        keys &= set(state.keys())
    if not keys:
        return {}

    counts = np.asarray([max(0, count) for _, count in nonempty], dtype=np.float64)
    if float(np.sum(counts)) <= 0.0:
        counts = np.ones(len(nonempty), dtype=np.float64)
    weights = counts / float(np.sum(counts))

    averaged: Dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        first = nonempty[0][0][key].detach().cpu()
        if not torch.is_floating_point(first):
            averaged[key] = first.clone()
            continue
        acc = torch.zeros_like(first, dtype=torch.float32)
        for (state, _), weight in zip(nonempty, weights):
            acc = acc + state[key].detach().cpu().float() * float(weight)
        averaged[key] = acc.to(dtype=first.dtype)
    return averaged


def prediction_frame_from_arrays(
    arrays: Dict[str, np.ndarray],
    ds: SelectiveResidualDataset,
    prepared: PreparedData,
    residual_eps: float,
    cfg: FullConfig,
) -> pd.DataFrame:
    if arrays["target"].size == 0:
        return pd.DataFrame()
    idx = arrays["target_index"].astype(int)
    timestamps = ds.split.df.iloc[idx][cfg.data.datetime_col].astype(str).to_numpy()
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
    pred_class_last = take_last_step(arrays["gate_label_pred"], last_idx).astype(np.int64)
    residual_initial = y_true_scaled - base_scaled
    active_initial = np.abs(residual_initial) > float(residual_eps)
    direction_correct_initial = np.full_like(residual_initial, np.nan, dtype=np.float64)
    direction_correct_initial[active_initial] = (
        arrays["corrections"][:, 0][active_initial] * residual_initial[active_initial] > 0.0
    ).astype(float)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true,
            "base_pred": base_pred,
            "final_pred": final_pred,
            "base_error_abs": base_error_abs,
            "final_error_abs": final_error_abs,
            "improved": improved,
            "pred_class_last": pred_class_last,
            "direction_correct_initial": direction_correct_initial,
            "iteration_count": arrays["iteration_count"].astype(np.int64),
        }
    )


def regional_metrics_from_prediction_frames(frames: List[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    valid_frames = [df for df in frames if df is not None and not df.empty]
    if not valid_frames:
        return pd.DataFrame(), {
            "base_RMSE": float("nan"),
            "final_RMSE": float("nan"),
            "RMSE_improvement_percent": float("nan"),
        }
    concat = pd.concat(valid_frames, ignore_index=True)
    regional = (
        concat.groupby("timestamp", as_index=False)[["y_true", "base_pred", "final_pred"]]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    base_metrics = compute_metrics(regional["y_true"].to_numpy(), regional["base_pred"].to_numpy())
    final_metrics = compute_metrics(regional["y_true"].to_numpy(), regional["final_pred"].to_numpy())
    improvement = (base_metrics["RMSE"] - final_metrics["RMSE"]) / (base_metrics["RMSE"] + 1e-8) * 100.0
    metrics: Dict[str, float] = {}
    for k, v in base_metrics.items():
        metrics[f"base_{k}"] = float(v)
    for k, v in final_metrics.items():
        metrics[f"final_{k}"] = float(v)
    metrics["RMSE_improvement_percent"] = float(improvement)
    return regional, metrics


def finite_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


class SelectiveFederatedClient:
    def __init__(self, client_id: int, client_name: str, data_path: str, cfg: FullConfig):
        self.client_id = int(client_id)
        self.client_name = str(client_name)
        self.data_path = data_path
        self.cfg = copy.deepcopy(cfg)
        self.cfg.data.data_path = data_path
        self.cfg.data.save_dir = os.path.join(cfg.federated.save_dir, client_name)
        self.device = self.cfg.train.device

        self.prepared: Optional[PreparedData] = None
        self.base_train_loader: Optional[DataLoader] = None
        self.base_val_loader: Optional[DataLoader] = None
        self.base_test_loader: Optional[DataLoader] = None
        self.base_model: Optional[TrendPeriodBaseModel] = None
        self.refiner_model: Optional[SelectiveResidualFeedbackModel] = None
        self.train_ds: Optional[SelectiveResidualDataset] = None
        self.val_ds: Optional[SelectiveResidualDataset] = None
        self.test_ds: Optional[SelectiveResidualDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.label_stats: Optional[Dict[str, object]] = None
        self.class_weights: Optional[torch.Tensor] = None
        self.scalar_scaler: Optional[SimpleStandardScaler] = None
        self.personalized_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_local_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_final_rmse: float = float("inf")

    def prepare_data(self):
        self.prepared = load_and_prepare_data(self.cfg)
        self.base_train_loader, self.base_val_loader, self.base_test_loader = make_base_loaders(self.prepared, self.cfg)

    def build_base_model(self):
        if self.prepared is None:
            self.prepare_data()
        if self.base_model is None:
            self.base_model = TrendPeriodBaseModel(time_dim=len(self.prepared.time_cols), cfg=self.cfg).to(self.device)

    def build_refiner_model(self):
        if self.train_ds is None:
            raise ValueError("rebuild_refiner_datasets must run before build_refiner_model.")
        if self.refiner_model is None:
            sample = self.train_ds[0]
            self.refiner_model = SelectiveResidualFeedbackModel(
                hist_dim=sample["hist_seq"].shape[-1],
                future_dim=sample["future_features"].numel(),
                scalar_dim=sample["scalar_context_base"].numel(),
                cfg=self.cfg,
            ).to(self.device)

    def combined_model(self) -> FederatedSelectiveNetLoadModel:
        if self.base_model is None:
            self.build_base_model()
        if self.refiner_model is None:
            self.rebuild_refiner_datasets(recompute_label_stats=True)
        return FederatedSelectiveNetLoadModel(self.base_model, self.refiner_model)

    def freeze_base(self):
        if self.base_model is not None:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def generate_base_predictions(self, split: str) -> np.ndarray:
        if self.prepared is None:
            self.prepare_data()
        if self.base_model is None:
            self.build_base_model()
        split_obj = getattr(self.prepared, split)
        return predict_base_series(self.base_model, split_obj, self.prepared.time_cols, self.cfg)

    def rebuild_refiner_datasets(self, recompute_label_stats: bool = True):
        if self.prepared is None:
            self.prepare_data()
        if self.base_model is None:
            self.build_base_model()
        base_train = self.generate_base_predictions("train")
        base_val = self.generate_base_predictions("val")
        base_test = self.generate_base_predictions("test")
        self.train_ds = SelectiveResidualDataset(self.prepared.train, base_train, self.cfg, fit_scalar=True)
        self.val_ds = SelectiveResidualDataset(
            self.prepared.val,
            base_val,
            self.cfg,
            scalar_scaler=self.train_ds.scalar_scaler,
        )
        self.test_ds = SelectiveResidualDataset(
            self.prepared.test,
            base_test,
            self.cfg,
            scalar_scaler=self.train_ds.scalar_scaler,
        )
        kwargs = {"batch_size": self.cfg.train.batch_size}
        self.train_loader = DataLoader(self.train_ds, shuffle=True, **kwargs)
        self.val_loader = DataLoader(self.val_ds, shuffle=False, **kwargs)
        self.test_loader = DataLoader(self.test_ds, shuffle=False, **kwargs)
        self.scalar_scaler = self.train_ds.scalar_scaler
        if recompute_label_stats or self.label_stats is None:
            self.label_stats = build_label_stats(self.train_ds, self.cfg)
            self.class_weights = label_stats_weights_tensor(self.label_stats, self.cfg)
        self.build_refiner_model()

    def load_shared_state(self, shared_state: Optional[Dict[str, torch.Tensor]], strict: bool = False) -> List[str]:
        if not shared_state or self.cfg.federated.aggregation_scope == "none":
            return []
        return load_shared_state(self.combined_model(), shared_state, strict=strict)

    def export_shared_state(self, aggregation_scope: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if aggregation_scope == "none" or self.cfg.federated.aggregation_scope == "none":
            return {}
        model = self.combined_model()
        names = get_shared_param_names(model, self.cfg)
        return extract_shared_state(model, names)

    def train_sample_count(self) -> int:
        if self.train_ds is not None:
            return int(len(self.train_ds))
        if self.base_train_loader is not None:
            return int(len(self.base_train_loader.dataset))
        return 0

    def _trainable_params(self, module: nn.Module) -> List[nn.Parameter]:
        return [param for param in module.parameters() if param.requires_grad]

    def local_train_base(self, epochs: int) -> pd.DataFrame:
        if epochs <= 0:
            return pd.DataFrame()
        if self.base_model is None:
            self.build_base_model()
        params = self._trainable_params(self.base_model)
        if not params:
            return pd.DataFrame()
        optimizer = torch.optim.Adam(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        rows = []
        for epoch in range(1, int(epochs) + 1):
            self.base_model.train()
            losses = []
            for batch in self.base_train_loader:
                batch = move_batch(batch, self.device)
                pred = self.base_model(batch["net_seq"], batch["time_seq"], batch["time_future"])
                loss = F.mse_loss(pred, batch["target"])
                if self.cfg.base.smoothness_lambda > 0.0:
                    loss = loss + self.cfg.base.smoothness_lambda * F.mse_loss(pred, batch["net_seq"][:, -1, :])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            val_loss = evaluate_base_loss(self.base_model, self.base_val_loader, self.cfg)
            rows.append(
                {
                    "client_id": self.client_id,
                    "client_name": self.client_name,
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)) if losses else float("nan"),
                    "val_loss": float(val_loss),
                }
            )
        return pd.DataFrame(rows)

    def local_train_refiner(self, epochs: int) -> pd.DataFrame:
        if epochs <= 0:
            return pd.DataFrame()
        if self.train_ds is None or self.train_loader is None:
            self.rebuild_refiner_datasets(recompute_label_stats=True)
        if self.refiner_model is None:
            self.build_refiner_model()
        params = self._trainable_params(self.refiner_model)
        if not params:
            return pd.DataFrame()
        residual_eps = float(self.label_stats["residual_eps"])
        optimizer = torch.optim.Adam(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        rows = []
        for epoch in range(1, int(epochs) + 1):
            self.refiner_model.train()
            train_parts: Dict[str, List[float]] = {}
            for batch in self.train_loader:
                batch = move_batch(batch, self.device)
                outputs = self.refiner_model(batch)
                loss, parts = compute_selective_refinement_loss(
                    outputs,
                    batch["target"],
                    self.class_weights,
                    residual_eps,
                    self.cfg,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for key, value in parts.items():
                    train_parts.setdefault(key, []).append(value)
            val_parts = evaluate_refinement_loss(
                self.refiner_model,
                self.val_loader,
                self.class_weights,
                residual_eps,
                self.cfg,
            )
            row: Dict[str, object] = {"client_id": self.client_id, "client_name": self.client_name, "epoch": epoch}
            for key, values in train_parts.items():
                row[f"train_{key}"] = float(np.mean(values))
            for key, value in val_parts.items():
                row[f"val_{key}"] = float(value)
            rows.append(row)
        return pd.DataFrame(rows)

    def local_update(
        self,
        shared_state: Optional[Dict[str, torch.Tensor]],
        round_id: int,
        local_epochs: int,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        if self.base_model is None:
            self.build_base_model()
        if self.refiner_model is None:
            self.rebuild_refiner_datasets(recompute_label_stats=True)
        self.load_shared_state(shared_state, strict=False)

        if self.cfg.federated.train_base_in_federated_rounds:
            self.local_train_base(local_epochs)
        recompute = bool(self.cfg.federated.rebuild_label_every_round or self.label_stats is None)
        self.rebuild_refiner_datasets(recompute_label_stats=recompute)
        if self.cfg.federated.train_refiner_in_federated_rounds:
            self.local_train_refiner(local_epochs)

        self.personalized_state = clone_state_dict(self.combined_model().state_dict())
        update = self.export_shared_state(self.cfg.federated.aggregation_scope)
        return update, self.train_sample_count()

    def evaluate(self, split_name: str = "val") -> Tuple[Dict[str, float], pd.DataFrame]:
        if self.train_ds is None:
            self.rebuild_refiner_datasets(recompute_label_stats=True)
        if split_name == "val":
            ds = self.val_ds
            loader = self.val_loader
        elif split_name == "test":
            ds = self.test_ds
            loader = self.test_loader
        elif split_name == "train":
            ds = self.train_ds
            loader = self.train_loader
        else:
            raise ValueError("split_name must be 'train', 'val', or 'test'.")
        residual_eps = float(self.label_stats["residual_eps"])
        loss_parts = evaluate_refinement_loss(self.refiner_model, loader, self.class_weights, residual_eps, self.cfg)
        arrays = predict_refinement(self.refiner_model, loader, self.cfg)
        metrics = summarize_prediction_arrays(arrays, self.prepared, residual_eps, test_loss_parts=loss_parts)
        metrics["client_id"] = self.client_id
        metrics["client_name"] = self.client_name
        metrics["split"] = split_name
        pred_df = prediction_frame_from_arrays(arrays, ds, self.prepared, residual_eps, self.cfg)
        if not pred_df.empty:
            pred_df.insert(0, "client_name", self.client_name)
            pred_df.insert(0, "client_id", self.client_id)
        return metrics, pred_df

    def save_client_outputs(self, save_dir: str) -> Tuple[Dict[str, float], pd.DataFrame]:
        ensure_dir(save_dir)
        if self.train_ds is None:
            self.rebuild_refiner_datasets(recompute_label_stats=True)
        residual_eps = float(self.label_stats["residual_eps"])
        loss_parts = evaluate_refinement_loss(self.refiner_model, self.test_loader, self.class_weights, residual_eps, self.cfg)
        arrays = predict_refinement(self.refiner_model, self.test_loader, self.cfg)
        pred_df, metrics = save_prediction_outputs(
            arrays,
            self.test_ds,
            self.prepared,
            residual_eps,
            self.cfg,
            save_dir,
            test_loss_parts=loss_parts,
        )
        metrics["client_id"] = self.client_id
        metrics["client_name"] = self.client_name
        save_json(os.path.join(save_dir, "label_stats.json"), self.label_stats)
        save_json(os.path.join(save_dir, "config.json"), config_payload(self.cfg, self.scalar_scaler, self.label_stats))
        if self.cfg.federated.save_personalized_client_models:
            torch.save(self.combined_model().state_dict(), os.path.join(save_dir, "personalized_model_final.pth"))
        if not pred_df.empty:
            pred_df.insert(0, "client_name", self.client_name)
            pred_df.insert(0, "client_id", self.client_id)
        return metrics, pred_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Personalized federated selective residual net-load forecasting."
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--client-files", type=str, default=None, help="Comma separated CSV paths.")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--client-fraction", type=float, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--base-pretrain-epochs", type=int, default=None)
    parser.add_argument("--final-local-finetune-epochs", type=int, default=None)
    parser.add_argument("--final-local-train-base", type=int, choices=[0, 1], default=None)
    parser.add_argument("--final-local-train-refiner", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze-base-before-final-local", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--aggregation-scope",
        type=str,
        choices=["none", "base_only", "base_and_encoder", "encoder_only", "all_shared"],
        default=None,
    )
    parser.add_argument("--aggregation-weight", type=str, choices=["num_samples", "uniform"], default=None)
    parser.add_argument("--train-base-in-rounds", type=int, choices=[0, 1], default=None)
    parser.add_argument("--train-refiner-in-rounds", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze-base-after-pretrain", type=int, choices=[0, 1], default=None)
    parser.add_argument("--rebuild-label-every-round", type=int, choices=[0, 1], default=None)
    parser.add_argument("--refiner-backbone", type=str, choices=["cnn_lstm_attention", "weibull_lstm"], default=None)
    parser.add_argument("--feedback-mode", type=str, choices=["none", "static", "dynamic_final_update"], default=None)
    parser.add_argument("--feedback-steps", type=int, default=None)
    parser.add_argument("--safe-max-feedback-steps", type=int, default=None)
    parser.add_argument("--residual-eps-quantile", type=float, default=None)
    parser.add_argument("--final-loss-weight", type=float, default=None)
    parser.add_argument("--cls-loss-weight", type=float, default=None)
    parser.add_argument("--amp-loss-weight", type=float, default=None)
    parser.add_argument("--hold-loss-weight", type=float, default=None)
    parser.add_argument("--use-class-weight", type=int, choices=[0, 1], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--expert-epochs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--base-target-mode", type=str, choices=["raw", "smooth"], default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--residual-eps-mode", type=str, choices=["quantile", "fixed"], default=None)
    parser.add_argument("--residual-eps-fixed", type=float, default=None)
    parser.add_argument("--step-loss-decay", type=float, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg: FullConfig, args: argparse.Namespace) -> FullConfig:
    cfg.base.reuse_client2_base = False
    cfg.base.freeze_after_train = False

    if args.client_files is not None:
        cfg.federated.client_files = [x.strip() for x in args.client_files.split(",") if x.strip()]
    if args.aggregation_scope is not None:
        cfg.federated.aggregation_scope = args.aggregation_scope
    if args.save_dir is not None:
        cfg.federated.save_dir = args.save_dir
    elif cfg.federated.aggregation_scope == "base_only":
        cfg.federated.save_dir = "runs/federated_selective_base_only"
    cfg.data.save_dir = cfg.federated.save_dir

    if args.rounds is not None:
        cfg.federated.rounds = int(args.rounds)
    if args.client_fraction is not None:
        cfg.federated.client_fraction = float(args.client_fraction)
    if args.local_epochs is not None:
        cfg.federated.local_epochs = int(args.local_epochs)
    if args.early_stop_patience is not None:
        cfg.federated.early_stop_patience = int(args.early_stop_patience)
    if args.base_pretrain_epochs is not None:
        cfg.federated.base_pretrain_epochs = int(args.base_pretrain_epochs)
    if args.final_local_finetune_epochs is not None:
        cfg.federated.final_local_finetune_epochs = int(args.final_local_finetune_epochs)
    if args.final_local_train_base is not None:
        cfg.federated.final_local_train_base = bool(args.final_local_train_base)
    if args.final_local_train_refiner is not None:
        cfg.federated.final_local_train_refiner = bool(args.final_local_train_refiner)
    if args.freeze_base_before_final_local is not None:
        cfg.federated.freeze_base_before_final_local = bool(args.freeze_base_before_final_local)
    if args.aggregation_weight is not None:
        cfg.federated.aggregation_weight = args.aggregation_weight
    if args.train_base_in_rounds is not None:
        cfg.federated.train_base_in_federated_rounds = bool(args.train_base_in_rounds)
    if args.train_refiner_in_rounds is not None:
        cfg.federated.train_refiner_in_federated_rounds = bool(args.train_refiner_in_rounds)
    if args.freeze_base_after_pretrain is not None:
        cfg.federated.freeze_base_after_pretrain = bool(args.freeze_base_after_pretrain)
    if args.rebuild_label_every_round is not None:
        cfg.federated.rebuild_label_every_round = bool(args.rebuild_label_every_round)
    if args.refiner_backbone is not None:
        cfg.refiner.backbone = args.refiner_backbone
    if args.feedback_mode is not None:
        cfg.feedback.feedback_mode = args.feedback_mode
    if args.feedback_steps is not None:
        cfg.feedback.feedback_steps = int(args.feedback_steps)
    if args.safe_max_feedback_steps is not None:
        cfg.feedback.safe_max_feedback_steps = int(args.safe_max_feedback_steps)
    if args.residual_eps_quantile is not None:
        cfg.loss.residual_eps_quantile = float(args.residual_eps_quantile)
    if args.final_loss_weight is not None:
        cfg.loss.final_loss_weight = float(args.final_loss_weight)
    if args.cls_loss_weight is not None:
        cfg.loss.cls_loss_weight = float(args.cls_loss_weight)
    if args.amp_loss_weight is not None:
        cfg.loss.amp_loss_weight = float(args.amp_loss_weight)
    if args.hold_loss_weight is not None:
        cfg.loss.hold_loss_weight = float(args.hold_loss_weight)
    if args.use_class_weight is not None:
        cfg.loss.use_class_weight = bool(args.use_class_weight)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    if args.epochs is not None:
        cfg.train.base_epochs = int(args.epochs)
        cfg.train.expert_epochs = int(args.epochs)
        cfg.federated.base_pretrain_epochs = int(args.epochs)
        cfg.federated.final_local_finetune_epochs = int(args.epochs)
    if args.base_epochs is not None:
        cfg.train.base_epochs = int(args.base_epochs)
        if args.base_pretrain_epochs is None:
            cfg.federated.base_pretrain_epochs = int(args.base_epochs)
    if args.expert_epochs is not None:
        cfg.train.expert_epochs = int(args.expert_epochs)
        if args.final_local_finetune_epochs is None:
            cfg.federated.final_local_finetune_epochs = int(args.expert_epochs)
    if args.base_target_mode is not None:
        cfg.base.target_mode = args.base_target_mode
    if args.smooth_window is not None:
        cfg.base.smooth_window = int(args.smooth_window)
    if args.residual_eps_mode is not None:
        cfg.loss.residual_eps_mode = args.residual_eps_mode
    if args.residual_eps_fixed is not None:
        cfg.loss.residual_eps_fixed = float(args.residual_eps_fixed)
    if args.step_loss_decay is not None:
        cfg.loss.step_loss_decay = float(args.step_loss_decay)

    cfg.federated.client_fraction = float(np.clip(cfg.federated.client_fraction, 0.0, 1.0))
    cfg.federated.rounds = max(0, int(cfg.federated.rounds))
    cfg.federated.local_epochs = max(0, int(cfg.federated.local_epochs))
    cfg.federated.early_stop_patience = max(0, int(cfg.federated.early_stop_patience))
    cfg.federated.base_pretrain_epochs = max(0, int(cfg.federated.base_pretrain_epochs))
    cfg.federated.final_local_finetune_epochs = max(0, int(cfg.federated.final_local_finetune_epochs))
    cfg.feedback.feedback_steps = max(1, int(cfg.feedback.feedback_steps))
    cfg.feedback.safe_max_feedback_steps = max(1, int(cfg.feedback.safe_max_feedback_steps))
    cfg.feedback.min_feedback_steps = max(1, int(cfg.feedback.min_feedback_steps))
    cfg.feedback.safe_max_feedback_steps = max(cfg.feedback.safe_max_feedback_steps, cfg.feedback.min_feedback_steps)
    cfg.loss.residual_eps_quantile = float(np.clip(cfg.loss.residual_eps_quantile, 0.0, 1.0))
    cfg.loss.residual_eps_fixed = max(0.0, float(cfg.loss.residual_eps_fixed))
    cfg.loss.step_loss_decay = max(0.0, float(cfg.loss.step_loss_decay))
    return apply_aggregation_scope(cfg)


def select_clients(clients: List[SelectiveFederatedClient], fraction: float) -> List[SelectiveFederatedClient]:
    if fraction >= 1.0:
        return list(clients)
    count = max(1, int(math.ceil(len(clients) * max(0.0, fraction))))
    return random.sample(clients, count)


def evaluate_clients(
    clients: List[SelectiveFederatedClient],
    split_name: str,
) -> Tuple[List[Dict[str, float]], List[pd.DataFrame], Dict[str, float], pd.DataFrame]:
    metrics_rows: List[Dict[str, float]] = []
    frames: List[pd.DataFrame] = []
    for client in clients:
        metrics, pred_df = client.evaluate(split_name)
        metrics_rows.append(metrics)
        frames.append(pred_df)
    regional_df, regional_metrics = regional_metrics_from_prediction_frames(frames)
    return metrics_rows, frames, regional_metrics, regional_df


def make_round_log_row(
    round_id: int,
    selected_clients: List[SelectiveFederatedClient],
    client_metrics: List[Dict[str, float]],
    regional_metrics: Dict[str, float],
    cfg: FullConfig,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "round": int(round_id),
        "selected_clients": ",".join(client.client_name for client in selected_clients),
        "aggregation_scope": cfg.federated.aggregation_scope,
    }
    row["avg_client_val_base_RMSE"] = finite_mean([float(m.get("base_RMSE", np.nan)) for m in client_metrics])
    row["avg_client_val_final_RMSE"] = finite_mean([float(m.get("final_RMSE", np.nan)) for m in client_metrics])
    row["avg_client_val_RMSE_improvement_percent"] = finite_mean(
        [float(m.get("RMSE_improvement_percent", np.nan)) for m in client_metrics]
    )
    row["regional_val_base_RMSE"] = float(regional_metrics.get("base_RMSE", np.nan))
    row["regional_val_final_RMSE"] = float(regional_metrics.get("final_RMSE", np.nan))
    row["regional_val_RMSE_improvement_percent"] = float(
        regional_metrics.get("RMSE_improvement_percent", np.nan)
    )
    row["avg_direction_accuracy"] = finite_mean(
        [float(m.get("direction_accuracy_on_active_residual", np.nan)) for m in client_metrics]
    )
    row["avg_correction_effective_ratio"] = finite_mean(
        [float(m.get("correction_effective_ratio", np.nan)) for m in client_metrics]
    )
    row["avg_hold_ratio_pred"] = finite_mean([float(m.get("hold_ratio_pred", np.nan)) for m in client_metrics])
    row["avg_up_ratio_pred"] = finite_mean([float(m.get("up_ratio_pred", np.nan)) for m in client_metrics])
    row["avg_down_ratio_pred"] = finite_mean([float(m.get("down_ratio_pred", np.nan)) for m in client_metrics])
    return row


def save_regional_outputs(regional_df: pd.DataFrame, regional_metrics: Dict[str, float], save_dir: str):
    regional_df.to_csv(os.path.join(save_dir, "regional_test_predictions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([regional_metrics]).to_csv(
        os.path.join(save_dir, "regional_test_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    if not regional_df.empty:
        _plot_prediction(
            os.path.join(save_dir, "regional_base_vs_final_prediction.png"),
            regional_df,
            [("y_true", "true"), ("base_pred", "base"), ("final_pred", "final")],
            "Regional base vs final prediction",
        )


def save_best_personalized_states(clients: List[SelectiveFederatedClient], path: str):
    payload = {
        client.client_name: {
            "client_id": client.client_id,
            "state_dict": client.best_local_state or clone_state_dict(client.combined_model().state_dict()),
            "best_val_final_rmse": float(client.best_val_final_rmse),
            "label_stats": client.label_stats,
        }
        for client in clients
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    cfg = apply_cli_overrides(FullConfig(), args)
    set_seed(cfg.train.seed)
    save_dir = cfg.federated.save_dir
    ensure_dir(save_dir)
    save_json(os.path.join(save_dir, "config.json"), asdict(cfg))

    print(f"Using device: {cfg.train.device}")
    print(f"Save dir: {save_dir}")
    print(f"Clients: {len(cfg.federated.client_files)}")
    print(f"Aggregation scope: {cfg.federated.aggregation_scope}")
    print(f"Refiner backbone: {cfg.refiner.backbone}")
    print(f"Feedback mode: {cfg.feedback.feedback_mode}")

    clients: List[SelectiveFederatedClient] = []
    for idx, data_path in enumerate(cfg.federated.client_files, start=1):
        client = SelectiveFederatedClient(idx, f"client_{idx}", data_path, cfg)
        print(f"[prepare] {client.client_name}: {data_path}")
        client.prepare_data()
        client.build_base_model()
        clients.append(client)

    print(f"[pretrain] base_pretrain_epochs={cfg.federated.base_pretrain_epochs}")
    for client in clients:
        if cfg.federated.base_pretrain_epochs > 0:
            client.local_train_base(cfg.federated.base_pretrain_epochs)
        if cfg.federated.freeze_base_after_pretrain:
            client.freeze_base()
        client.rebuild_refiner_datasets(recompute_label_stats=True)
        print(
            f"[pretrain] {client.client_name} samples={client.train_sample_count()} "
            f"residual_eps={float(client.label_stats['residual_eps']):.6f}"
        )

    if cfg.federated.aggregation_scope != "none":
        global_shared_state = clients[0].export_shared_state(cfg.federated.aggregation_scope)
    else:
        global_shared_state = {}

    round_logs: List[Dict[str, object]] = []
    best_metric = float("inf")
    best_global_shared_state = clone_state_dict(global_shared_state)
    best_round = 0
    bad_rounds = 0

    for round_id in range(1, cfg.federated.rounds + 1):
        selected = select_clients(clients, cfg.federated.client_fraction)
        print(f"[round {round_id}] selected: {', '.join(client.client_name for client in selected)}")
        client_updates: List[Dict[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        for client in selected:
            update, sample_count = client.local_update(
                global_shared_state,
                round_id=round_id,
                local_epochs=cfg.federated.local_epochs,
            )
            client_updates.append(update)
            sample_counts.append(sample_count)

        if cfg.federated.aggregation_scope != "none":
            if cfg.federated.aggregation_weight == "uniform":
                agg_counts = [1 for _ in sample_counts]
            else:
                agg_counts = sample_counts
            global_shared_state = fedavg_shared_states(client_updates, agg_counts)
            for client in clients:
                client.load_shared_state(global_shared_state, strict=False)
                client.rebuild_refiner_datasets(
                    recompute_label_stats=bool(cfg.federated.rebuild_label_every_round)
                )

        if cfg.federated.evaluate_every_round:
            client_metrics, _, regional_metrics, _ = evaluate_clients(clients, "val")
            row = make_round_log_row(round_id, selected, client_metrics, regional_metrics, cfg)
            round_logs.append(row)
            pd.DataFrame(round_logs).to_csv(
                os.path.join(save_dir, "federated_round_logs.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            current_metric = float(row.get(cfg.federated.best_metric, row["avg_client_val_final_RMSE"]))
            print(
                f"[round {round_id}] avg_val_base_RMSE={row['avg_client_val_base_RMSE']:.6f} "
                f"avg_val_final_RMSE={row['avg_client_val_final_RMSE']:.6f} "
                f"regional_val_final_RMSE={row['regional_val_final_RMSE']:.6f}"
            )
            if current_metric < best_metric - 1e-10:
                best_metric = current_metric
                best_round = round_id
                best_global_shared_state = clone_state_dict(global_shared_state)
                bad_rounds = 0
                for client, metrics in zip(clients, client_metrics):
                    client.best_local_state = clone_state_dict(client.combined_model().state_dict())
                    client.best_val_final_rmse = float(metrics.get("final_RMSE", np.nan))
                if cfg.federated.save_global_shared_state:
                    torch.save(best_global_shared_state, os.path.join(save_dir, "global_shared_state_best.pth"))
                if cfg.federated.save_personalized_client_models:
                    save_best_personalized_states(
                        clients,
                        os.path.join(save_dir, "client_personalized_states_best.pth"),
                    )
            else:
                bad_rounds += 1
                patience = int(cfg.federated.early_stop_patience)
                if patience > 0 and bad_rounds >= patience:
                    print(f"[round {round_id}] early stopping after {bad_rounds} stale rounds.")
                    break

    if not round_logs and cfg.federated.evaluate_every_round:
        client_metrics, _, regional_metrics, _ = evaluate_clients(clients, "val")
        row = make_round_log_row(0, clients, client_metrics, regional_metrics, cfg)
        round_logs.append(row)
        pd.DataFrame(round_logs).to_csv(
            os.path.join(save_dir, "federated_round_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        best_global_shared_state = clone_state_dict(global_shared_state)
        for client, metrics in zip(clients, client_metrics):
            client.best_local_state = clone_state_dict(client.combined_model().state_dict())
            client.best_val_final_rmse = float(metrics.get("final_RMSE", np.nan))

    if cfg.federated.aggregation_scope != "none" and best_global_shared_state:
        print(f"[best] loading best shared state from round {best_round}")
        for client in clients:
            client.load_shared_state(best_global_shared_state, strict=False)
            client.rebuild_refiner_datasets(recompute_label_stats=True)

    if cfg.federated.final_local_finetune_epochs > 0:
        final_train_base = (
            cfg.federated.train_base_in_federated_rounds
            if cfg.federated.final_local_train_base is None
            else bool(cfg.federated.final_local_train_base)
        )
        final_train_refiner = (
            cfg.federated.train_refiner_in_federated_rounds
            if cfg.federated.final_local_train_refiner is None
            else bool(cfg.federated.final_local_train_refiner)
        )
        print(
            f"[finetune] final_local_finetune_epochs={cfg.federated.final_local_finetune_epochs} "
            f"train_base={final_train_base} train_refiner={final_train_refiner} "
            f"freeze_base={cfg.federated.freeze_base_before_final_local}"
        )
        for client in clients:
            if cfg.federated.freeze_base_before_final_local:
                client.freeze_base()
            if final_train_base:
                client.local_train_base(cfg.federated.final_local_finetune_epochs)
                client.rebuild_refiner_datasets(recompute_label_stats=True)
            if final_train_refiner:
                client.local_train_refiner(cfg.federated.final_local_finetune_epochs)
            client.personalized_state = clone_state_dict(client.combined_model().state_dict())

    if cfg.federated.save_global_shared_state:
        torch.save(global_shared_state, os.path.join(save_dir, "global_shared_state_final.pth"))
        if not os.path.exists(os.path.join(save_dir, "global_shared_state_best.pth")):
            torch.save(best_global_shared_state, os.path.join(save_dir, "global_shared_state_best.pth"))
    if cfg.federated.save_personalized_client_models:
        save_best_personalized_states(clients, os.path.join(save_dir, "client_personalized_states_best.pth"))

    test_metric_rows: List[Dict[str, float]] = []
    test_frames: List[pd.DataFrame] = []
    for client in clients:
        client_dir = os.path.join(save_dir, client.client_name)
        metrics, pred_df = client.save_client_outputs(client_dir)
        test_metric_rows.append(metrics)
        test_frames.append(pred_df)

    pd.DataFrame(test_metric_rows).to_csv(
        os.path.join(save_dir, "all_clients_test_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    regional_df, regional_metrics = regional_metrics_from_prediction_frames(test_frames)
    save_regional_outputs(regional_df, regional_metrics, save_dir)

    final_payload = asdict(cfg)
    final_payload["best_round"] = int(best_round)
    final_payload["best_metric_value"] = float(best_metric)
    final_payload["client_label_stats"] = {client.client_name: client.label_stats for client in clients}
    save_json(os.path.join(save_dir, "config.json"), final_payload)

    avg_client_base_rmse = finite_mean([float(row.get("base_RMSE", np.nan)) for row in test_metric_rows])
    avg_client_final_rmse = finite_mean([float(row.get("final_RMSE", np.nan)) for row in test_metric_rows])
    avg_client_improvement = finite_mean(
        [float(row.get("RMSE_improvement_percent", np.nan)) for row in test_metric_rows]
    )
    regional_base_rmse = float(regional_metrics.get("base_RMSE", np.nan))
    regional_final_rmse = float(regional_metrics.get("final_RMSE", np.nan))
    regional_improvement = float(regional_metrics.get("RMSE_improvement_percent", np.nan))

    print(f"avg client base RMSE: {avg_client_base_rmse:.6f}")
    print(f"avg client final RMSE: {avg_client_final_rmse:.6f}")
    print(f"avg client improvement: {avg_client_improvement:.2f}%")
    print(f"regional base RMSE: {regional_base_rmse:.6f}")
    print(f"regional final RMSE: {regional_final_rmse:.6f}")
    print(f"regional improvement: {regional_improvement:.2f}%")
    print(f"results saved to {save_dir}")


if __name__ == "__main__":
    main()
