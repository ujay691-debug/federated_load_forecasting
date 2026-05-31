import json
import os
import random
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore", message=".*padding='same'.*")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs_cnn_lstm_netload_three_experiments")


def build_default_client_files() -> List[str]:
    order = [1, 5, 6, 7, 8, 9, 2, 3, 4]
    return [
        os.path.join(PROJECT_ROOT, "per_client_merged", f"client_{idx}_load_weather_30min.csv")
        for idx in order
    ]


@dataclass
class DataConfig:
    client_files: List[str] = field(default_factory=build_default_client_files)
    datetime_col: str = "timestamp"
    target_col: str = "gc"
    net_load_col: str = "net_load"

    seq_len: int = 48
    horizon: int = 1

    train_ratio: float = 0.8
    val_ratio: float = 0.1

    dropna: bool = True
    sort_by_time: bool = True
    freq_minutes: str = "auto"
    save_dir: str = os.path.join(RUNS_DIR, "net_load_multi_dataset")

    use_time_range: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    drop_duplicate_test_timestamps: bool = True


@dataclass
class FeatureConfig:
    raw_feature_cols: List[str] = field(default_factory=lambda: ["gc"])

    use_slot_sin_cos: bool = True
    use_weekday_sin_cos: bool = True
    use_month_sin_cos: bool = True
    use_is_weekend: bool = True
    use_is_holiday: bool = False

    use_temp_c: bool = True
    temp_source_mode: str = "auto"
    temp_c_col: str = "temp2m_c"
    temp_k_col: str = "temp2m_k"

    use_rh: bool = False
    rh_col: str = "rh2m_pct"

    use_wind: bool = True
    wind_col: str = "wind10m_ms"

    use_ghi: bool = False
    ghi_col: str = "ghi_wm2"

    use_apparent_temp: bool = False

    no_scale_cols: List[str] = field(default_factory=lambda: ["is_weekend", "is_holiday"])


@dataclass
class ModelConfig:
    use_attention: bool = False

    conv1_channels: int = 32
    conv2_channels: int = 64
    conv1_kernel: int = 3
    conv2_kernel: int = 3

    pool1_kernel: int = 2
    pool2_kernel: int = 3

    lstm_hidden1: int = 32
    lstm_hidden2: int = 16

    attn_units: int = 20
    fc_hidden: int = 16
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    random_seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True

    loss_name: str = "mse"
    optimizer_name: str = "adam"

    scaler_x: str = "minmax"
    scaler_y: str = "minmax"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    run_scope: str = "both"  # "per_client", "aggregate", "both"
    method_mode: str = "both"  # "direct", "indirect", "both"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config(cfg: Config, save_dir: str) -> None:
    ensure_dir(save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def infer_freq_minutes(dt_series: pd.Series) -> int:
    dt_sorted = pd.to_datetime(dt_series).sort_values().drop_duplicates()
    diffs = dt_sorted.diff().dropna()
    if len(diffs) == 0:
        raise ValueError("Cannot infer data frequency from the timestamp column.")
    return int(diffs.mode().iloc[0].total_seconds() // 60)


def get_slots_per_day(freq_minutes: int) -> int:
    if freq_minutes <= 0 or 1440 % freq_minutes != 0:
        raise ValueError(f"Invalid frequency minutes: {freq_minutes}")
    return 1440 // freq_minutes


def get_scaler(name: str):
    name = name.lower()
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    if name == "none":
        return None
    raise ValueError(f"Unsupported scaler type: {name}")


def inverse_transform_array(scaler, arr: np.ndarray) -> np.ndarray:
    if scaler is None:
        return arr
    original_shape = arr.shape
    arr_2d = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    restored = scaler.inverse_transform(arr_2d)
    return restored.reshape(original_shape)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
        "R2": float(r2),
    }


def print_metrics(metrics: dict, title: str) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if "percent" in key.lower():
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.6f}")


def save_metrics_csv(metrics: dict, save_path: str) -> None:
    pd.DataFrame([metrics]).to_csv(save_path, index=False, encoding="utf-8-sig")


def add_derived_columns(df: pd.DataFrame, net_load_col: str) -> pd.DataFrame:
    out = df.copy()
    if "gc" not in out.columns:
        raise ValueError("The dataset is missing the gc column.")
    if "gg" not in out.columns:
        raise ValueError("The dataset is missing the gg column.")
    out[net_load_col] = out["gc"].astype(float) - out["gg"].astype(float)
    return out


def add_timestamp_occurrence_key(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out["_timestamp_occurrence"] = out.groupby(timestamp_col).cumcount()
    return out


def drop_duplicate_timestamps(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.drop_duplicates(subset=[timestamp_col], keep="first")
    return out.sort_values(timestamp_col).reset_index(drop=True)


def get_temp_source_col(df: pd.DataFrame, fc: FeatureConfig) -> Optional[str]:
    mode = fc.temp_source_mode.lower()
    if mode == "auto":
        if fc.temp_c_col in df.columns:
            return fc.temp_c_col
        if fc.temp_k_col in df.columns:
            return fc.temp_k_col
        raise ValueError(
            f"Temperature features are enabled, but neither {fc.temp_c_col} nor {fc.temp_k_col} exists."
        )
    if mode == "c":
        if fc.temp_c_col not in df.columns:
            raise ValueError(f"Missing required column: {fc.temp_c_col}")
        return fc.temp_c_col
    if mode == "k":
        if fc.temp_k_col not in df.columns:
            raise ValueError(f"Missing required column: {fc.temp_k_col}")
        return fc.temp_k_col
    raise ValueError('temp_source_mode must be "auto", "c", or "k".')


def get_weather_source_cols(cfg: Config) -> List[str]:
    cols: List[str] = []
    fc = cfg.feature
    mode = fc.temp_source_mode.lower()
    if fc.use_temp_c:
        if mode == "auto":
            cols.extend([fc.temp_c_col, fc.temp_k_col])
        elif mode == "c":
            cols.append(fc.temp_c_col)
        elif mode == "k":
            cols.append(fc.temp_k_col)
        else:
            raise ValueError('temp_source_mode must be "auto", "c", or "k".')
    if fc.use_rh:
        cols.append(fc.rh_col)
    if fc.use_wind:
        cols.append(fc.wind_col)
    if fc.use_ghi:
        cols.append(fc.ghi_col)
    return list(dict.fromkeys(cols))


def prepare_common_df_from_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_derived_columns(df, cfg.data.net_load_col)
    dt_col = cfg.data.datetime_col

    if dt_col not in df.columns:
        raise ValueError(f"The dataset is missing the datetime column: {dt_col}")

    df[dt_col] = pd.to_datetime(df[dt_col])
    if cfg.data.sort_by_time:
        df = df.sort_values(dt_col).reset_index(drop=True)

    if cfg.data.use_time_range:
        if cfg.data.start_time is not None:
            df = df[df[dt_col] >= pd.to_datetime(cfg.data.start_time)]
        if cfg.data.end_time is not None:
            df = df[df[dt_col] <= pd.to_datetime(cfg.data.end_time)]
        df = df.reset_index(drop=True)

    required_cols = [dt_col, "gc", "gg", cfg.data.net_load_col]
    if cfg.feature.use_temp_c:
        required_cols.append(get_temp_source_col(df, cfg.feature))
    if cfg.feature.use_rh:
        if cfg.feature.rh_col not in df.columns:
            raise ValueError(f"Missing required column: {cfg.feature.rh_col}")
        required_cols.append(cfg.feature.rh_col)
    if cfg.feature.use_wind:
        if cfg.feature.wind_col not in df.columns:
            raise ValueError(f"Missing required column: {cfg.feature.wind_col}")
        required_cols.append(cfg.feature.wind_col)
    if cfg.feature.use_ghi:
        if cfg.feature.ghi_col not in df.columns:
            raise ValueError(f"Missing required column: {cfg.feature.ghi_col}")
        required_cols.append(cfg.feature.ghi_col)

    if cfg.data.dropna:
        df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df


def load_common_df(data_path: str, cfg: Config) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file does not exist: {data_path}")
    return prepare_common_df_from_dataframe(pd.read_csv(data_path), cfg)


def build_aggregate_common_df(client_files: List[str], cfg: Config) -> pd.DataFrame:
    dt_col = cfg.data.datetime_col
    weather_cols = get_weather_source_cols(cfg)

    merged = None
    for idx, path in enumerate(client_files, start=1):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file does not exist: {path}")

        df = pd.read_csv(path)
        df = add_derived_columns(df, cfg.data.net_load_col)

        if dt_col not in df.columns:
            raise ValueError(f"{path} is missing the datetime column: {dt_col}")

        df[dt_col] = pd.to_datetime(df[dt_col])
        if cfg.data.sort_by_time:
            df = df.sort_values(dt_col).reset_index(drop=True)

        if cfg.data.use_time_range:
            if cfg.data.start_time is not None:
                df = df[df[dt_col] >= pd.to_datetime(cfg.data.start_time)]
            if cfg.data.end_time is not None:
                df = df[df[dt_col] <= pd.to_datetime(cfg.data.end_time)]
            df = df.reset_index(drop=True)

        keep_cols = [dt_col, "gc", "gg", cfg.data.net_load_col]
        for col in weather_cols:
            if col in df.columns:
                keep_cols.append(col)
        keep_cols = list(dict.fromkeys(keep_cols))

        tmp = add_timestamp_occurrence_key(df[keep_cols].copy(), dt_col)
        rename_map = {
            col: f"{col}_client_{idx}"
            for col in tmp.columns
            if col not in [dt_col, "_timestamp_occurrence"]
        }
        tmp = tmp.rename(columns=rename_map)

        if merged is None:
            merged = tmp
        else:
            merged = pd.merge(merged, tmp, on=[dt_col, "_timestamp_occurrence"], how="inner")

    if merged is None or len(merged) == 0:
        raise ValueError("The aggregated dataset is empty after aligning all client timestamps.")

    out = pd.DataFrame({dt_col: merged[dt_col].copy()})
    gc_cols = [col for col in merged.columns if col.startswith("gc_client_")]
    gg_cols = [col for col in merged.columns if col.startswith("gg_client_")]
    out["gc"] = merged[gc_cols].sum(axis=1)
    out["gg"] = merged[gg_cols].sum(axis=1)
    out[cfg.data.net_load_col] = out["gc"] - out["gg"]

    for col in weather_cols:
        candidate_cols = [name for name in merged.columns if name.startswith(f"{col}_client_")]
        if len(candidate_cols) > 0:
            out[col] = merged[candidate_cols].mean(axis=1)

    return prepare_common_df_from_dataframe(out, cfg)


def compute_split_indices(n: int, cfg: Config) -> Tuple[int, int]:
    train_end = int(n * cfg.data.train_ratio)
    val_end = int(n * (cfg.data.train_ratio + cfg.data.val_ratio))
    if train_end <= cfg.data.seq_len or val_end <= train_end:
        raise ValueError("The train/validation split is too small to create valid sequences.")
    return train_end, val_end


def flatten_columns(columns) -> List[str]:
    names = []
    for col in columns:
        if isinstance(col, tuple):
            names.append("_".join([str(item) for item in col if str(item) != ""]))
        else:
            names.append(str(col))
    return names


def plot_curves(train_losses: List[float], val_losses: List[float], val_r2_list: List[float], save_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Train / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_r2_list, label="Validation R2")
    plt.title("Validation R2")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_r2_curve.png"), dpi=200)
    plt.close()


def plot_prediction_from_df(pred_df: pd.DataFrame, save_path: str, title: str, ylabel: str, show_n: int = 200) -> None:
    show_n = min(show_n, len(pred_df))
    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["y_true_step_1"].values[:show_n], label=f"True {ylabel}")
    plt.plot(pred_df["y_pred_step_1"].values[:show_n], label=f"Predicted {ylabel}")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def build_method_compare_df(direct_pred_df: pd.DataFrame, indirect_pred_df: pd.DataFrame) -> pd.DataFrame:
    left = add_timestamp_occurrence_key(direct_pred_df, "timestamp")
    right = add_timestamp_occurrence_key(indirect_pred_df, "timestamp")
    merged = pd.merge(
        left,
        right,
        on=["timestamp", "_timestamp_occurrence"],
        how="inner",
        suffixes=("_direct", "_indirect"),
    )

    return pd.DataFrame({
        "timestamp": merged["timestamp"],
        "y_true_direct": merged["y_true_step_1_direct"],
        "y_pred_direct": merged["y_pred_step_1_direct"],
        "y_true_indirect": merged["y_true_step_1_indirect"],
        "y_pred_indirect": merged["y_pred_step_1_indirect"],
    }).sort_values("timestamp").reset_index(drop=True)


def plot_netload_method_compare(direct_pred_df: pd.DataFrame, indirect_pred_df: pd.DataFrame, save_path: str, show_n: int = 200) -> None:
    merged = build_method_compare_df(direct_pred_df, indirect_pred_df)
    show_n = min(show_n, len(merged))

    plt.figure(figsize=(12, 5))
    plt.plot(merged["y_true_direct"].values[:show_n], label="True Net Load")
    plt.plot(merged["y_pred_direct"].values[:show_n], label="Direct Method")
    plt.plot(merged["y_pred_indirect"].values[:show_n], label="Indirect Method")
    plt.title("Direct vs Indirect Net Load Prediction")
    plt.xlabel("Sample")
    plt.ylabel("net_load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


class SamePadMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        total_pad = self.kernel_size - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = F.pad(x, (pad_left, pad_right), mode="constant", value=float("-inf"))
        return F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)


class Attention(nn.Module):
    def __init__(self, input_dim: int, attn_units: int):
        super().__init__()
        self.score_vec = nn.Linear(input_dim, input_dim, bias=False)
        self.attn_out = nn.Linear(input_dim * 2, attn_units, bias=False)

    def forward(self, x):
        score_first_part = self.score_vec(x)
        h_t = x[:, -1, :]
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        pre_activation = torch.cat([context, h_t], dim=1)
        return torch.tanh(self.attn_out(pre_activation))


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: ModelConfig):
        super().__init__()
        self.use_attention = cfg.use_attention

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cfg.conv1_channels,
            kernel_size=cfg.conv1_kernel,
            stride=1,
            padding=cfg.conv1_kernel // 2,
        )
        self.pool1 = SamePadMaxPool1d(kernel_size=cfg.pool1_kernel, stride=1)

        self.conv2 = nn.Conv1d(
            in_channels=cfg.conv1_channels,
            out_channels=cfg.conv2_channels,
            kernel_size=cfg.conv2_kernel,
            stride=1,
            padding=cfg.conv2_kernel // 2,
        )
        self.pool2 = SamePadMaxPool1d(kernel_size=cfg.pool2_kernel, stride=1)

        self.dropout = nn.Dropout(cfg.dropout)
        self.lstm1 = nn.LSTM(input_size=cfg.conv2_channels, hidden_size=cfg.lstm_hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=cfg.lstm_hidden1, hidden_size=cfg.lstm_hidden2, batch_first=True)

        if self.use_attention:
            self.attention = Attention(input_dim=cfg.lstm_hidden2, attn_units=cfg.attn_units)
            self.fc1 = nn.Linear(cfg.attn_units, cfg.fc_hidden)
        else:
            self.fc1 = nn.Linear(cfg.lstm_hidden2, cfg.fc_hidden)

        self.fc2 = nn.Linear(cfg.fc_hidden, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        if self.use_attention:
            x = self.attention(x)
        else:
            x = x[:, -1, :]

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    dc = cfg.data
    fc = cfg.feature

    if dc.datetime_col not in df.columns:
        raise ValueError(f"Missing datetime column: {dc.datetime_col}")

    df[dc.datetime_col] = pd.to_datetime(df[dc.datetime_col])
    if dc.sort_by_time:
        df = df.sort_values(dc.datetime_col).reset_index(drop=True)

    if dc.use_time_range:
        if dc.start_time is not None:
            df = df[df[dc.datetime_col] >= pd.to_datetime(dc.start_time)]
        if dc.end_time is not None:
            df = df[df[dc.datetime_col] <= pd.to_datetime(dc.end_time)]
        df = df.reset_index(drop=True)

    freq_minutes = infer_freq_minutes(df[dc.datetime_col]) if dc.freq_minutes == "auto" else int(dc.freq_minutes)
    slots_per_day = get_slots_per_day(freq_minutes)

    dt = df[dc.datetime_col]
    slot_idx = (dt.dt.hour * 60 + dt.dt.minute) // freq_minutes
    weekday_idx = dt.dt.weekday
    month_idx = dt.dt.month - 1

    if fc.use_slot_sin_cos:
        df["slot_sin"] = np.sin(2 * np.pi * slot_idx / slots_per_day)
        df["slot_cos"] = np.cos(2 * np.pi * slot_idx / slots_per_day)

    if fc.use_weekday_sin_cos:
        df["weekday_sin"] = np.sin(2 * np.pi * weekday_idx / 7.0)
        df["weekday_cos"] = np.cos(2 * np.pi * weekday_idx / 7.0)

    if fc.use_month_sin_cos:
        df["month_sin"] = np.sin(2 * np.pi * month_idx / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * month_idx / 12.0)

    if fc.use_is_weekend:
        df["is_weekend"] = (weekday_idx >= 5).astype(int)

    if fc.use_is_holiday and "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    if fc.use_temp_c:
        temp_mode = fc.temp_source_mode.lower()
        if temp_mode == "auto":
            if fc.temp_c_col in df.columns:
                df["temp_c"] = df[fc.temp_c_col].astype(float)
            elif fc.temp_k_col in df.columns:
                df["temp_c"] = df[fc.temp_k_col].astype(float) - 273.15
            else:
                raise ValueError(f"Neither {fc.temp_c_col} nor {fc.temp_k_col} exists in the dataset.")
        elif temp_mode == "c":
            if fc.temp_c_col not in df.columns:
                raise ValueError(f"Missing required column: {fc.temp_c_col}")
            df["temp_c"] = df[fc.temp_c_col].astype(float)
        elif temp_mode == "k":
            if fc.temp_k_col not in df.columns:
                raise ValueError(f"Missing required column: {fc.temp_k_col}")
            df["temp_c"] = df[fc.temp_k_col].astype(float) - 273.15
        else:
            raise ValueError('temp_source_mode must be "auto", "c", or "k".')

    if fc.use_rh:
        if fc.rh_col not in df.columns:
            raise ValueError(f"Missing required column: {fc.rh_col}")
        df["rh2m_pct"] = df[fc.rh_col].clip(lower=0, upper=100)

    if fc.use_wind:
        if fc.wind_col not in df.columns:
            raise ValueError(f"Missing required column: {fc.wind_col}")
        df["wind10m_ms"] = df[fc.wind_col].clip(lower=0)

    if fc.use_ghi:
        if fc.ghi_col not in df.columns:
            raise ValueError(f"Missing required column: {fc.ghi_col}")
        df["ghi_wm2"] = df[fc.ghi_col].astype(float)

    if fc.use_apparent_temp:
        needed = ["temp_c", "rh2m_pct", "wind10m_ms"]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for apparent temperature: {missing}")
        e = (df["rh2m_pct"] / 100.0) * 6.105 * np.exp(17.27 * df["temp_c"] / (237.7 + df["temp_c"]))
        df["apparent_temp_c"] = df["temp_c"] + 0.33 * e - 0.70 * df["wind10m_ms"] - 4.0

    feature_cols: List[str] = []
    feature_cols.extend(fc.raw_feature_cols)

    optional_cols = [
        "slot_sin", "slot_cos",
        "weekday_sin", "weekday_cos",
        "month_sin", "month_cos",
        "is_weekend", "is_holiday",
        "temp_c", "rh2m_pct", "wind10m_ms", "ghi_wm2", "apparent_temp_c",
    ]
    for col in optional_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)

    missing_required = [col for col in [dc.target_col] + feature_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    if dc.dropna:
        df = df.dropna(subset=feature_cols + [dc.target_col]).reset_index(drop=True)

    return df, feature_cols


def split_df_by_fixed_indices(df: pd.DataFrame, train_end: int, val_end: int):
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def fit_and_transform_x(train_df, val_df, test_df, feature_cols, cfg: Config):
    no_scale_cols = set(cfg.feature.no_scale_cols)
    scale_cols = [col for col in feature_cols if col not in no_scale_cols]
    keep_cols = [col for col in feature_cols if col in no_scale_cols]

    x_scaler = get_scaler(cfg.train.scaler_x)
    if x_scaler is not None and len(scale_cols) > 0:
        x_scaler.fit(train_df[scale_cols].values)

        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()

        train_scaled.loc[:, scale_cols] = x_scaler.transform(train_df[scale_cols].values)
        val_scaled.loc[:, scale_cols] = x_scaler.transform(val_df[scale_cols].values)
        test_scaled.loc[:, scale_cols] = x_scaler.transform(test_df[scale_cols].values)
    else:
        x_scaler = None
        train_scaled, val_scaled, test_scaled = train_df.copy(), val_df.copy(), test_df.copy()

    return train_scaled, val_scaled, test_scaled, x_scaler, scale_cols, keep_cols


def fit_and_transform_y(train_df, val_df, test_df, cfg: Config):
    target_col = cfg.data.target_col
    y_scaler = get_scaler(cfg.train.scaler_y)

    train_y = train_df[[target_col]].values
    val_y = val_df[[target_col]].values
    test_y = test_df[[target_col]].values

    if y_scaler is not None:
        train_y = y_scaler.fit_transform(train_y)
        val_y = y_scaler.transform(val_y)
        test_y = y_scaler.transform(test_y)

    return train_y, val_y, test_y, y_scaler


def create_sequences(
    x_values: np.ndarray,
    y_values: np.ndarray,
    timestamps,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[pd.Timestamp]]]:
    x_seq: List[np.ndarray] = []
    y_seq: List[np.ndarray] = []
    ts_seq: List[List[pd.Timestamp]] = []

    total = len(x_values)
    limit = total - seq_len - horizon + 1
    if limit <= 0:
        return (
            np.empty((0, seq_len, x_values.shape[1]), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
            [],
        )

    timestamps = pd.to_datetime(pd.Series(timestamps)).tolist()

    for start in range(limit):
        end = start + seq_len
        horizon_end = end + horizon
        x_seq.append(x_values[start:end])
        y_seq.append(y_values[end:horizon_end].reshape(-1))
        ts_seq.append(timestamps[end:horizon_end])

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        ts_seq,
    )


def make_dataloader(x: np.ndarray, y: np.ndarray, cfg: Config, shuffle: bool) -> DataLoader:
    dataset = SeqDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and cfg.train.device.startswith("cuda"),
    )


def prepare_single_dataset(common_df: pd.DataFrame, cfg: Config) -> dict:
    feature_df, feature_cols = build_features(common_df, cfg)
    train_end, val_end = compute_split_indices(len(feature_df), cfg)
    train_df, val_df, test_df = split_df_by_fixed_indices(feature_df, train_end, val_end)

    train_scaled, val_scaled, test_scaled, x_scaler, _, _ = fit_and_transform_x(
        train_df, val_df, test_df, feature_cols, cfg
    )
    train_y, val_y, test_y, y_scaler = fit_and_transform_y(train_df, val_df, test_df, cfg)

    x_train = train_scaled[feature_cols].values.astype(np.float32)
    x_val = val_scaled[feature_cols].values.astype(np.float32)
    x_test = test_scaled[feature_cols].values.astype(np.float32)

    train_ts = train_df[cfg.data.datetime_col].tolist()
    val_ts = val_df[cfg.data.datetime_col].tolist()
    test_ts = test_df[cfg.data.datetime_col].tolist()

    x_train_seq, y_train_seq, train_ts_seq = create_sequences(
        x_train, train_y, train_ts, cfg.data.seq_len, cfg.data.horizon
    )
    x_val_seq, y_val_seq, val_ts_seq = create_sequences(
        x_val, val_y, val_ts, cfg.data.seq_len, cfg.data.horizon
    )
    x_test_seq, y_test_seq, test_ts_seq = create_sequences(
        x_test, test_y, test_ts, cfg.data.seq_len, cfg.data.horizon
    )

    counts = {
        "train_seq_count": len(x_train_seq),
        "val_seq_count": len(x_val_seq),
        "test_seq_count": len(x_test_seq),
    }
    empty_splits = [name for name, count in counts.items() if count == 0]
    if empty_splits:
        raise ValueError(f"Insufficient samples to build sequences for: {empty_splits}")

    return {
        "feature_df": feature_df,
        "feature_cols": feature_cols,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "loaders": {
            "train": make_dataloader(x_train_seq, y_train_seq, cfg, shuffle=True),
            "train_eval": make_dataloader(x_train_seq, y_train_seq, cfg, shuffle=False),
            "val": make_dataloader(x_val_seq, y_val_seq, cfg, shuffle=False),
            "test": make_dataloader(x_test_seq, y_test_seq, cfg, shuffle=False),
        },
        "timestamps": {
            "train": train_ts_seq,
            "val": val_ts_seq,
            "test": test_ts_seq,
        },
        "counts": counts,
        "splits": {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        },
    }


def get_loss_fn(cfg: Config):
    name = cfg.train.loss_name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name in ["mae", "l1"]:
        return nn.L1Loss()
    if name in ["smoothl1", "huber"]:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss function: {cfg.train.loss_name}")


def get_optimizer(model: nn.Module, cfg: Config):
    name = cfg.train.optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer_name}")


def run_one_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0
    pred_batches: List[np.ndarray] = []
    true_batches: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(xb)
            loss = loss_fn(out, yb)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        pred_batches.append(out.detach().cpu().numpy())
        true_batches.append(yb.detach().cpu().numpy())

    avg_loss = total_loss / max(total_count, 1)
    y_pred = np.concatenate(pred_batches, axis=0) if pred_batches else np.empty((0, 1), dtype=np.float32)
    y_true = np.concatenate(true_batches, axis=0) if true_batches else np.empty((0, 1), dtype=np.float32)
    return avg_loss, y_true, y_pred


def build_prediction_df(
    timestamps: List[List[pd.Timestamp]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    timestamp_col: str,
) -> pd.DataFrame:
    if len(timestamps) != len(y_true) or len(timestamps) != len(y_pred):
        raise ValueError("Prediction arrays and timestamps are not aligned.")

    data = {timestamp_col: [pd.to_datetime(items[0]) for items in timestamps]}
    for step in range(horizon):
        data[f"y_true_step_{step + 1}"] = y_true[:, step]
        data[f"y_pred_step_{step + 1}"] = y_pred[:, step]

    return pd.DataFrame(data)


def calc_prediction_df_metrics(pred_df: pd.DataFrame, horizon: int) -> dict:
    true_cols = [f"y_true_step_{step + 1}" for step in range(horizon)]
    pred_cols = [f"y_pred_step_{step + 1}" for step in range(horizon)]
    y_true = pred_df[true_cols].values.reshape(-1)
    y_pred = pred_df[pred_cols].values.reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return calc_metrics(y_true[mask], y_pred[mask])


def run_single_experiment(
    cfg: Config,
    common_df: pd.DataFrame,
    save_dir: str,
    experiment_name: str,
    ylabel: str,
) -> dict:
    ensure_dir(save_dir)
    save_config(cfg, save_dir)

    prepared = prepare_single_dataset(common_df, cfg)
    feature_cols = prepared["feature_cols"]
    y_scaler = prepared["y_scaler"]
    loaders = prepared["loaders"]
    timestamp_sequences = prepared["timestamps"]

    model = CNNLSTMModel(
        input_dim=len(feature_cols),
        output_dim=cfg.data.horizon,
        cfg=cfg.model,
    ).to(cfg.train.device)
    loss_fn = get_loss_fn(cfg)
    optimizer = get_optimizer(model, cfg)

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_r2_list: List[float] = []
    epoch_rows: List[dict] = []

    best_epoch = 0
    best_val_rmse = float("inf")
    best_state_dict = None

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, _, _ = run_one_epoch(model, loaders["train"], loss_fn, cfg.train.device, optimizer=optimizer)
        val_loss, val_true_scaled, val_pred_scaled = run_one_epoch(model, loaders["val"], loss_fn, cfg.train.device)

        val_true = inverse_transform_array(y_scaler, val_true_scaled)
        val_pred = inverse_transform_array(y_scaler, val_pred_scaled)
        val_metrics = calc_metrics(val_true.reshape(-1), val_pred.reshape(-1))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_list.append(val_metrics["R2"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_MAE": val_metrics["MAE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAPE_percent": val_metrics["MAPE_percent"],
            "val_R2": val_metrics["R2"],
        }
        epoch_rows.append(row)

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"[{experiment_name}] Epoch {epoch:03d}/{cfg.train.epochs:03d} | "
            f"TrainLoss={train_loss:.6f} | ValLoss={val_loss:.6f} | "
            f"ValRMSE={val_metrics['RMSE']:.6f} | ValR2={val_metrics['R2']:.6f}"
        )

    if best_state_dict is None:
        raise RuntimeError(f"Failed to train {experiment_name}: no best model was captured.")

    model.load_state_dict(best_state_dict)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "feature_cols": feature_cols,
            "config": asdict(cfg),
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
        },
        os.path.join(save_dir, "best_model.pth"),
    )
    if prepared["x_scaler"] is not None:
        joblib.dump(prepared["x_scaler"], os.path.join(save_dir, "x_scaler.pkl"))
    if y_scaler is not None:
        joblib.dump(y_scaler, os.path.join(save_dir, "y_scaler.pkl"))

    eval_results: Dict[str, dict] = {}
    for split_name in ["train_eval", "val", "test"]:
        alias = "train" if split_name == "train_eval" else split_name
        loss_value, y_true_scaled, y_pred_scaled = run_one_epoch(
            model, loaders[split_name], loss_fn, cfg.train.device
        )
        y_true = inverse_transform_array(y_scaler, y_true_scaled)
        y_pred = inverse_transform_array(y_scaler, y_pred_scaled)
        pred_df = build_prediction_df(
            timestamp_sequences[alias],
            y_true,
            y_pred,
            cfg.data.horizon,
            cfg.data.datetime_col,
        )
        if alias == "test" and cfg.data.drop_duplicate_test_timestamps:
            pred_df = drop_duplicate_timestamps(pred_df, cfg.data.datetime_col)
        metrics = calc_prediction_df_metrics(pred_df, cfg.data.horizon)
        metrics["Loss"] = float(loss_value)
        eval_results[alias] = {
            "metrics": metrics,
            "pred_df": pred_df,
        }

    pd.DataFrame(epoch_rows).to_csv(
        os.path.join(save_dir, "epoch_logs.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_curves(train_losses, val_losses, val_r2_list, save_dir)

    for split_name in ["train", "val", "test"]:
        metrics = eval_results[split_name]["metrics"]
        pred_df = eval_results[split_name]["pred_df"]
        save_metrics_csv(metrics, os.path.join(save_dir, f"{split_name}_metrics.csv"))
        if split_name in ["val", "test"]:
            pred_df.to_csv(
                os.path.join(save_dir, f"{split_name}_predictions.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    plot_prediction_from_df(
        eval_results["test"]["pred_df"],
        os.path.join(save_dir, "test_prediction.png"),
        title=f"{experiment_name} Test Prediction",
        ylabel=ylabel,
    )

    print_metrics(eval_results["train"]["metrics"], f"{experiment_name} Train Metrics")
    print_metrics(eval_results["val"]["metrics"], f"{experiment_name} Validation Metrics")
    print_metrics(eval_results["test"]["metrics"], f"{experiment_name} Test Metrics")

    return {
        "train_metrics": eval_results["train"]["metrics"],
        "val_metrics": eval_results["val"]["metrics"],
        "test_metrics": eval_results["test"]["metrics"],
        "train_pred_df": eval_results["train"]["pred_df"],
        "val_pred_df": eval_results["val"]["pred_df"],
        "test_pred_df": eval_results["test"]["pred_df"],
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "feature_cols": feature_cols,
        "sample_counts": {
            "train_samples": len(eval_results["train"]["pred_df"]),
            "val_samples": len(eval_results["val"]["pred_df"]),
            "test_samples": len(eval_results["test"]["pred_df"]),
            **prepared["counts"],
        },
        "save_dir": save_dir,
    }


def combine_prediction_frames(
    lhs_df: pd.DataFrame,
    rhs_df: pd.DataFrame,
    horizon: int,
    operation: str = "subtract",
) -> pd.DataFrame:
    left = add_timestamp_occurrence_key(lhs_df.copy(), "timestamp")
    right = add_timestamp_occurrence_key(rhs_df.copy(), "timestamp")
    merged = pd.merge(
        left,
        right,
        on=["timestamp", "_timestamp_occurrence"],
        how="inner",
        suffixes=("_lhs", "_rhs"),
    )

    if len(merged) == 0:
        raise ValueError("No overlapping prediction timestamps were found while combining two models.")

    if operation == "subtract":
        sign = -1.0
    elif operation == "add":
        sign = 1.0
    else:
        raise ValueError('operation must be "subtract" or "add".')

    out = pd.DataFrame({"timestamp": merged["timestamp"]})
    for step in range(horizon):
        left_true = merged[f"y_true_step_{step + 1}_lhs"]
        left_pred = merged[f"y_pred_step_{step + 1}_lhs"]
        right_true = merged[f"y_true_step_{step + 1}_rhs"]
        right_pred = merged[f"y_pred_step_{step + 1}_rhs"]
        out[f"y_true_step_{step + 1}"] = left_true + sign * right_true
        out[f"y_pred_step_{step + 1}"] = left_pred + sign * right_pred

    return out.sort_values("timestamp").reset_index(drop=True)


def build_direct_cfg(base_cfg: Config, save_dir: str) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.data.target_col = cfg.data.net_load_col
    cfg.data.save_dir = save_dir
    cfg.feature.raw_feature_cols = [cfg.data.net_load_col]
    return cfg


def build_gc_cfg(base_cfg: Config, save_dir: str) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.data.target_col = "gc"
    cfg.data.save_dir = save_dir
    cfg.feature.raw_feature_cols = ["gc"]
    return cfg


def build_gg_cfg(base_cfg: Config, save_dir: str) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.data.target_col = "gg"
    cfg.data.save_dir = save_dir
    cfg.feature.raw_feature_cols = ["gg"]
    cfg.feature.use_slot_sin_cos = False
    cfg.feature.use_weekday_sin_cos = False
    cfg.feature.use_month_sin_cos = False
    cfg.feature.use_is_weekend = False
    cfg.feature.use_is_holiday = False
    return cfg


def run_direct_method(base_cfg: Config, common_df: pd.DataFrame, save_dir: str) -> dict:
    direct_cfg = build_direct_cfg(base_cfg, save_dir)
    return run_single_experiment(
        direct_cfg,
        common_df,
        save_dir=save_dir,
        experiment_name="Direct Net Load",
        ylabel="Net Load",
    )


def run_indirect_method(base_cfg: Config, common_df: pd.DataFrame, save_dir: str) -> dict:
    ensure_dir(save_dir)
    save_config(base_cfg, save_dir)

    gc_save_dir = os.path.join(save_dir, "separate_gc")
    gg_save_dir = os.path.join(save_dir, "separate_gg")

    gc_cfg = build_gc_cfg(base_cfg, gc_save_dir)
    gg_cfg = build_gg_cfg(base_cfg, gg_save_dir)

    gc_result = run_single_experiment(
        gc_cfg,
        common_df,
        save_dir=gc_save_dir,
        experiment_name="Indirect GC",
        ylabel="GC",
    )
    gg_result = run_single_experiment(
        gg_cfg,
        common_df,
        save_dir=gg_save_dir,
        experiment_name="Indirect GG",
        ylabel="GG",
    )

    combined_pred = {}
    combined_metrics = {}
    for split_name in ["train", "val", "test"]:
        pred_df = combine_prediction_frames(
            gc_result[f"{split_name}_pred_df"],
            gg_result[f"{split_name}_pred_df"],
            horizon=base_cfg.data.horizon,
            operation="subtract",
        )
        if split_name == "test" and base_cfg.data.drop_duplicate_test_timestamps:
            pred_df = drop_duplicate_timestamps(pred_df, base_cfg.data.datetime_col)
        combined_pred[split_name] = pred_df
        combined_metrics[split_name] = calc_prediction_df_metrics(pred_df, base_cfg.data.horizon)

    for split_name in ["train", "val", "test"]:
        save_metrics_csv(combined_metrics[split_name], os.path.join(save_dir, f"{split_name}_metrics.csv"))

    combined_pred["val"].to_csv(
        os.path.join(save_dir, "val_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    combined_pred["test"].to_csv(
        os.path.join(save_dir, "test_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    component_rows = []
    for component_name, result in [("gc", gc_result), ("gg", gg_result)]:
        for split_name in ["train", "val", "test"]:
            row = {
                "component": component_name,
                "split": split_name,
                **result[f"{split_name}_metrics"],
            }
            component_rows.append(row)
    pd.DataFrame(component_rows).to_csv(
        os.path.join(save_dir, "component_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_prediction_from_df(
        combined_pred["test"],
        os.path.join(save_dir, "test_prediction.png"),
        title="Indirect Net Load Test Prediction",
        ylabel="Net Load",
    )

    print_metrics(combined_metrics["train"], "Indirect Net Load Train Metrics")
    print_metrics(combined_metrics["val"], "Indirect Net Load Validation Metrics")
    print_metrics(combined_metrics["test"], "Indirect Net Load Test Metrics")

    return {
        "train_metrics": combined_metrics["train"],
        "val_metrics": combined_metrics["val"],
        "test_metrics": combined_metrics["test"],
        "train_pred_df": combined_pred["train"],
        "val_pred_df": combined_pred["val"],
        "test_pred_df": combined_pred["test"],
        "best_epoch": None,
        "gc_best_epoch": gc_result["best_epoch"],
        "gg_best_epoch": gg_result["best_epoch"],
        "sample_counts": {
            "train_samples": len(combined_pred["train"]),
            "val_samples": len(combined_pred["val"]),
            "test_samples": len(combined_pred["test"]),
        },
        "component_results": {
            "gc": gc_result,
            "gg": gg_result,
        },
        "save_dir": save_dir,
    }


def run_dataset_suite(base_cfg: Config, common_df: pd.DataFrame, dataset_name: str, dataset_save_dir: str) -> dict:
    ensure_dir(dataset_save_dir)
    results: Dict[str, dict] = {}
    method_mode = base_cfg.experiment.method_mode.lower()

    print(f"\n========== Dataset: {dataset_name} ==========")
    if method_mode in ["direct", "both"]:
        results["direct"] = run_direct_method(
            base_cfg,
            common_df,
            save_dir=os.path.join(dataset_save_dir, "direct"),
        )

    if method_mode in ["indirect", "both"]:
        results["indirect"] = run_indirect_method(
            base_cfg,
            common_df,
            save_dir=os.path.join(dataset_save_dir, "indirect"),
        )

    if method_mode == "both":
        compare_df = build_method_compare_df(
            results["direct"]["test_pred_df"],
            results["indirect"]["test_pred_df"],
        )
        compare_df.to_csv(
            os.path.join(dataset_save_dir, "net_load_method_compare.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        plot_netload_method_compare(
            results["direct"]["test_pred_df"],
            results["indirect"]["test_pred_df"],
            os.path.join(dataset_save_dir, "net_load_method_compare.png"),
        )

    return results


def derive_dataset_name_from_path(path: str) -> str:
    basename = os.path.basename(path)
    suffix = "_load_weather_30min.csv"
    if basename.endswith(suffix):
        return basename[: -len(suffix)]
    return os.path.splitext(basename)[0]


def append_summary_rows(summary_rows: List[dict], scope: str, dataset_name: str, results: Dict[str, dict]) -> None:
    for method_name, result in results.items():
        row = {
            "scope": scope,
            "dataset": dataset_name,
            "method": method_name,
            "best_epoch": result.get("best_epoch"),
            "gc_best_epoch": result.get("gc_best_epoch"),
            "gg_best_epoch": result.get("gg_best_epoch"),
        }

        for split_name in ["train", "val", "test"]:
            metrics = result[f"{split_name}_metrics"]
            for metric_name, metric_value in metrics.items():
                row[f"{split_name}_{metric_name}"] = metric_value

        for count_name, count_value in result.get("sample_counts", {}).items():
            row[count_name] = count_value

        summary_rows.append(row)


def save_summary_tables(summary_rows: List[dict], save_dir: str, prefix: str) -> None:
    if not summary_rows:
        return

    ensure_dir(save_dir)
    df = pd.DataFrame(summary_rows).sort_values(["scope", "dataset", "method"]).reset_index(drop=True)
    long_path = os.path.join(save_dir, f"{prefix}.csv")
    df.to_csv(long_path, index=False, encoding="utf-8-sig")

    value_cols = [col for col in df.columns if col not in ["scope", "dataset", "method"]]
    wide_df = df.pivot(index=["scope", "dataset"], columns="method", values=value_cols).reset_index()
    wide_df.columns = flatten_columns(wide_df.columns)
    wide_path = os.path.join(save_dir, f"{prefix}_wide.csv")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")


def main():
    base_cfg = deepcopy(CFG)
    set_seed(base_cfg.train.random_seed)
    ensure_dir(base_cfg.data.save_dir)
    save_config(base_cfg, base_cfg.data.save_dir)

    run_scope = base_cfg.experiment.run_scope.lower()
    method_mode = base_cfg.experiment.method_mode.lower()
    valid_scopes = {"per_client", "aggregate", "both"}
    valid_methods = {"direct", "indirect", "both"}

    if run_scope not in valid_scopes:
        raise ValueError(f"Invalid run_scope: {base_cfg.experiment.run_scope}. Options: {sorted(valid_scopes)}")
    if method_mode not in valid_methods:
        raise ValueError(
            f"Invalid method_mode: {base_cfg.experiment.method_mode}. Options: {sorted(valid_methods)}"
        )

    print(f"Device: {base_cfg.train.device}")
    print(f"Run scope: {run_scope}")
    print(f"Method mode: {method_mode}")
    print(f"Output dir: {base_cfg.data.save_dir}")

    all_summary_rows: List[dict] = []
    per_client_summary_rows: List[dict] = []
    aggregate_summary_rows: List[dict] = []

    if run_scope in ["per_client", "both"]:
        for path in base_cfg.data.client_files:
            dataset_name = derive_dataset_name_from_path(path)
            common_df = load_common_df(path, base_cfg)
            dataset_save_dir = os.path.join(base_cfg.data.save_dir, "per_client", dataset_name)
            results = run_dataset_suite(base_cfg, common_df, dataset_name, dataset_save_dir)
            append_summary_rows(per_client_summary_rows, "per_client", dataset_name, results)
            append_summary_rows(all_summary_rows, "per_client", dataset_name, results)

        save_summary_tables(
            per_client_summary_rows,
            os.path.join(base_cfg.data.save_dir, "per_client"),
            "per_client_test_metrics",
        )

    if run_scope in ["aggregate", "both"]:
        aggregate_df = build_aggregate_common_df(base_cfg.data.client_files, base_cfg)
        aggregate_name = "aggregate_total"
        aggregate_save_dir = os.path.join(base_cfg.data.save_dir, aggregate_name)
        results = run_dataset_suite(base_cfg, aggregate_df, aggregate_name, aggregate_save_dir)
        append_summary_rows(aggregate_summary_rows, "aggregate", aggregate_name, results)
        append_summary_rows(all_summary_rows, "aggregate", aggregate_name, results)
        save_summary_tables(aggregate_summary_rows, aggregate_save_dir, "aggregate_test_metrics")

    save_summary_tables(all_summary_rows, base_cfg.data.save_dir, "all_test_metrics")

    print("\nAll experiments finished.")
    print(f"Results saved to: {base_cfg.data.save_dir}")


if __name__ == "__main__":
    main()
