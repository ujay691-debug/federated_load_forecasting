import os
import json
import random
import joblib
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore", message=".*padding='same'.*")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 配置区
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs_cnn_lstm_netload_multi_scope")
PER_CLIENT_DIR = os.path.join(PROJECT_ROOT, "per_client_merged")


def build_default_client_files() -> List[str]:
    return [os.path.join(PER_CLIENT_DIR, f"client_{i}_load_weather_30min.csv") for i in range(1, 10)]


@dataclass
class DataConfig:
    client_files: List[str] = field(default_factory=build_default_client_files)
    datetime_col: str = "timestamp"

    seq_len: int = 48
    horizon: int = 1

    train_ratio: float = 0.8
    val_ratio: float = 0.1

    dropna: bool = True
    sort_by_time: bool = True

    freq_minutes: str = "auto"
    save_dir: str = os.path.join(RUNS_DIR, "netload_multi_scope_results")

    use_time_range: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    aggregate_weather_mode: str = "mean"  # mean / first


@dataclass
class FeatureConfig:
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
    selected_client_ids: List[int] = field(default_factory=lambda: [1,2,3,4,5,6,7,8,9])

    # 关闭单客户端分别训练
    run_per_client: bool = True

    # 关闭“各客户端分别预测后再求和”的区域评估
    run_regional_sum_from_clients: bool = True

    # 开启“九客户端先聚合成总数据集再训练”
    run_aggregated_dataset: bool = False

    # 可选: "direct" / "indirect" / "both"
    method_mode: str = "both"

    # 聚合天气方式
    aggregate_weather_mode: str = "mean"
    show_n: int = 200

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


CFG = Config()


# ============================================================
# 工具函数
# ============================================================
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
        raise ValueError("无法从 datetime 推断分辨率，时间列样本不足。")
    return int(diffs.mode().iloc[0].total_seconds() // 60)


def get_slots_per_day(freq_minutes: int) -> int:
    if freq_minutes <= 0 or 1440 % freq_minutes != 0:
        raise ValueError(f"freq_minutes={freq_minutes} 不合法，无法整除一天。")
    return 1440 // freq_minutes


def get_scaler(name: str):
    name = name.lower()
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    if name == "none":
        return None
    raise ValueError(f"不支持的 scaler 类型: {name}")


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
    for k, v in metrics.items():
        if "percent" in k.lower():
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.6f}")


def inverse_transform_array(scaler, arr: np.ndarray) -> np.ndarray:
    if scaler is None:
        return arr
    original_shape = arr.shape
    arr_2d = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    restored = scaler.inverse_transform(arr_2d)
    return restored.reshape(original_shape)


def save_metrics_csv(metrics: dict, save_path: str) -> None:
    pd.DataFrame([metrics]).to_csv(save_path, index=False, encoding="utf-8-sig")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gc" not in df.columns:
        raise ValueError("数据中缺少 gc 列，无法构造净负荷。")
    if "gg" not in df.columns:
        raise ValueError("数据中缺少 gg 列，无法构造净负荷。")
    df["net_load"] = df["gc"].astype(float) - df["gg"].astype(float)
    return df


def add_timestamp_occurrence_key(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    # Preserve repeated wall-clock timestamps such as DST fallback hours.
    out["_timestamp_occurrence"] = out.groupby(timestamp_col).cumcount()
    return out


def get_temp_source_col(df: pd.DataFrame, fc: FeatureConfig) -> Optional[str]:
    mode = fc.temp_source_mode.lower()
    if mode == "auto":
        if fc.temp_c_col in df.columns:
            return fc.temp_c_col
        if fc.temp_k_col in df.columns:
            return fc.temp_k_col
        raise ValueError(f"已启用温度特征，但 {fc.temp_c_col} 和 {fc.temp_k_col} 都不存在。")
    if mode == "c":
        if fc.temp_c_col not in df.columns:
            raise ValueError(f"缺少列: {fc.temp_c_col}")
        return fc.temp_c_col
    if mode == "k":
        if fc.temp_k_col not in df.columns:
            raise ValueError(f"缺少列: {fc.temp_k_col}")
        return fc.temp_k_col
    raise ValueError('temp_source_mode 只支持 "auto" / "c" / "k"')


def apply_time_filter(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    dt_col = cfg.data.datetime_col
    df[dt_col] = pd.to_datetime(df[dt_col])
    if cfg.data.sort_by_time:
        df = df.sort_values(dt_col).reset_index(drop=True)
    if cfg.data.use_time_range:
        if cfg.data.start_time is not None:
            df = df[df[dt_col] >= pd.to_datetime(cfg.data.start_time)]
        if cfg.data.end_time is not None:
            df = df[df[dt_col] <= pd.to_datetime(cfg.data.end_time)]
        df = df.reset_index(drop=True)
    return df


def prepare_common_df_from_path(data_path: str, cfg: Config) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    df = pd.read_csv(data_path)
    return prepare_common_df_from_df(df, cfg)


def prepare_common_df_from_df(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_derived_columns(df)
    dt_col = cfg.data.datetime_col
    if dt_col not in df.columns:
        raise ValueError(f"数据中缺少时间列: {dt_col}")

    df = apply_time_filter(df, cfg)

    required_cols = [dt_col, "gc", "gg", "net_load"]
    if cfg.feature.use_temp_c:
        required_cols.append(get_temp_source_col(df, cfg.feature))
    if cfg.feature.use_rh:
        if cfg.feature.rh_col not in df.columns:
            raise ValueError(f"缺少列: {cfg.feature.rh_col}")
        required_cols.append(cfg.feature.rh_col)
    if cfg.feature.use_wind:
        if cfg.feature.wind_col not in df.columns:
            raise ValueError(f"缺少列: {cfg.feature.wind_col}")
        required_cols.append(cfg.feature.wind_col)
    if cfg.feature.use_ghi:
        if cfg.feature.ghi_col not in df.columns:
            raise ValueError(f"缺少列: {cfg.feature.ghi_col}")
        required_cols.append(cfg.feature.ghi_col)

    if cfg.data.dropna:
        df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def build_aggregated_common_df(client_paths: List[str], cfg: Config) -> pd.DataFrame:
    dt_col = cfg.data.datetime_col
    weather_cols = []

    temp_source_col_name = None
    if cfg.feature.use_temp_c:
        probe_df = pd.read_csv(client_paths[0], nrows=5)
        temp_source_col_name = get_temp_source_col(probe_df, cfg.feature)
        weather_cols.append(temp_source_col_name)
    if cfg.feature.use_rh:
        weather_cols.append(cfg.feature.rh_col)
    if cfg.feature.use_wind:
        weather_cols.append(cfg.feature.wind_col)
    if cfg.feature.use_ghi:
        weather_cols.append(cfg.feature.ghi_col)

    weather_cols = list(dict.fromkeys(weather_cols))

    merged = None
    for idx, path in enumerate(client_paths, start=1):
        df = pd.read_csv(path)
        df = prepare_common_df_from_df(df, cfg)
        keep_cols = [dt_col, "gc", "gg", "net_load"] + [c for c in weather_cols if c in df.columns]
        tmp = df[keep_cols].copy()

        rename_map = {
            "gc": f"gc_client_{idx}",
            "gg": f"gg_client_{idx}",
            "net_load": f"net_load_client_{idx}",
        }
        for c in weather_cols:
            if c in tmp.columns:
                rename_map[c] = f"{c}_client_{idx}"
        tmp = tmp.rename(columns=rename_map)
        tmp = add_timestamp_occurrence_key(tmp, dt_col)

        if merged is None:
            merged = tmp
        else:
            merged = pd.merge(merged, tmp, on=[dt_col, "_timestamp_occurrence"], how="inner")

    if merged is None or len(merged) == 0:
        raise ValueError("聚合后的总数据为空，请检查各客户端时间戳。")

    out = pd.DataFrame({dt_col: merged[dt_col].copy()})
    gc_cols = [c for c in merged.columns if c.startswith("gc_client_")]
    gg_cols = [c for c in merged.columns if c.startswith("gg_client_")]
    net_cols = [c for c in merged.columns if c.startswith("net_load_client_")]

    out["gc"] = merged[gc_cols].sum(axis=1)
    out["gg"] = merged[gg_cols].sum(axis=1)
    out["net_load"] = merged[net_cols].sum(axis=1)

    for c in weather_cols:
        cand = [col for col in merged.columns if col.startswith(f"{c}_client_")]
        if not cand:
            continue
        if cfg.data.aggregate_weather_mode.lower() == "first":
            out[c] = merged[cand[0]].values
        else:
            out[c] = merged[cand].mean(axis=1)

    out = out.sort_values(dt_col).reset_index(drop=True)
    return out


def compute_split_indices(n: int, cfg: Config) -> Tuple[int, int]:
    train_end = int(n * cfg.data.train_ratio)
    val_end = int(n * (cfg.data.train_ratio + cfg.data.val_ratio))
    if train_end <= cfg.data.seq_len or val_end <= train_end:
        raise ValueError("训练/验证划分太小，无法形成有效序列。")
    return train_end, val_end


# ============================================================
# 可视化
# ============================================================
def plot_curves(epoch_df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_df["epoch"], epoch_df["train_loss"], label="训练集Loss")
    plt.plot(epoch_df["epoch"], epoch_df["val_loss"], label="验证集Loss")
    plt.title("训练/验证 Loss 曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_df["epoch"], epoch_df["val_r2"], label="验证集R2")
    plt.title("验证集 R2 曲线")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_r2_curve.png"), dpi=200)
    plt.close()


def plot_prediction(true_real, pred_real, save_path: str, title: str, ylabel: str, show_n: int = 200):
    true_real = np.asarray(true_real)
    pred_real = np.asarray(pred_real)
    show_n = min(show_n, len(true_real))

    plt.figure(figsize=(12, 5))
    if true_real.ndim == 2 and true_real.shape[1] > 1:
        plt.plot(true_real[:show_n, 0], label=f"真实{ylabel}(第1步)")
        plt.plot(pred_real[:show_n, 0], label=f"预测{ylabel}(第1步)")
    else:
        plt.plot(np.squeeze(true_real[:show_n]), label=f"真实{ylabel}")
        plt.plot(np.squeeze(pred_real[:show_n]), label=f"预测{ylabel}")
    plt.title(title)
    plt.xlabel("样本点")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_netload_method_compare(pred_direct_df: pd.DataFrame, pred_indirect_df: pd.DataFrame, save_path: str, title: str, show_n: int = 200):
    merged = pred_direct_df[["timestamp", "y_true", "y_pred"]].merge(
        pred_indirect_df[["timestamp", "y_true", "y_pred"]],
        on="timestamp",
        suffixes=("_direct", "_indirect"),
    )
    show_n = min(show_n, len(merged))

    plt.figure(figsize=(12, 5))
    plt.plot(merged["y_true_direct"].values[:show_n], label="真实净负荷")
    plt.plot(merged["y_pred_direct"].values[:show_n], label="直接法预测净负荷")
    plt.plot(merged["y_pred_indirect"].values[:show_n], label="间接法预测净负荷")
    plt.title(title)
    plt.xlabel("样本点")
    plt.ylabel("net_load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# 模型
# ============================================================
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
        attn_vector = torch.tanh(self.attn_out(pre_activation))
        return attn_vector


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
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


# ============================================================
# 数据处理
# ============================================================
class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_features(df: pd.DataFrame, cfg: Config, target_col: str, history_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    dc = cfg.data
    fc = cfg.feature

    if dc.datetime_col not in df.columns:
        raise ValueError(f"数据中缺少时间列: {dc.datetime_col}")

    df = apply_time_filter(df, cfg)

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
                raise ValueError(f"已启用 temp_c，但数据中既没有 {fc.temp_c_col}，也没有 {fc.temp_k_col}")
        elif temp_mode == "c":
            if fc.temp_c_col not in df.columns:
                raise ValueError(f"缺少列 {fc.temp_c_col}")
            df["temp_c"] = df[fc.temp_c_col].astype(float)
        elif temp_mode == "k":
            if fc.temp_k_col not in df.columns:
                raise ValueError(f"缺少列 {fc.temp_k_col}")
            df["temp_c"] = df[fc.temp_k_col].astype(float) - 273.15
        else:
            raise ValueError('temp_source_mode 只支持 "auto" / "c" / "k"')

    if fc.use_rh:
        if fc.rh_col not in df.columns:
            raise ValueError(f"已启用 rh2m_pct，但数据中缺少列 {fc.rh_col}")
        df["rh2m_pct"] = df[fc.rh_col].clip(lower=0, upper=100)

    if fc.use_wind:
        if fc.wind_col not in df.columns:
            raise ValueError(f"已启用 wind10m_ms，但数据中缺少列 {fc.wind_col}")
        df["wind10m_ms"] = df[fc.wind_col].clip(lower=0)

    if fc.use_ghi:
        if fc.ghi_col not in df.columns:
            raise ValueError(f"已启用 ghi_wm2，但数据中缺少列 {fc.ghi_col}")
        df["ghi_wm2"] = df[fc.ghi_col].astype(float)

    if fc.use_apparent_temp:
        needed = ["temp_c", "rh2m_pct", "wind10m_ms"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"已启用 apparent_temp_c，但缺少列: {missing}")
        e = (df["rh2m_pct"] / 100.0) * 6.105 * np.exp(17.27 * df["temp_c"] / (237.7 + df["temp_c"]))
        df["apparent_temp_c"] = df["temp_c"] + 0.33 * e - 0.70 * df["wind10m_ms"] - 4.0

    feature_cols: List[str] = []
    for c in history_cols:
        if c not in feature_cols:
            feature_cols.append(c)

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

    missing_required = [c for c in [target_col] + feature_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"数据中缺少这些必要列: {missing_required}")

    if dc.dropna:
        df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    return df, feature_cols


def split_df_by_fixed_indices(df: pd.DataFrame, train_end: int, val_end: int):
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def fit_and_transform_x(train_df, val_df, test_df, feature_cols, cfg: Config):
    no_scale_cols = set(cfg.feature.no_scale_cols)
    scale_cols = [c for c in feature_cols if c not in no_scale_cols]
    keep_cols = [c for c in feature_cols if c in no_scale_cols]

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


def fit_and_transform_y(train_df, val_df, test_df, target_col: str, cfg: Config):
    y_scaler = get_scaler(cfg.train.scaler_y)

    train_y = train_df[[target_col]].values
    val_y = val_df[[target_col]].values
    test_y = test_df[[target_col]].values

    if y_scaler is not None:
        train_y = y_scaler.fit_transform(train_y)
        val_y = y_scaler.transform(val_y)
        test_y = y_scaler.transform(test_y)

    return train_y, val_y, test_y, y_scaler


def create_sequences(feature_array: np.ndarray, target_array: np.ndarray, timestamp_array, seq_len: int, horizon: int):
    xs, ys, ts = [], [], []
    total_len = len(feature_array)
    for end_idx in range(seq_len, total_len - horizon + 1):
        start_idx = end_idx - seq_len
        x = feature_array[start_idx:end_idx, :]
        y = target_array[end_idx:end_idx + horizon, 0]
        label_ts = timestamp_array[end_idx:end_idx + horizon]
        xs.append(x)
        ys.append(y)
        ts.append(label_ts)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    ts = np.asarray(ts)

    if horizon == 1:
        ys = ys.reshape(-1, 1)
        ts = ts.reshape(-1, 1)
    return xs, ys, ts


def make_dataloader(x_seq, y_seq, cfg: Config, shuffle: bool):
    return DataLoader(
        SeqDataset(x_seq, y_seq),
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and str(cfg.train.device).startswith("cuda"),
    )


def prepare_dataset(common_df: pd.DataFrame, cfg: Config, target_col: str, history_cols: List[str]) -> Dict:
    df, feature_cols = build_features(common_df, cfg, target_col=target_col, history_cols=history_cols)
    train_end, val_end = compute_split_indices(len(df), cfg)
    train_df, val_df, test_df = split_df_by_fixed_indices(df, train_end, val_end)

    train_scaled_df, val_scaled_df, test_scaled_df, x_scaler, scale_cols, keep_cols = fit_and_transform_x(
        train_df, val_df, test_df, feature_cols, cfg
    )
    y_train_scaled, y_val_scaled, y_test_scaled, y_scaler = fit_and_transform_y(train_df, val_df, test_df, target_col, cfg)

    x_train = train_scaled_df[feature_cols].values.astype(np.float32)
    x_val = val_scaled_df[feature_cols].values.astype(np.float32)
    x_test = test_scaled_df[feature_cols].values.astype(np.float32)

    ts_train = pd.to_datetime(train_df[cfg.data.datetime_col]).values
    ts_val = pd.to_datetime(val_df[cfg.data.datetime_col]).values
    ts_test = pd.to_datetime(test_df[cfg.data.datetime_col]).values

    x_train_seq, y_train_seq, ts_train_seq = create_sequences(x_train, y_train_scaled, ts_train, cfg.data.seq_len, cfg.data.horizon)
    x_val_seq, y_val_seq, ts_val_seq = create_sequences(x_val, y_val_scaled, ts_val, cfg.data.seq_len, cfg.data.horizon)
    x_test_seq, y_test_seq, ts_test_seq = create_sequences(x_test, y_test_scaled, ts_test, cfg.data.seq_len, cfg.data.horizon)

    return {
        "feature_cols": feature_cols,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "scale_cols": scale_cols,
        "keep_cols": keep_cols,
        "train_loader": make_dataloader(x_train_seq, y_train_seq, cfg, shuffle=True),
        "val_loader": make_dataloader(x_val_seq, y_val_seq, cfg, shuffle=False),
        "test_loader": make_dataloader(x_test_seq, y_test_seq, cfg, shuffle=False),
        "train_timestamps": ts_train_seq,
        "val_timestamps": ts_val_seq,
        "test_timestamps": ts_test_seq,
        "train_samples": len(x_train_seq),
        "val_samples": len(x_val_seq),
        "test_samples": len(x_test_seq),
        "df": df,
    }


# ============================================================
# 训练与评估
# ============================================================
def get_loss_fn(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    raise ValueError(f"不支持的损失函数: {loss_name}")


def get_optimizer(optimizer_name: str, model: nn.Module, lr: float):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"不支持的优化器: {optimizer_name}")


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    count = 0
    preds_all = []
    trues_all = []

    with torch.enable_grad() if train else torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if train:
                optimizer.zero_grad()

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            if train:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * batch_x.size(0)
            count += batch_x.size(0)

            preds_all.append(pred.detach().cpu().numpy())
            trues_all.append(batch_y.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    avg_loss = loss_sum / max(count, 1)
    return avg_loss, preds_all, trues_all


def train_and_test_one_target(
    cfg: Config,
    common_df: pd.DataFrame,
    target_col: str,
    history_cols: List[str],
    save_dir: str,
    exp_name: str,
) -> Dict:
    ensure_dir(save_dir)
    data = prepare_dataset(common_df, cfg, target_col=target_col, history_cols=history_cols)

    device = torch.device(cfg.train.device)
    model = CNNLSTMModel(
        input_dim=len(data["feature_cols"]),
        output_dim=cfg.data.horizon,
        cfg=cfg.model,
    ).to(device)

    criterion = get_loss_fn(cfg.train.loss_name)
    optimizer = get_optimizer(cfg.train.optimizer_name, model, cfg.train.lr)

    best_val_loss = float("inf")
    epoch_logs = []
    model_path = os.path.join(save_dir, "best_model.pth")

    if data["x_scaler"] is not None:
        joblib.dump(data["x_scaler"], os.path.join(save_dir, "x_scaler.save"))
    if data["y_scaler"] is not None:
        joblib.dump(data["y_scaler"], os.path.join(save_dir, "y_scaler.save"))

    print("=" * 100)
    print(f"开始训练: {exp_name}")
    print(f"target_col = {target_col}")
    print(f"history_cols = {history_cols}")
    print(f"device = {cfg.train.device}")
    print(f"feature_cols = {data['feature_cols']}")
    print(f"train/val/test samples = {data['train_samples']}/{data['val_samples']}/{data['test_samples']}")

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, _, _ = run_one_epoch(model, data["train_loader"], criterion, optimizer, device, train=True)
        val_loss, val_pred_scaled, val_true_scaled = run_one_epoch(model, data["val_loader"], criterion, optimizer, device, train=False)

        val_pred_real = inverse_transform_array(data["y_scaler"], val_pred_scaled)
        val_true_real = inverse_transform_array(data["y_scaler"], val_true_scaled)
        val_r2 = r2_score(val_true_real.reshape(-1), val_pred_real.reshape(-1))

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_r2": float(val_r2),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        print(
            f"Epoch [{epoch:03d}/{cfg.train.epochs}] | "
            f"TrainLoss: {train_loss:.6f} | "
            f"ValLoss: {val_loss:.6f} | "
            f"ValR2: {val_r2:.6f}"
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_pred_scaled, test_true_scaled = run_one_epoch(model, data["test_loader"], criterion, optimizer, device, train=False)

    pred_real = inverse_transform_array(data["y_scaler"], test_pred_scaled)
    true_real = inverse_transform_array(data["y_scaler"], test_true_scaled)
    metrics = calc_metrics(true_real.reshape(-1), pred_real.reshape(-1))
    metrics["test_loss"] = float(test_loss)

    print_metrics(metrics, title=f"{exp_name} Test Metrics")

    pred_df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["test_timestamps"][:, 0]),
        "y_true": true_real.reshape(-1),
        "y_pred": pred_real.reshape(-1),
    }).sort_values("timestamp").reset_index(drop=True)

    pred_df.to_csv(os.path.join(save_dir, f"{exp_name}_test_predictions.csv"), index=False, encoding="utf-8-sig")
    save_metrics_csv(metrics, os.path.join(save_dir, f"{exp_name}_test_metrics.csv"))

    epoch_df = pd.DataFrame(epoch_logs)
    epoch_df.to_csv(os.path.join(save_dir, f"{exp_name}_epoch_logs.csv"), index=False, encoding="utf-8-sig")
    plot_curves(epoch_df, save_dir)
    plot_prediction(
        true_real,
        pred_real,
        save_path=os.path.join(save_dir, f"{exp_name}_test_prediction.png"),
        title=f"{exp_name} 测试集预测效果",
        ylabel=target_col,
        show_n=cfg.experiment.show_n,
    )

    return {
        "pred_df": pred_df,
        "metrics": metrics,
        "epoch_df": epoch_df,
        "feature_cols": data["feature_cols"],
    }


def build_indirect_netload_predictions(pred_gc_df: pd.DataFrame, pred_gg_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    left = add_timestamp_occurrence_key(pred_gc_df, "timestamp")
    right = add_timestamp_occurrence_key(pred_gg_df, "timestamp")
    merged = left.merge(right, on=["timestamp", "_timestamp_occurrence"], suffixes=("_gc", "_gg"))
    out = pd.DataFrame({
        "timestamp": merged["timestamp"],
        "y_true": merged["y_true_gc"] - merged["y_true_gg"],
        "y_pred": merged["y_pred_gc"] - merged["y_pred_gg"],
    }).sort_values("timestamp").reset_index(drop=True)
    metrics = calc_metrics(out["y_true"].values, out["y_pred"].values)
    return out, metrics


def run_direct_method(cfg: Config, common_df: pd.DataFrame, save_dir: str, prefix: str) -> Dict:
    direct_dir = os.path.join(save_dir, "direct_net_load")
    return train_and_test_one_target(
        cfg=cfg,
        common_df=common_df,
        target_col="net_load",
        history_cols=["net_load"],
        save_dir=direct_dir,
        exp_name=f"{prefix}_direct_net_load",
    )


def run_indirect_method(cfg: Config, common_df: pd.DataFrame, save_dir: str, prefix: str) -> Dict:
    gc_dir = os.path.join(save_dir, "indirect_gc")
    gg_dir = os.path.join(save_dir, "indirect_gg")
    out_dir = os.path.join(save_dir, "indirect_net_load")
    ensure_dir(out_dir)

    gc_result = train_and_test_one_target(
        cfg=cfg,
        common_df=common_df,
        target_col="gc",
        history_cols=["gc"],
        save_dir=gc_dir,
        exp_name=f"{prefix}_indirect_gc",
    )
    gg_result = train_and_test_one_target(
        cfg=cfg,
        common_df=common_df,
        target_col="gg",
        history_cols=["gg"],
        save_dir=gg_dir,
        exp_name=f"{prefix}_indirect_gg",
    )

    pred_indirect_df, indirect_metrics = build_indirect_netload_predictions(gc_result["pred_df"], gg_result["pred_df"])
    pred_indirect_df.to_csv(
        os.path.join(out_dir, f"{prefix}_indirect_net_load_test_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_metrics_csv(indirect_metrics, os.path.join(out_dir, f"{prefix}_indirect_net_load_test_metrics.csv"))
    plot_prediction(
        pred_indirect_df["y_true"].values,
        pred_indirect_df["y_pred"].values,
        save_path=os.path.join(out_dir, f"{prefix}_indirect_net_load_test_prediction.png"),
        title=f"{prefix} 间接法净负荷测试集预测效果",
        ylabel="net_load",
        show_n=cfg.experiment.show_n,
    )

    return {
        "gc_result": gc_result,
        "gg_result": gg_result,
        "pred_df": pred_indirect_df,
        "metrics": indirect_metrics,
    }


def normalize_method_mode(method_mode: str) -> str:
    mode = method_mode.lower()
    valid = {"direct", "indirect", "both"}
    if mode not in valid:
        raise ValueError(f"method_mode 只支持 {valid}，当前为 {method_mode}")
    return mode


def merge_regional_predictions(pred_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if len(pred_dfs) == 0:
        raise ValueError("pred_dfs 为空，无法聚合区域预测。")

    merged = None
    for idx, df in enumerate(pred_dfs, start=1):
        tmp = add_timestamp_occurrence_key(df.copy(), "timestamp").rename(columns={
            "y_true": f"y_true_client_{idx}",
            "y_pred": f"y_pred_client_{idx}",
        })
        if merged is None:
            merged = tmp
        else:
            merged = pd.merge(merged, tmp, on=["timestamp", "_timestamp_occurrence"], how="inner")

    if merged is None or len(merged) == 0:
        raise ValueError("各客户端测试集时间戳无法对齐，无法做区域聚合。")

    true_cols = [c for c in merged.columns if c.startswith("y_true_client_")]
    pred_cols = [c for c in merged.columns if c.startswith("y_pred_client_")]

    out = pd.DataFrame({
        "timestamp": merged["timestamp"],
        "y_true": merged[true_cols].sum(axis=1),
        "y_pred": merged[pred_cols].sum(axis=1),
    }).sort_values("timestamp").reset_index(drop=True)
    return out


def save_method_compare(direct_result: Optional[Dict], indirect_result: Optional[Dict], save_dir: str, prefix: str, show_n: int):
    rows = []
    if direct_result is not None:
        rows.append({"method": "direct_net_load", **direct_result["metrics"]})
    if indirect_result is not None:
        rows.append({"method": "indirect_gc_minus_gg", **indirect_result["metrics"]})

    if rows:
        compare_df = pd.DataFrame(rows)
        compare_df.to_csv(os.path.join(save_dir, f"{prefix}_method_compare.csv"), index=False, encoding="utf-8-sig")

    if direct_result is not None and indirect_result is not None:
        plot_netload_method_compare(
            direct_result["pred_df"],
            indirect_result["pred_df"],
            save_path=os.path.join(save_dir, f"{prefix}_method_compare.png"),
            title=f"{prefix} 两种净负荷预测方法对比",
            show_n=show_n,
        )


# ============================================================
# 多场景运行
# ============================================================
def get_selected_client_paths(cfg: Config) -> List[Tuple[int, str]]:
    pairs = []
    for cid in cfg.experiment.selected_client_ids:
        if cid < 1 or cid > len(cfg.data.client_files):
            raise ValueError(f"客户端编号 {cid} 超出范围。")
        pairs.append((cid, cfg.data.client_files[cid - 1]))
    return pairs


def run_per_client_experiments(cfg: Config) -> Dict[str, List[Dict]]:
    mode = normalize_method_mode(cfg.experiment.method_mode)
    out_root = os.path.join(cfg.data.save_dir, "per_client_results")
    ensure_dir(out_root)

    direct_summary_rows = []
    indirect_summary_rows = []
    direct_pred_dfs = []
    indirect_pred_dfs = []

    for client_id, path in get_selected_client_paths(cfg):
        client_name = f"client_{client_id}"
        client_dir = os.path.join(out_root, client_name)
        ensure_dir(client_dir)
        common_df = prepare_common_df_from_path(path, cfg)

        direct_result = None
        indirect_result = None

        if mode in {"direct", "both"}:
            direct_result = run_direct_method(cfg, common_df, client_dir, prefix=client_name)
            direct_summary_rows.append({"client_id": client_id, "client_name": client_name, **direct_result["metrics"]})
            direct_pred_dfs.append(direct_result["pred_df"])

        if mode in {"indirect", "both"}:
            indirect_result = run_indirect_method(cfg, common_df, client_dir, prefix=client_name)
            indirect_summary_rows.append({"client_id": client_id, "client_name": client_name, **indirect_result["metrics"]})
            indirect_pred_dfs.append(indirect_result["pred_df"])

        save_method_compare(direct_result, indirect_result, client_dir, prefix=client_name, show_n=cfg.experiment.show_n)

    if direct_summary_rows:
        pd.DataFrame(direct_summary_rows).sort_values("client_id").to_csv(
            os.path.join(out_root, "per_client_direct_metrics_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if indirect_summary_rows:
        pd.DataFrame(indirect_summary_rows).sort_values("client_id").to_csv(
            os.path.join(out_root, "per_client_indirect_metrics_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if direct_summary_rows or indirect_summary_rows:
        compare_rows = []
        if direct_summary_rows:
            direct_df = pd.DataFrame(direct_summary_rows)
            compare_rows.append({
                "method": "direct_net_load",
                "MAE_mean": direct_df["MAE"].mean(),
                "RMSE_mean": direct_df["RMSE"].mean(),
                "MAPE_percent_mean": direct_df["MAPE_percent"].mean(),
                "R2_mean": direct_df["R2"].mean(),
            })
        if indirect_summary_rows:
            indirect_df = pd.DataFrame(indirect_summary_rows)
            compare_rows.append({
                "method": "indirect_gc_minus_gg",
                "MAE_mean": indirect_df["MAE"].mean(),
                "RMSE_mean": indirect_df["RMSE"].mean(),
                "MAPE_percent_mean": indirect_df["MAPE_percent"].mean(),
                "R2_mean": indirect_df["R2"].mean(),
            })
        pd.DataFrame(compare_rows).to_csv(
            os.path.join(out_root, "per_client_method_compare_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    return {
        "direct_summary_rows": direct_summary_rows,
        "indirect_summary_rows": indirect_summary_rows,
        "direct_pred_dfs": direct_pred_dfs,
        "indirect_pred_dfs": indirect_pred_dfs,
        "save_dir": out_root,
    }


def run_regional_sum_from_clients(cfg: Config, per_client_outputs: Dict[str, List[Dict]]):
    out_root = os.path.join(cfg.data.save_dir, "regional_sum_from_clients")
    ensure_dir(out_root)

    direct_result = None
    indirect_result = None

    if per_client_outputs["direct_pred_dfs"]:
        direct_df = merge_regional_predictions(per_client_outputs["direct_pred_dfs"])
        direct_metrics = calc_metrics(direct_df["y_true"].values, direct_df["y_pred"].values)
        direct_df.to_csv(os.path.join(out_root, "regional_sum_direct_test_predictions.csv"), index=False, encoding="utf-8-sig")
        save_metrics_csv(direct_metrics, os.path.join(out_root, "regional_sum_direct_test_metrics.csv"))
        plot_prediction(
            direct_df["y_true"].values,
            direct_df["y_pred"].values,
            save_path=os.path.join(out_root, "regional_sum_direct_test_prediction.png"),
            title="按客户端分别训练后再聚合的直接法总净负荷预测",
            ylabel="net_load",
            show_n=cfg.experiment.show_n,
        )
        direct_result = {"pred_df": direct_df, "metrics": direct_metrics}

    if per_client_outputs["indirect_pred_dfs"]:
        indirect_df = merge_regional_predictions(per_client_outputs["indirect_pred_dfs"])
        indirect_metrics = calc_metrics(indirect_df["y_true"].values, indirect_df["y_pred"].values)
        indirect_df.to_csv(os.path.join(out_root, "regional_sum_indirect_test_predictions.csv"), index=False, encoding="utf-8-sig")
        save_metrics_csv(indirect_metrics, os.path.join(out_root, "regional_sum_indirect_test_metrics.csv"))
        plot_prediction(
            indirect_df["y_true"].values,
            indirect_df["y_pred"].values,
            save_path=os.path.join(out_root, "regional_sum_indirect_test_prediction.png"),
            title="按客户端分别训练后再聚合的间接法总净负荷预测",
            ylabel="net_load",
            show_n=cfg.experiment.show_n,
        )
        indirect_result = {"pred_df": indirect_df, "metrics": indirect_metrics}

    save_method_compare(direct_result, indirect_result, out_root, prefix="regional_sum", show_n=cfg.experiment.show_n)
    return {"direct": direct_result, "indirect": indirect_result, "save_dir": out_root}


def run_aggregated_dataset_experiments(cfg: Config):
    mode = normalize_method_mode(cfg.experiment.method_mode)
    out_root = os.path.join(cfg.data.save_dir, "aggregated_dataset_results")
    ensure_dir(out_root)

    client_paths = [path for _, path in get_selected_client_paths(cfg)]
    common_df = build_aggregated_common_df(client_paths, cfg)

    agg_df_to_save = common_df.copy()
    agg_df_to_save.to_csv(os.path.join(out_root, "aggregated_dataset.csv"), index=False, encoding="utf-8-sig")

    direct_result = None
    indirect_result = None
    if mode in {"direct", "both"}:
        direct_result = run_direct_method(cfg, common_df, out_root, prefix="aggregated_dataset")
    if mode in {"indirect", "both"}:
        indirect_result = run_indirect_method(cfg, common_df, out_root, prefix="aggregated_dataset")

    save_method_compare(direct_result, indirect_result, out_root, prefix="aggregated_dataset", show_n=cfg.experiment.show_n)
    return {"direct": direct_result, "indirect": indirect_result, "save_dir": out_root}


def build_all_results_summary(cfg: Config, per_client_outputs: Optional[Dict], regional_outputs: Optional[Dict], aggregate_outputs: Optional[Dict]):
    rows = []

    if per_client_outputs is not None:
        for row in per_client_outputs.get("direct_summary_rows", []):
            rows.append({"scope": "per_client", "method": "direct_net_load", **row})
        for row in per_client_outputs.get("indirect_summary_rows", []):
            rows.append({"scope": "per_client", "method": "indirect_gc_minus_gg", **row})

    if regional_outputs is not None:
        if regional_outputs.get("direct") is not None:
            rows.append({
                "scope": "regional_sum_from_clients",
                "method": "direct_net_load",
                "client_id": "all",
                "client_name": "regional_sum",
                **regional_outputs["direct"]["metrics"],
            })
        if regional_outputs.get("indirect") is not None:
            rows.append({
                "scope": "regional_sum_from_clients",
                "method": "indirect_gc_minus_gg",
                "client_id": "all",
                "client_name": "regional_sum",
                **regional_outputs["indirect"]["metrics"],
            })

    if aggregate_outputs is not None:
        if aggregate_outputs.get("direct") is not None:
            rows.append({
                "scope": "aggregated_dataset",
                "method": "direct_net_load",
                "client_id": "all",
                "client_name": "aggregated_dataset",
                **aggregate_outputs["direct"]["metrics"],
            })
        if aggregate_outputs.get("indirect") is not None:
            rows.append({
                "scope": "aggregated_dataset",
                "method": "indirect_gc_minus_gg",
                "client_id": "all",
                "client_name": "aggregated_dataset",
                **aggregate_outputs["indirect"]["metrics"],
            })

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(cfg.data.save_dir, "all_results_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def main():
    cfg = deepcopy(CFG)
    set_seed(cfg.train.random_seed)
    ensure_dir(cfg.data.save_dir)
    save_config(cfg, cfg.data.save_dir)

    print("=" * 100)
    print("CNN-LSTM 净负荷多场景实验开始")
    print(f"device = {cfg.train.device}")
    print(f"method_mode = {cfg.experiment.method_mode}")
    print(f"selected_client_ids = {cfg.experiment.selected_client_ids}")
    print(f"run_per_client = {cfg.experiment.run_per_client}")
    print(f"run_regional_sum_from_clients = {cfg.experiment.run_regional_sum_from_clients}")
    print(f"run_aggregated_dataset = {cfg.experiment.run_aggregated_dataset}")
    print("=" * 100)

    per_client_outputs = None
    regional_outputs = None
    aggregate_outputs = None

    if cfg.experiment.run_per_client:
        per_client_outputs = run_per_client_experiments(cfg)

    if cfg.experiment.run_regional_sum_from_clients:
        if per_client_outputs is None:
            raise ValueError("run_regional_sum_from_clients=True 时，必须先运行 per_client。")
        regional_outputs = run_regional_sum_from_clients(cfg, per_client_outputs)

    if cfg.experiment.run_aggregated_dataset:
        aggregate_outputs = run_aggregated_dataset_experiments(cfg)

    build_all_results_summary(cfg, per_client_outputs, regional_outputs, aggregate_outputs)

    print("\n" + "=" * 100)
    print("全部实验完成")
    print(f"总输出目录: {cfg.data.save_dir}")
    if per_client_outputs is not None:
        print(f"分客户端结果目录: {per_client_outputs['save_dir']}")
    if regional_outputs is not None:
        print(f"按客户端预测后再聚合目录: {regional_outputs['save_dir']}")
    if aggregate_outputs is not None:
        print(f"九客户端聚合数据训练目录: {aggregate_outputs['save_dir']}")
    print(f"总汇总表: {os.path.join(cfg.data.save_dir, 'all_results_summary.csv')}")


if __name__ == "__main__":
    main()
