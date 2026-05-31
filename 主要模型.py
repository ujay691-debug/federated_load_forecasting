
import os
import math
import random
import joblib
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

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
# 以后你主要只改这里
# ============================================================
@dataclass
class DataConfig:
    data_path: str = r"per_client_merged/client_2_load_weather_30min.csv"
    datetime_col: str = "timestamp"
    target_col: str = "gc"

    seq_len: int = 48
    horizon: int = 1

    train_ratio: float = 0.8
    val_ratio: float = 0.1

    dropna: bool = True
    sort_by_time: bool = True

    freq_minutes: str = "auto"   # "auto" / 60 / 30 等
    save_dir: str = "runs/cnn_lstm_configurable"

    # 可选时间过滤
    use_time_range: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class FeatureConfig:
    # 直接从原始数据中取的历史特征
    raw_feature_cols: List[str] = field(default_factory=lambda: ["gc"])

    # 派生特征开关
    use_slot_sin_cos: bool =True
    use_weekday_sin_cos: bool = True
    use_month_sin_cos: bool = True
    use_is_weekend: bool = True
    use_is_holiday: bool = False

    # 气象特征
    use_temp_c: bool = True
    temp_source_mode: str = "auto"   # "auto" / "c" / "k"
    temp_c_col: str = "temp2m_c"
    temp_k_col: str = "temp2m_k"

    use_rh: bool = False
    rh_col: str = "rh2m_pct"

    use_wind: bool = True
    wind_col: str = "wind10m_ms"

    use_ghi: bool = False
    ghi_col: str = "ghi_wm2"

    use_apparent_temp: bool = False

    # 哪些列不要缩放
    no_scale_cols: List[str] = field(default_factory=lambda: ["is_weekend", "is_holiday"])


@dataclass
class ModelConfig:
    use_attention: bool = True

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
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    random_seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True

    loss_name: str = "mse"    # "mse" / "mae"
    optimizer_name: str = "adam"

    scaler_x: str = "minmax"  # "minmax" / "standard" / "none"
    scaler_y: str = "minmax"  # "minmax" / "standard" / "none"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


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


def infer_freq_minutes(dt_series: pd.Series) -> int:
    dt_sorted = dt_series.sort_values().drop_duplicates()
    diffs = dt_sorted.diff().dropna()
    if len(diffs) == 0:
        raise ValueError("无法从 datetime 推断分辨率，时间列样本不足。")
    minutes = int(diffs.mode().iloc[0].total_seconds() // 60)
    return minutes


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
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "R2": r2}


def inverse_transform_array(scaler, arr: np.ndarray) -> np.ndarray:
    if scaler is None:
        return arr
    original_shape = arr.shape
    arr_2d = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    restored = scaler.inverse_transform(arr_2d)
    return restored.reshape(original_shape)


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
        # x: [B, T, H]
        score_first_part = self.score_vec(x)                               # [B, T, H]
        h_t = x[:, -1, :]                                                  # [B, H]
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)   # [B, T]
        attn_weights = torch.softmax(score, dim=1)                         # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)       # [B, H]
        pre_activation = torch.cat([context, h_t], dim=1)                  # [B, 2H]
        attn_vector = torch.tanh(self.attn_out(pre_activation))            # [B, A]
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
        # x: [B, T, F]
        x = x.permute(0, 2, 1)          # [B, F, T]
        x = F.relu(self.conv1(x))       # [B, C1, T]
        x = self.pool1(x)               # [B, C1, T]
        x = self.dropout(x)

        x = F.relu(self.conv2(x))       # [B, C2, T]
        x = self.pool2(x)               # [B, C2, T]
        x = self.dropout(x)

        x = x.permute(0, 2, 1)          # [B, T, C2]
        x, _ = self.lstm1(x)            # [B, T, H1]
        x, _ = self.lstm2(x)            # [B, T, H2]

        if self.use_attention:
            x = self.attention(x)       # [B, A]
        else:
            x = x[:, -1, :]             # [B, H2]

        x = F.relu(self.fc1(x))         # [B, FC]
        x = self.fc2(x)                 # [B, output_dim]
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


def build_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    dc = cfg.data
    fc = cfg.feature

    if dc.datetime_col not in df.columns:
        raise ValueError(f"数据中缺少时间列: {dc.datetime_col}")

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
                raise ValueError(
                    f"已启用 temp_c，但数据中既没有 {fc.temp_c_col}，也没有 {fc.temp_k_col}"
                )
        elif temp_mode == "c":
            if fc.temp_c_col not in df.columns:
                raise ValueError(f"已启用 temp_c，且设为摄氏度模式，但数据中缺少列 {fc.temp_c_col}")
            df["temp_c"] = df[fc.temp_c_col].astype(float)
        elif temp_mode == "k":
            if fc.temp_k_col not in df.columns:
                raise ValueError(f"已启用 temp_c，且设为开尔文模式，但数据中缺少列 {fc.temp_k_col}")
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

    missing_required = [c for c in [dc.target_col] + feature_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"数据中缺少这些必要列: {missing_required}")

    if dc.dropna:
        df = df.dropna(subset=feature_cols + [dc.target_col]).reset_index(drop=True)

    return df, feature_cols


def split_df_by_time(df: pd.DataFrame, cfg: Config):
    n = len(df)
    train_end = int(n * cfg.data.train_ratio)
    val_end = int(n * (cfg.data.train_ratio + cfg.data.val_ratio))

    if train_end <= cfg.data.seq_len or val_end <= train_end:
        raise ValueError("训练/验证划分太小，无法形成有效序列。")

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


def create_sequences(feature_array: np.ndarray, target_array: np.ndarray, seq_len: int, horizon: int):
    xs, ys = [], []
    total_len = len(feature_array)
    for end_idx in range(seq_len, total_len - horizon + 1):
        start_idx = end_idx - seq_len
        x = feature_array[start_idx:end_idx, :]
        y = target_array[end_idx:end_idx + horizon, 0]
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    if horizon == 1:
        ys = ys.reshape(-1, 1)
    return xs, ys


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


def save_config(cfg: Config, save_dir: str):
    ensure_dir(save_dir)
    cfg_dict = asdict(cfg)
    import json
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)


def plot_curves(train_losses, val_losses, val_r2_list, save_dir: str):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="训练集Loss")
    plt.plot(val_losses, label="验证集Loss")
    plt.title("训练/验证 Loss 曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_r2_list, label="验证集R2")
    plt.title("验证集 R2 曲线")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_r2_curve.png"), dpi=200)
    plt.close()


def plot_prediction(true_real, pred_real, save_dir: str, target_col: str, show_n: int = 200):
    show_n = min(show_n, len(true_real))
    plt.figure(figsize=(12, 5))
    if true_real.ndim == 2 and true_real.shape[1] > 1:
        plt.plot(true_real[:show_n, 0], label=f"真实{target_col}(第1步)")
        plt.plot(pred_real[:show_n, 0], label=f"预测{target_col}(第1步)")
    else:
        plt.plot(np.squeeze(true_real[:show_n]), label=f"真实{target_col}")
        plt.plot(np.squeeze(pred_real[:show_n]), label=f"预测{target_col}")
    plt.title(f"测试集 {target_col} 预测效果")
    plt.xlabel("样本点")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_prediction.png"), dpi=200)
    plt.close()


def main(cfg: Config):
    set_seed(cfg.train.random_seed)
    ensure_dir(cfg.data.save_dir)
    save_config(cfg, cfg.data.save_dir)

    print("=" * 80)
    print("当前配置")
    print(f"数据路径: {cfg.data.data_path}")
    print(f"目标列: {cfg.data.target_col}")
    print(f"输入窗口长度: {cfg.data.seq_len}")
    print(f"预测步长 horizon: {cfg.data.horizon}")
    print(f"是否使用Attention: {cfg.model.use_attention}")
    print("=" * 80)

    df = pd.read_csv(cfg.data.data_path)
    df, feature_cols = build_features(df, cfg)
    train_df, val_df, test_df = split_df_by_time(df, cfg)

    train_scaled_df, val_scaled_df, test_scaled_df, x_scaler, scaled_cols, kept_cols = fit_and_transform_x(
        train_df, val_df, test_df, feature_cols, cfg
    )
    y_train_scaled, y_val_scaled, y_test_scaled, y_scaler = fit_and_transform_y(train_df, val_df, test_df, cfg)

    x_train = train_scaled_df[feature_cols].values
    x_val = val_scaled_df[feature_cols].values
    x_test = test_scaled_df[feature_cols].values

    x_train_seq, y_train_seq = create_sequences(x_train, y_train_scaled, cfg.data.seq_len, cfg.data.horizon)
    x_val_seq, y_val_seq = create_sequences(x_val, y_val_scaled, cfg.data.seq_len, cfg.data.horizon)
    x_test_seq, y_test_seq = create_sequences(x_test, y_test_scaled, cfg.data.seq_len, cfg.data.horizon)

    print(f"数据行数: {len(df)}")
    print(f"时间范围: {df[cfg.data.datetime_col].min()} -> {df[cfg.data.datetime_col].max()}")
    print(f"输入特征列: {feature_cols}")
    print(f"缩放列: {scaled_cols}")
    print(f"不缩放列: {kept_cols}")
    print(f"x_train shape: {x_train_seq.shape}, y_train shape: {y_train_seq.shape}")
    print(f"x_val   shape: {x_val_seq.shape}, y_val   shape: {y_val_seq.shape}")
    print(f"x_test  shape: {x_test_seq.shape}, y_test  shape: {y_test_seq.shape}")

    train_loader = DataLoader(
        SeqDataset(x_train_seq, y_train_seq),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and cfg.train.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        SeqDataset(x_val_seq, y_val_seq),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and cfg.train.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        SeqDataset(x_test_seq, y_test_seq),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and cfg.train.device.startswith("cuda"),
    )

    device = torch.device(cfg.train.device)
    model = CNNLSTMModel(
        input_dim=len(feature_cols),
        output_dim=cfg.data.horizon,
        cfg=cfg.model,
    ).to(device)

    criterion = get_loss_fn(cfg.train.loss_name)
    optimizer = get_optimizer(cfg.train.optimizer_name, model, cfg.train.lr)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_r2_list = []

    model_path = os.path.join(cfg.data.save_dir, "best_model.pth")
    x_scaler_path = os.path.join(cfg.data.save_dir, "x_scaler.save")
    y_scaler_path = os.path.join(cfg.data.save_dir, "y_scaler.save")

    if x_scaler is not None:
        joblib.dump(x_scaler, x_scaler_path)
    if y_scaler is not None:
        joblib.dump(y_scaler, y_scaler_path)

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, _, _ = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_pred_scaled, val_true_scaled = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        val_pred_real = inverse_transform_array(y_scaler, val_pred_scaled)
        val_true_real = inverse_transform_array(y_scaler, val_true_scaled)

        if cfg.data.horizon == 1:
            val_r2 = r2_score(val_true_real.reshape(-1), val_pred_real.reshape(-1))
        else:
            val_r2 = r2_score(val_true_real.reshape(-1), val_pred_real.reshape(-1))
        val_r2_list.append(val_r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        print(
            f"Epoch [{epoch:03d}/{cfg.train.epochs}] | "
            f"TrainLoss: {train_loss:.6f} | "
            f"ValLoss: {val_loss:.6f} | "
            f"ValR2: {val_r2:.6f}"
        )

    print(f"\n最优模型已保存到: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_pred_scaled, test_true_scaled = run_one_epoch(model, test_loader, criterion, optimizer, device, train=False)

    pred_real = inverse_transform_array(y_scaler, test_pred_scaled)
    true_real = inverse_transform_array(y_scaler, test_true_scaled)

    metrics = calc_metrics(true_real.reshape(-1), pred_real.reshape(-1))

    print("\n[Test Metrics]")
    for k, v in metrics.items():
        if "percent" in k.lower():
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.6f}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(cfg.data.save_dir, "test_metrics.csv"), index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "y_true": true_real.reshape(-1),
        "y_pred": pred_real.reshape(-1),
    })
    pred_df.to_csv(os.path.join(cfg.data.save_dir, "test_predictions.csv"), index=False, encoding="utf-8-sig")

    plot_curves(train_losses, val_losses, val_r2_list, cfg.data.save_dir)
    plot_prediction(true_real, pred_real, cfg.data.save_dir, cfg.data.target_col, show_n=200)


if __name__ == "__main__":
    main(CFG)
