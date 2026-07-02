import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.cnn_lstm import CNNLSTMModel

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    MinMaxScaler = None

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None


class SimpleMinMaxScaler:
    """Small fallback for environments without sklearn."""

    def fit(self, x):
        arr = np.asarray(x, dtype=np.float64)
        self.data_min_ = np.min(arr, axis=0)
        self.data_max_ = np.max(arr, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.safe_range_ = np.where(self.data_range_ == 0.0, 1.0, self.data_range_)
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
    """Small fallback for expert inputs when sklearn is unavailable."""

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


@dataclass
class ModelConfig:
    use_attention: bool = True
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


class NetLoadNextStepDataset(Dataset):
    """x uses previous seq_len feature rows; y is future net-load."""

    def __init__(self, scaled_features, scaled_target, timestamps, seq_len, horizon):
        self.scaled_features = np.asarray(scaled_features, dtype=np.float32)
        self.scaled_target = np.asarray(scaled_target, dtype=np.float32).reshape(-1)
        self.timestamps = np.asarray(timestamps)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        if self.scaled_features.ndim != 2:
            raise ValueError("scaled_features must be a 2D array: [rows, features].")
        if len(self.scaled_features) != len(self.scaled_target) or len(self.scaled_target) != len(self.timestamps):
            raise ValueError("scaled_features, scaled_target, and timestamps must have the same length.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")

    def __len__(self):
        return max(0, len(self.scaled_target) - self.seq_len - self.horizon + 1)

    def __getitem__(self, idx):
        x = self.scaled_features[idx : idx + self.seq_len]
        y = self.scaled_target[idx + self.seq_len : idx + self.seq_len + self.horizon]
        current_idx = idx + self.seq_len - 1
        target_idx = idx + self.seq_len
        timestamp_target = "|".join(
            str(ts) for ts in self.timestamps[idx + self.seq_len : idx + self.seq_len + self.horizon]
        )
        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy()),
            timestamp_target,
            torch.tensor(current_idx, dtype=torch.long),
            torch.tensor(target_idx, dtype=torch.long),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local CNN-LSTM-Attention net-load forecasting with optional hard-gated expert correction."
    )
    parser.add_argument(
        "--data-path",
        default="per_client_merged/client_2_load_weather_30min.csv",
        help="CSV path containing timestamp, gc, gg columns.",
    )
    parser.add_argument(
        "--save-dir",
        default="runs/cnn_lstm_attention_netload_pv_load_experts",
        help="Output directory.",
    )
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--feature-mode",
        default="net_load_weather",
        choices=["net_load_only", "net_load_weather"],
        help="Input feature set. net_load_weather adds historical GHI, temperature, and wind.",
    )
    parser.add_argument("--ghi-col", default="ghi_wm2", help="Historical GHI column used as an input feature.")
    parser.add_argument("--temp-col", default="temp_c", help="Historical temperature column used as an input feature.")
    parser.add_argument("--wind-col", default="wind10m_ms", help="Historical wind column used as an input feature.")
    parser.add_argument("--conv1-channels", type=int, default=32)
    parser.add_argument("--conv2-channels", type=int, default=64)
    parser.add_argument("--lstm-hidden1", type=int, default=48)
    parser.add_argument("--lstm-hidden2", type=int, default=24)
    parser.add_argument("--attn-units", type=int, default=24)
    parser.add_argument("--fc-hidden", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--enable-expert-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the second-stage hard-gated expert correction model.",
    )
    parser.add_argument("--expert-epochs", type=int, default=80)
    parser.add_argument("--expert-lr", type=float, default=1e-3)
    parser.add_argument("--expert-patience", type=int, default=10)
    parser.add_argument("--expert-batch-size", type=int, default=512)
    parser.add_argument("--expert-lambda", type=float, default=0.2)
    parser.add_argument("--sign-lambda", type=float, default=0.05)
    parser.add_argument("--corr-lambda", type=float, default=0.02)
    parser.add_argument("--expert-dropout", type=float, default=0.05)
    parser.add_argument("--expert-hidden1", type=int, default=32)
    parser.add_argument("--expert-hidden2", type=int, default=16)
    parser.add_argument("--expert-amp1", type=float, default=2.0)
    parser.add_argument("--expert-amp2", type=float, default=2.5)
    parser.add_argument("--expert-amp3", type=float, default=2.0)
    parser.add_argument("--expert-amp4", type=float, default=2.0)
    parser.add_argument("--expert-amp5", type=float, default=1.5)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_scaler():
    if MinMaxScaler is not None:
        return MinMaxScaler()
    return SimpleMinMaxScaler()


def make_standard_scaler():
    if StandardScaler is not None:
        return StandardScaler()
    return SimpleStandardScaler()


def scaler_to_dict(scaler):
    result = {"type": "sklearn.MinMaxScaler" if MinMaxScaler is not None else "SimpleMinMaxScaler"}
    for name in ["data_min_", "data_max_", "data_range_", "scale_", "min_"]:
        if hasattr(scaler, name):
            result[name.rstrip("_")] = np.asarray(getattr(scaler, name), dtype=float).reshape(-1).tolist()
    if hasattr(scaler, "safe_range_"):
        result["safe_range"] = np.asarray(scaler.safe_range_, dtype=float).reshape(-1).tolist()
    return result


def standard_scaler_to_dict(scaler):
    result = {"type": "sklearn.StandardScaler" if StandardScaler is not None else "SimpleStandardScaler"}
    for name in ["mean_", "scale_", "var_"]:
        if hasattr(scaler, name):
            result[name.rstrip("_")] = np.asarray(getattr(scaler, name), dtype=float).reshape(-1).tolist()
    return result


def normalize_weather_columns(df):
    df = df.copy()
    if "temp_c" not in df.columns:
        if "temp2m_c" in df.columns:
            df["temp_c"] = df["temp2m_c"]
        elif "temp2m_k" in df.columns:
            df["temp_c"] = pd.to_numeric(df["temp2m_k"], errors="coerce") - 273.15
    return df


def detect_future_weather_mode(df):
    forecast_cols = {
        "GHI_f": "ghi_forecast_wm2",
        "temp_f": "temp_forecast_c",
        "wind_f": "wind_forecast_ms",
    }
    if all(col in df.columns for col in forecast_cols.values()):
        return "explicit_forecast_columns", forecast_cols
    return (
        "target_time_observed_as_forecast_proxy",
        {"GHI_f": "ghi_wm2", "temp_f": "temp_c", "wind_f": "wind10m_ms"},
    )


def load_and_prepare_data(data_path, seq_len, horizon, train_ratio, val_ratio, feature_mode, ghi_col, temp_col, wind_col):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Please confirm --data-path."
        )

    df = normalize_weather_columns(pd.read_csv(data_path))
    raw_rows = len(df)
    feature_cols = ["net_load"]
    if feature_mode == "net_load_weather":
        feature_cols.extend([ghi_col, temp_col, wind_col])

    required = {"timestamp", "gc", "gg", *[col for col in feature_cols if col != "net_load"]}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}.")
    if feature_mode == "net_load_weather" and wind_col not in df.columns:
        raise ValueError(f"CSV is missing selected wind column '{wind_col}'.")
    expert_required = {"timestamp", "ghi_wm2", "temp_c", "wind10m_ms"}
    expert_missing = sorted(expert_required.difference(df.columns))
    if expert_missing:
        raise ValueError(f"CSV is missing required expert weather columns: {', '.join(expert_missing)}.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["gc"] = pd.to_numeric(df["gc"], errors="coerce")
    df["gg"] = pd.to_numeric(df["gg"], errors="coerce")
    for col in feature_cols:
        if col != "net_load":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "gc", "gg", *[col for col in feature_cols if col != "net_load"]])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["net_load"] = df["gc"] - df["gg"]
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    future_weather_mode, future_weather_cols = detect_future_weather_mode(df)
    expert_numeric_cols = {"ghi_wm2", "temp_c", "wind10m_ms", *future_weather_cols.values()}
    for col in expert_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=sorted(expert_numeric_cols)).reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=np.int64)

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    if len(df) <= seq_len + horizon:
        raise ValueError(f"Not enough rows ({len(df)}) for seq_len={seq_len}, horizon={horizon}.")

    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) <= seq_len + horizon - 1:
            raise ValueError(
                f"{split_name} split has only {len(split_df)} rows; seq_len={seq_len}, horizon={horizon}."
            )

    feature_scaler = make_scaler()
    target_scaler = make_scaler()
    train_features = feature_scaler.fit_transform(train_df[feature_cols].values).astype(np.float32)
    val_features = feature_scaler.transform(val_df[feature_cols].values).astype(np.float32)
    test_features = feature_scaler.transform(test_df[feature_cols].values).astype(np.float32)

    train_target = target_scaler.fit_transform(train_df[["net_load"]].values).astype(np.float32).reshape(-1)
    val_target = target_scaler.transform(val_df[["net_load"]].values).astype(np.float32).reshape(-1)
    test_target = target_scaler.transform(test_df[["net_load"]].values).astype(np.float32).reshape(-1)
    all_target = np.concatenate([train_target, val_target, test_target])

    datasets = {
        "train": NetLoadNextStepDataset(train_features, train_target, train_df["timestamp"].values, seq_len, horizon),
        "val": NetLoadNextStepDataset(val_features, val_target, val_df["timestamp"].values, seq_len, horizon),
        "test": NetLoadNextStepDataset(test_features, test_target, test_df["timestamp"].values, seq_len, horizon),
    }

    split_info = {
        "raw_rows": int(raw_rows),
        "clean_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_samples": int(len(datasets["train"])),
        "val_samples": int(len(datasets["val"])),
        "test_samples": int(len(datasets["test"])),
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "input_dim": int(len(feature_cols)),
        "future_weather_mode": future_weather_mode,
        "future_weather_cols": future_weather_cols,
        "train_scaled_min": float(np.min(train_target)),
        "train_scaled_max": float(np.max(train_target)),
        "all_scaled_min": float(np.min(all_target)),
        "all_scaled_max": float(np.max(all_target)),
    }
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    return df, datasets, feature_scaler, target_scaler, split_info, split_dfs


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    sample_count = 0

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = criterion(pred, y)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    if sample_count <= 0:
        raise ValueError("DataLoader has no samples.")
    return total_loss / sample_count


def inverse_transform_1d(scaler, values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)


def compute_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    valid = np.abs(y_true) > eps
    mape = np.mean(np.abs(error[valid] / y_true[valid])) * 100.0 if np.any(valid) else np.nan
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if ss_tot <= eps else 1.0 - ss_res / ss_tot
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
        "R2": float(r2),
    }


def collect_predictions(model, loader, device):
    model.eval()
    preds = []
    trues = []
    timestamps = []

    with torch.no_grad():
        for batch in loader:
            x, y, ts = batch[0], batch[1], batch[2]
            x = x.to(device=device, dtype=torch.float32)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
            timestamps.extend(list(ts))

    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0), np.asarray(timestamps)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, model_cfg, config, epoch, val_loss):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_cfg),
            "config": config,
            "epoch": int(epoch),
            "val_mse_loss": float(val_loss),
        },
        path,
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_loss_curve(logs, path):
    log_df = pd.DataFrame(logs)
    plt.figure(figsize=(9, 5))
    plt.plot(log_df["epoch"], log_df["train_mse_loss"], label="Train MSE loss")
    plt.plot(log_df["epoch"], log_df["val_mse_loss"], label="Validation MSE loss")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled-space MSE")
    plt.title("CNN-LSTM-Attention Train/Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_test_plot(path, y_true, y_pred, max_points=300):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = min(len(y_true), int(max_points))
    x_axis = np.arange(n)
    plt.figure(figsize=(11, 5))
    plt.plot(x_axis, y_true[:n], label="True net load")
    plt.plot(x_axis, y_pred[:n], label="Predicted net load")
    plt.xlabel("Test sample")
    plt.ylabel("Net load")
    plt.title("CNN-LSTM-Attention Test Multi-Step Prediction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


PV_EXPERT_INPUT_COLS = [
    "GHI_t",
    "temp_t",
    "wind_t",
    "GHI_f",
    "temp_f",
    "wind_f",
    "y_base",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

LOAD_EXPERT_INPUT_COLS = [
    "N_t",
    "N_recent_mean",
    "N_recent_std",
    "y_base",
    "temp_t",
    "temp_f",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]

EXPERT_NAMES = [
    "pv_start_expert",
    "pv_day_expert",
    "pv_exit_expert",
    "load_ramp_expert",
    "load_fall_expert",
]

SIGN_PRIORS = [-1.0, -1.0, 1.0, 1.0, -1.0]


def build_expert_frame_for_split(split_df, base_pred_real, true_real, seq_len):
    base = np.asarray(base_pred_real, dtype=np.float64).reshape(-1)
    true = np.asarray(true_real, dtype=np.float64).reshape(-1)
    expected_samples = len(split_df) - int(seq_len)
    if len(base) != expected_samples or len(true) != expected_samples:
        raise ValueError(
            "Expert frame length mismatch: "
            f"expected {expected_samples}, got base={len(base)}, true={len(true)}."
        )

    future_weather_mode, future_cols = detect_future_weather_mode(split_df)
    needed = ["timestamp", "net_load", "ghi_wm2", "temp_c", "wind10m_ms", *future_cols.values()]
    missing = sorted(set(needed).difference(split_df.columns))
    if missing:
        raise ValueError(f"Split is missing columns needed by expert frame: {', '.join(missing)}.")

    timestamps = pd.to_datetime(split_df["timestamp"]).reset_index(drop=True)
    numeric = {}
    for col in sorted(set(needed).difference({"timestamp"})):
        numeric[col] = pd.to_numeric(split_df[col], errors="coerce").reset_index(drop=True).to_numpy(dtype=np.float64)

    rows = []
    for sample_idx in range(expected_samples):
        current_idx = sample_idx + int(seq_len) - 1
        target_idx = sample_idx + int(seq_len)
        prev_idx = max(0, current_idx - 1)
        target_ts = pd.Timestamp(timestamps.iloc[target_idx])
        current_ts = pd.Timestamp(timestamps.iloc[current_idx])

        n_t = numeric["net_load"][current_idx]
        n_t_minus_1 = numeric["net_load"][prev_idx]
        recent_mean_start = max(0, current_idx - 3)
        recent_std_start = max(0, current_idx - 5)
        n_recent_mean = np.mean(
            numeric["net_load"][recent_mean_start : current_idx + 1]
        )
        n_recent_std = np.std(
            numeric["net_load"][recent_std_start : current_idx + 1], ddof=0
        )
        ghi_t = numeric["ghi_wm2"][current_idx]
        temp_t = numeric["temp_c"][current_idx]
        wind_t = numeric["wind10m_ms"][current_idx]
        ghi_f = numeric[future_cols["GHI_f"]][target_idx]
        temp_f = numeric[future_cols["temp_f"]][target_idx]
        wind_f = numeric[future_cols["wind_f"]][target_idx]
        y_base = base[sample_idx]
        y_true = true[sample_idx]
        target_hour = int(target_ts.hour)
        target_weekday = int(target_ts.weekday())
        target_month = int(target_ts.month)

        rows.append(
            {
                "sample_index": int(sample_idx),
                "current_idx": int(current_idx),
                "target_idx": int(target_idx),
                "current_timestamp": current_ts,
                "target_timestamp": target_ts,
                "N_t": float(n_t),
                "N_t_minus_1": float(n_t_minus_1),
                "N_recent_mean": float(n_recent_mean),
                "N_recent_std": float(n_recent_std),
                "GHI_t": float(ghi_t),
                "temp_t": float(temp_t),
                "wind_t": float(wind_t),
                "GHI_f": float(ghi_f),
                "temp_f": float(temp_f),
                "wind_f": float(wind_f),
                "y_base": float(y_base),
                "y_true": float(y_true),
                "residual": float(y_true - y_base),
                "dN_hist": float(n_t - n_t_minus_1),
                "dGHI_f": float(ghi_f - ghi_t),
                "dBase": float(y_base - n_t),
                "target_hour": target_hour,
                "target_weekday": target_weekday,
                "target_month": target_month,
                "is_weekend": float(target_weekday >= 5),
                "hour_sin": float(np.sin(2.0 * np.pi * target_hour / 24.0)),
                "hour_cos": float(np.cos(2.0 * np.pi * target_hour / 24.0)),
                "weekday_sin": float(np.sin(2.0 * np.pi * target_weekday / 7.0)),
                "weekday_cos": float(np.cos(2.0 * np.pi * target_weekday / 7.0)),
                "month_sin": float(np.sin(2.0 * np.pi * (target_month - 1) / 12.0)),
                "month_cos": float(np.cos(2.0 * np.pi * (target_month - 1) / 12.0)),
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["future_weather_mode"] = future_weather_mode
    frame.attrs["future_weather_cols"] = future_cols
    return frame


def add_expert_gate_masks(expert_frame, thresholds):
    frame = expert_frame.copy()
    g_off = thresholds["G_off"]
    g_pv = thresholds["G_pv"]
    g_day = thresholds["G_day"]
    g_up = thresholds["G_up"]
    g_down = thresholds["G_down"]
    n_mid = thresholds["N_mid"]

    hour = frame["target_hour"]
    frame["m1"] = (
        (hour >= 7)
        & (hour < 9)
        & (frame["GHI_t"] <= g_pv)
        & (frame["GHI_f"] > g_pv)
        & (frame["dGHI_f"] > g_up)
    ).astype(np.float32)
    frame["m2"] = (
        (hour >= 9)
        & (hour < 15)
        & (frame["GHI_f"] > g_day)
    ).astype(np.float32)
    frame["m3"] = (
        (hour >= 15)
        & (hour < 20)
        & ((frame["GHI_f"] < g_pv) | (frame["dGHI_f"] < -g_down))
    ).astype(np.float32)
    frame["m4"] = (
        (((hour >= 5) & (hour < 8)) | ((hour >= 16) & (hour < 20)))
        & ((frame["dN_hist"] > 0.5) | (frame["dBase"] > 0.3))
    ).astype(np.float32)
    frame["m5"] = (
        (hour >= 20)
        & (hour < 23)
        & ((frame["dN_hist"] < -0.5) | (frame["dBase"] < 0.0))
        & ((frame["N_t"] > n_mid) | (frame["y_base"] > n_mid))
    ).astype(np.float32)
    return frame


class ExpertCorrectionModel(nn.Module):
    def __init__(
        self,
        pv_input_dim=len(PV_EXPERT_INPUT_COLS),
        load_input_dim=len(LOAD_EXPERT_INPUT_COLS),
        hidden1=32,
        hidden2=16,
        dropout=0.05,
        amps=None,
    ):
        super().__init__()
        amps = [2.0, 2.5, 2.0, 2.0, 1.5] if amps is None else amps

        def build_mlp(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.LayerNorm(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, 1),
            )

        self.pv_experts = nn.ModuleList(
            [build_mlp(pv_input_dim) for _ in range(3)]
        )
        self.load_experts = nn.ModuleList(
            [build_mlp(load_input_dim) for _ in range(2)]
        )
        for expert in [*self.pv_experts, *self.load_experts]:
            last_linear = expert[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
        self.register_buffer("amps", torch.tensor(amps, dtype=torch.float32))

    def forward(self, pv_x_scaled, load_x_scaled, masks, y_base_real):
        raw_outputs = [
            expert(pv_x_scaled).squeeze(1) for expert in self.pv_experts
        ]
        raw_outputs.extend(
            expert(load_x_scaled).squeeze(1) for expert in self.load_experts
        )
        raw_outputs = torch.stack(raw_outputs, dim=1)
        deltas = self.amps.unsqueeze(0) * torch.tanh(raw_outputs)
        corrections = masks * deltas
        y_final_real = y_base_real.reshape(-1) + corrections.sum(dim=1)
        return y_final_real, deltas, corrections


class ExpertDataset(Dataset):
    def __init__(
        self,
        pv_input_scaled,
        load_input_scaled,
        masks,
        y_base_real,
        y_true_real,
        residual_real,
        row_indices,
    ):
        self.pv_input_scaled = np.asarray(pv_input_scaled, dtype=np.float32)
        self.load_input_scaled = np.asarray(load_input_scaled, dtype=np.float32)
        self.masks = np.asarray(masks, dtype=np.float32)
        self.y_base_real = np.asarray(y_base_real, dtype=np.float32).reshape(-1)
        self.y_true_real = np.asarray(y_true_real, dtype=np.float32).reshape(-1)
        self.residual_real = np.asarray(residual_real, dtype=np.float32).reshape(-1)
        self.row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)

    def __len__(self):
        return len(self.y_true_real)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.pv_input_scaled[idx].copy()),
            torch.from_numpy(self.load_input_scaled[idx].copy()),
            torch.from_numpy(self.masks[idx].copy()),
            torch.tensor(self.y_base_real[idx], dtype=torch.float32),
            torch.tensor(self.y_true_real[idx], dtype=torch.float32),
            torch.tensor(self.residual_real[idx], dtype=torch.float32),
            torch.tensor(self.row_indices[idx], dtype=torch.long),
        )


def make_expert_dataset(frame, pv_input_scaled, load_input_scaled):
    masks = frame[[f"m{k}" for k in range(1, 6)]].to_numpy(dtype=np.float32)
    return ExpertDataset(
        pv_input_scaled=pv_input_scaled,
        load_input_scaled=load_input_scaled,
        masks=masks,
        y_base_real=frame["y_base"].to_numpy(dtype=np.float32),
        y_true_real=frame["y_true"].to_numpy(dtype=np.float32),
        residual_real=frame["residual"].to_numpy(dtype=np.float32),
        row_indices=frame.index.to_numpy(dtype=np.int64),
    )


def expert_loss_components(
    y_final,
    deltas,
    masks,
    y_true,
    residual,
    expert_lambda=0.2,
    sign_lambda=0.05,
    corr_lambda=0.02,
):
    final_loss = torch.mean((y_final - y_true.reshape(-1)) ** 2)
    expert_losses = []
    sign_losses = []
    for k in range(5):
        mask = masks[:, k]
        mask_sum = torch.sum(mask)
        if mask_sum.item() <= 0.0:
            expert_losses.append(deltas[:, k].sum() * 0.0)
            sign_losses.append(deltas[:, k].sum() * 0.0)
        else:
            expert_losses.append(
                torch.sum(mask * (deltas[:, k] - residual.reshape(-1)) ** 2)
                / (mask_sum + 1e-8)
            )
            if SIGN_PRIORS[k] > 0:
                wrong_direction = torch.relu(-deltas[:, k])
            else:
                wrong_direction = torch.relu(deltas[:, k])
            sign_losses.append(
                torch.sum(mask * wrong_direction ** 2) / (mask_sum + 1e-8)
            )
    expert_loss = torch.stack(expert_losses).mean()
    sign_loss = torch.stack(sign_losses).mean()
    correction_l1 = torch.mean(torch.sum(torch.abs(masks * deltas), dim=1))
    total_loss = (
        final_loss
        + float(expert_lambda) * expert_loss
        + float(sign_lambda) * sign_loss
        + float(corr_lambda) * correction_l1
    )
    return total_loss, final_loss, expert_loss, sign_loss, correction_l1


def run_expert_epoch(
    model,
    loader,
    device,
    optimizer=None,
    expert_lambda=0.2,
    sign_lambda=0.05,
    corr_lambda=0.02,
):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "final_loss": 0.0,
        "expert_loss": 0.0,
        "sign_loss": 0.0,
        "correction_l1": 0.0,
        "count": 0,
    }

    for pv_x, load_x, masks, y_base, y_true, residual, _ in loader:
        pv_x = pv_x.to(device=device, dtype=torch.float32)
        load_x = load_x.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=torch.float32)
        y_base = y_base.to(device=device, dtype=torch.float32)
        y_true = y_true.to(device=device, dtype=torch.float32)
        residual = residual.to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            y_final, deltas, _ = model(pv_x, load_x, masks, y_base)
            loss, final_loss, expert_loss, sign_loss, correction_l1 = (
                expert_loss_components(
                    y_final,
                    deltas,
                    masks,
                    y_true,
                    residual,
                    expert_lambda,
                    sign_lambda,
                    corr_lambda,
                )
            )
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(y_true.size(0))
        totals["loss"] += float(loss.item()) * batch_size
        totals["final_loss"] += float(final_loss.item()) * batch_size
        totals["expert_loss"] += float(expert_loss.item()) * batch_size
        totals["sign_loss"] += float(sign_loss.item()) * batch_size
        totals["correction_l1"] += float(correction_l1.item()) * batch_size
        totals["count"] += batch_size

    if totals["count"] <= 0:
        raise ValueError("Expert DataLoader has no samples.")
    return {
        key: totals[key] / totals["count"]
        for key in [
            "loss",
            "final_loss",
            "expert_loss",
            "sign_loss",
            "correction_l1",
        ]
    }


def predict_expert(
    model,
    loader,
    device,
    expert_lambda=0.2,
    sign_lambda=0.05,
    corr_lambda=0.02,
):
    model.eval()
    y_base_all, y_true_all, y_final_all = [], [], []
    masks_all, deltas_all, corrections_all, row_indices_all = [], [], [], []
    totals = {
        "loss": 0.0,
        "final_loss": 0.0,
        "expert_loss": 0.0,
        "sign_loss": 0.0,
        "correction_l1": 0.0,
        "count": 0,
    }

    with torch.no_grad():
        for pv_x, load_x, masks, y_base, y_true, residual, row_indices in loader:
            pv_x = pv_x.to(device=device, dtype=torch.float32)
            load_x = load_x.to(device=device, dtype=torch.float32)
            masks_dev = masks.to(device=device, dtype=torch.float32)
            y_base_dev = y_base.to(device=device, dtype=torch.float32)
            y_true_dev = y_true.to(device=device, dtype=torch.float32)
            residual_dev = residual.to(device=device, dtype=torch.float32)

            y_final, deltas, corrections = model(
                pv_x, load_x, masks_dev, y_base_dev
            )
            loss, final_loss, expert_loss, sign_loss, correction_l1 = (
                expert_loss_components(
                    y_final,
                    deltas,
                    masks_dev,
                    y_true_dev,
                    residual_dev,
                    expert_lambda,
                    sign_lambda,
                    corr_lambda,
                )
            )

            batch_size = int(y_true.size(0))
            totals["loss"] += float(loss.item()) * batch_size
            totals["final_loss"] += float(final_loss.item()) * batch_size
            totals["expert_loss"] += float(expert_loss.item()) * batch_size
            totals["sign_loss"] += float(sign_loss.item()) * batch_size
            totals["correction_l1"] += float(correction_l1.item()) * batch_size
            totals["count"] += batch_size

            y_base_all.append(y_base.numpy())
            y_true_all.append(y_true.numpy())
            y_final_all.append(y_final.cpu().numpy())
            masks_all.append(masks.numpy())
            deltas_all.append(deltas.cpu().numpy())
            corrections_all.append(corrections.cpu().numpy())
            row_indices_all.append(row_indices.numpy())

    result = {
        "y_base": np.concatenate(y_base_all),
        "y_true": np.concatenate(y_true_all),
        "y_final": np.concatenate(y_final_all),
        "masks": np.concatenate(masks_all, axis=0),
        "deltas": np.concatenate(deltas_all, axis=0),
        "corrections": np.concatenate(corrections_all, axis=0),
        "row_indices": np.concatenate(row_indices_all),
    }
    result.update(
        {
            key: totals[key] / totals["count"]
            for key in [
                "loss",
                "final_loss",
                "expert_loss",
                "sign_loss",
                "correction_l1",
            ]
        }
    )
    return result


def save_expert_prediction_plot(path, y_true, y_base, y_final, max_points=300):
    y_true = np.asarray(y_true).reshape(-1)
    y_base = np.asarray(y_base).reshape(-1)
    y_final = np.asarray(y_final).reshape(-1)
    n = min(len(y_true), int(max_points))
    x_axis = np.arange(n)
    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, y_true[:n], label="True net load")
    plt.plot(x_axis, y_base[:n], label="Base prediction")
    plt.plot(x_axis, y_final[:n], label="Expert corrected")
    plt.xlabel("Test sample")
    plt.ylabel("Net load")
    plt.title("Base vs Expert-Corrected Test Prediction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def gate_activation_summary(frames):
    rows = []
    for split_name, frame in frames.items():
        total = int(len(frame))
        for idx, name in enumerate(EXPERT_NAMES, start=1):
            active = int(frame[f"m{idx}"].sum())
            rows.append(
                {
                    "split": split_name,
                    "expert_name": name,
                    "active_count": active,
                    "total_count": total,
                    "active_ratio": float(active / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def metrics_rows_base_vs_expert(y_true, y_base, y_final):
    rows = []
    for model_name, pred in [("base", y_base), ("expert_corrected", y_final)]:
        row = {"model": model_name, "N": int(len(np.asarray(y_true).reshape(-1)))}
        row.update(compute_metrics(y_true, pred))
        rows.append(row)
    return rows


def save_expert_split_outputs(save_dir, split_name, frame, pred_dict):
    """Save expert-corrected predictions and metrics for one data split."""
    split_name = str(split_name).lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported expert output split: {split_name}")

    row_indices = np.asarray(pred_dict["row_indices"], dtype=np.int64).reshape(-1)
    aligned_frame = frame.loc[row_indices].reset_index(drop=True)
    sample_count = len(aligned_frame)

    y_true = np.asarray(pred_dict["y_true"], dtype=np.float64).reshape(-1)
    y_base = np.asarray(pred_dict["y_base"], dtype=np.float64).reshape(-1)
    y_final = np.asarray(pred_dict["y_final"], dtype=np.float64).reshape(-1)
    masks = np.asarray(pred_dict["masks"], dtype=np.float64)
    deltas = np.asarray(pred_dict["deltas"], dtype=np.float64)
    corrections = np.asarray(pred_dict["corrections"], dtype=np.float64)

    expected_shapes = {
        "y_true": (sample_count,),
        "y_base": (sample_count,),
        "y_final": (sample_count,),
        "masks": (sample_count, 5),
        "deltas": (sample_count, 5),
        "corrections": (sample_count, 5),
    }
    actual_arrays = {
        "y_true": y_true,
        "y_base": y_base,
        "y_final": y_final,
        "masks": masks,
        "deltas": deltas,
        "corrections": corrections,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_arrays[name].shape != expected_shape:
            raise ValueError(
                f"{split_name} expert output {name} has shape "
                f"{actual_arrays[name].shape}; expected {expected_shape}."
            )

    output = pd.DataFrame(
        {
            "timestamp": aligned_frame["target_timestamp"].to_numpy(),
            "y_true": y_true,
            "y_base": y_base,
            "y_final": y_final,
            "residual_base": y_true - y_base,
            "residual_final": y_true - y_final,
        }
    )
    for k in range(5):
        output[f"m{k + 1}"] = masks[:, k]
    for k in range(5):
        output[f"delta{k + 1}"] = deltas[:, k]
    for k in range(5):
        output[f"correction{k + 1}"] = corrections[:, k]
    for column in [
        "N_t",
        "GHI_t",
        "GHI_f",
        "dN_hist",
        "dGHI_f",
        "dBase",
        "target_hour",
        "target_weekday",
        "target_month",
        "is_weekend",
    ]:
        output[column] = aligned_frame[column].to_numpy()

    output.to_csv(
        os.path.join(save_dir, f"{split_name}_predictions_expert_corrected.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    overall_metrics = metrics_rows_base_vs_expert(y_true, y_base, y_final)
    pd.DataFrame(overall_metrics).to_csv(
        os.path.join(save_dir, f"{split_name}_metrics_base_vs_expert.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    for k in range(5):
        active = masks[:, k] > 0.5
        active_count = int(np.sum(active))
        if np.any(active):
            active_metrics = metrics_rows_base_vs_expert(
                y_true[active], y_base[active], y_final[active]
            )
            residual_base_active = y_true[active] - y_base[active]
            residual_final_active = y_true[active] - y_final[active]
            if SIGN_PRIORS[k] > 0:
                direction_match_ratio = np.mean(residual_base_active > 0.0)
            else:
                direction_match_ratio = np.mean(residual_base_active < 0.0)
            active_summary = {
                "active_count": active_count,
                "mean_residual_base": float(np.mean(residual_base_active)),
                "mean_residual_final": float(np.mean(residual_final_active)),
                "mean_delta_k": float(np.mean(deltas[active, k])),
                "mean_correction_k": float(np.mean(corrections[active, k])),
                "direction_match_ratio": float(direction_match_ratio),
            }
        else:
            active_metrics = [
                {
                    "model": model_name,
                    "N": 0,
                    "MAE": np.nan,
                    "MSE": np.nan,
                    "RMSE": np.nan,
                    "MAPE_percent": np.nan,
                    "R2": np.nan,
                }
                for model_name in ["base", "expert_corrected"]
            ]
            active_summary = {
                "active_count": 0,
                "mean_residual_base": np.nan,
                "mean_residual_final": np.nan,
                "mean_delta_k": np.nan,
                "mean_correction_k": np.nan,
                "direction_match_ratio": np.nan,
            }
        for row in active_metrics:
            row.update(active_summary)
        pd.DataFrame(active_metrics).to_csv(
            os.path.join(
                save_dir,
                f"{split_name}_expert_{k + 1}_active_metrics.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

    return overall_metrics


def main():
    args = parse_args()
    if args.enable_expert_correction and args.horizon != 1:
        raise ValueError("Expert correction currently supports horizon=1 only.")

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    df, datasets, feature_scaler, target_scaler, split_info, split_dfs = load_and_prepare_data(
        args.data_path,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        feature_mode=args.feature_mode,
        ghi_col=args.ghi_col,
        temp_col=args.temp_col,
        wind_col=args.wind_col,
    )

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    model_cfg = ModelConfig(
        use_attention=True,
        conv1_channels=args.conv1_channels,
        conv2_channels=args.conv2_channels,
        lstm_hidden1=args.lstm_hidden1,
        lstm_hidden2=args.lstm_hidden2,
        attn_units=args.attn_units,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(split_info["input_dim"])
    model = CNNLSTMModel(input_dim=input_dim, output_dim=args.horizon, cfg=model_cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "data_path": args.data_path,
        "data_path_abs": os.path.abspath(args.data_path),
        "save_dir": args.save_dir,
        "save_dir_abs": os.path.abspath(args.save_dir),
        "feature_mode": args.feature_mode,
        "feature_cols": split_info["feature_cols"],
        "future_weather_mode": split_info["future_weather_mode"],
        "future_weather_cols": split_info["future_weather_cols"],
        "input_dim": input_dim,
        "target": "net_load_multi_step_direct",
        "net_load_definition": "gc - gg",
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "method": "direct_net_load",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "early_stop_patience": args.patience,
        "enable_expert_correction": bool(args.enable_expert_correction),
        "expert_epochs": args.expert_epochs,
        "expert_learning_rate": args.expert_lr,
        "expert_patience": args.expert_patience,
        "expert_batch_size": args.expert_batch_size,
        "expert_lambda": args.expert_lambda,
        "expert_dropout": args.expert_dropout,
        "expert_hidden1": args.expert_hidden1,
        "expert_hidden2": args.expert_hidden2,
        "expert_amps": [
            args.expert_amp1,
            args.expert_amp2,
            args.expert_amp3,
            args.expert_amp4,
            args.expert_amp5,
        ],
        "optimizer": "Adam",
        "loss": "MSELoss in scaled net_load space",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "model": asdict(model_cfg),
        "split_info": split_info,
        "feature_scaler": scaler_to_dict(feature_scaler),
        "target_scaler": scaler_to_dict(target_scaler),
    }

    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Data rows: raw={split_info['raw_rows']}, clean={split_info['clean_rows']}")
    print(
        "Train/val/test samples: "
        f"{split_info['train_samples']}, {split_info['val_samples']}, {split_info['test_samples']}"
    )
    print(
        "net_load scaled range: "
        f"train=[{split_info['train_scaled_min']:.6f}, {split_info['train_scaled_max']:.6f}], "
        f"all=[{split_info['all_scaled_min']:.6f}, {split_info['all_scaled_max']:.6f}]"
    )
    print(f"Input features ({input_dim}): {', '.join(split_info['feature_cols'])}")
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Device: {device}")

    logs = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    final_model_path = os.path.join(args.save_dir, "final_model.pth")
    log_path = os.path.join(args.save_dir, "training_log.csv")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device, criterion, optimizer=None)

        row = {
            "epoch": epoch,
            "train_mse_loss": train_loss,
            "val_mse_loss": val_loss,
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"TrainMSE={train_loss:.6f} | ValMSE={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(best_model_path, model, model_cfg, config, epoch, best_val_loss)
        else:
            no_improve_epochs += 1

        if args.patience > 0 and no_improve_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation MSE was {best_val_loss:.6f} at epoch {best_epoch}."
            )
            break

    save_checkpoint(final_model_path, model, model_cfg, config, logs[-1]["epoch"], logs[-1]["val_mse_loss"])
    save_loss_curve(logs, os.path.join(args.save_dir, "train_val_loss_curve.png"))

    checkpoint = load_checkpoint(best_model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_scaled, true_scaled, timestamps_joined = collect_predictions(model, test_loader, device)
    pred_real = inverse_transform_1d(target_scaler, pred_scaled.reshape(-1)).reshape(pred_scaled.shape)
    true_real = inverse_transform_1d(target_scaler, true_scaled.reshape(-1)).reshape(true_scaled.shape)

    pred_data = {}
    timestamp_steps = [str(ts).split("|") for ts in timestamps_joined]
    for step in range(args.horizon):
        pred_data[f"timestamp_step_{step + 1}"] = [
            ts_parts[step] if step < len(ts_parts) else "" for ts_parts in timestamp_steps
        ]
        pred_data[f"y_true_step_{step + 1}_scaled"] = true_scaled[:, step]
        pred_data[f"y_pred_step_{step + 1}_scaled"] = pred_scaled[:, step]
        pred_data[f"y_true_step_{step + 1}"] = true_real[:, step]
        pred_data[f"y_pred_step_{step + 1}"] = pred_real[:, step]
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(os.path.join(args.save_dir, "test_predictions.csv"), index=False, encoding="utf-8-sig")

    long_rows = []
    for sample_idx in range(pred_scaled.shape[0]):
        for step in range(args.horizon):
            long_rows.append(
                {
                    "sample_index": int(sample_idx),
                    "step": int(step + 1),
                    "timestamp": pred_data[f"timestamp_step_{step + 1}"][sample_idx],
                    "y_true_scaled": float(true_scaled[sample_idx, step]),
                    "y_pred_scaled": float(pred_scaled[sample_idx, step]),
                    "y_true": float(true_real[sample_idx, step]),
                    "y_pred": float(pred_real[sample_idx, step]),
                }
            )
    long_pred_df = pd.DataFrame(long_rows)
    long_pred_df.to_csv(os.path.join(args.save_dir, "test_predictions_long.csv"), index=False, encoding="utf-8-sig")

    test_metrics = compute_metrics(true_real.reshape(-1), pred_real.reshape(-1))
    metrics_row = {"N": int(true_real.size), "horizon": int(args.horizon)}
    metrics_row.update(test_metrics)
    for step in range(args.horizon):
        step_metrics = compute_metrics(true_real[:, step], pred_real[:, step])
        for key, value in step_metrics.items():
            metrics_row[f"step_{step + 1}_{key}"] = value
    pd.DataFrame([metrics_row]).to_csv(
        os.path.join(args.save_dir, "test_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_test_plot(os.path.join(args.save_dir, "test_prediction_multi_step.png"), true_real, pred_real)

    expert_test_metrics = None
    if args.enable_expert_correction:
        print("Starting expert correction stage...")
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        train_base_scaled, train_true_scaled, _ = collect_predictions(model, train_eval_loader, device)
        val_base_scaled, val_true_scaled, _ = collect_predictions(model, val_loader, device)
        test_base_scaled, test_true_scaled, _ = collect_predictions(model, test_loader, device)

        train_base_real = inverse_transform_1d(target_scaler, train_base_scaled.reshape(-1))
        train_true_real = inverse_transform_1d(target_scaler, train_true_scaled.reshape(-1))
        val_base_real = inverse_transform_1d(target_scaler, val_base_scaled.reshape(-1))
        val_true_real = inverse_transform_1d(target_scaler, val_true_scaled.reshape(-1))
        test_base_real = inverse_transform_1d(target_scaler, test_base_scaled.reshape(-1))
        test_true_real = inverse_transform_1d(target_scaler, test_true_scaled.reshape(-1))

        expert_frames = {
            "train": build_expert_frame_for_split(split_dfs["train"], train_base_real, train_true_real, args.seq_len),
            "val": build_expert_frame_for_split(split_dfs["val"], val_base_real, val_true_real, args.seq_len),
            "test": build_expert_frame_for_split(split_dfs["test"], test_base_real, test_true_real, args.seq_len),
        }

        gate_thresholds = {
            "G_off": 10.0,
            "G_pv": 100.0,
            "G_day": 100.0,
            "G_up": 60.0,
            "G_down": 60.0,
            "N_mid": float(np.quantile(split_dfs["train"]["net_load"].to_numpy(dtype=np.float64), 0.50)),
        }
        expert_frames = {
            split_name: add_expert_gate_masks(frame, gate_thresholds)
            for split_name, frame in expert_frames.items()
        }

        gate_summary = gate_activation_summary(expert_frames)
        gate_summary.to_csv(
            os.path.join(args.save_dir, "expert_gate_activation_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        pv_expert_input_scaler = make_standard_scaler()
        load_expert_input_scaler = make_standard_scaler()

        train_pv_input = expert_frames["train"][PV_EXPERT_INPUT_COLS].to_numpy(dtype=np.float64)
        val_pv_input = expert_frames["val"][PV_EXPERT_INPUT_COLS].to_numpy(dtype=np.float64)
        test_pv_input = expert_frames["test"][PV_EXPERT_INPUT_COLS].to_numpy(dtype=np.float64)
        train_load_input = expert_frames["train"][LOAD_EXPERT_INPUT_COLS].to_numpy(dtype=np.float64)
        val_load_input = expert_frames["val"][LOAD_EXPERT_INPUT_COLS].to_numpy(dtype=np.float64)
        test_load_input = expert_frames["test"][LOAD_EXPERT_INPUT_COLS].to_numpy(dtype=np.float64)

        train_pv_scaled = pv_expert_input_scaler.fit_transform(train_pv_input).astype(np.float32)
        val_pv_scaled = pv_expert_input_scaler.transform(val_pv_input).astype(np.float32)
        test_pv_scaled = pv_expert_input_scaler.transform(test_pv_input).astype(np.float32)
        train_load_scaled = load_expert_input_scaler.fit_transform(train_load_input).astype(np.float32)
        val_load_scaled = load_expert_input_scaler.transform(val_load_input).astype(np.float32)
        test_load_scaled = load_expert_input_scaler.transform(test_load_input).astype(np.float32)

        expert_datasets = {
            "train": make_expert_dataset(
                expert_frames["train"], train_pv_scaled, train_load_scaled
            ),
            "val": make_expert_dataset(
                expert_frames["val"], val_pv_scaled, val_load_scaled
            ),
            "test": make_expert_dataset(
                expert_frames["test"], test_pv_scaled, test_load_scaled
            ),
        }
        expert_train_loader = DataLoader(
            expert_datasets["train"],
            batch_size=args.expert_batch_size,
            shuffle=True,
        )
        expert_train_eval_loader = DataLoader(
            expert_datasets["train"],
            batch_size=args.expert_batch_size,
            shuffle=False,
        )
        expert_val_loader = DataLoader(
            expert_datasets["val"],
            batch_size=args.expert_batch_size,
            shuffle=False,
        )
        expert_test_loader = DataLoader(
            expert_datasets["test"],
            batch_size=args.expert_batch_size,
            shuffle=False,
        )

        expert_amps = [
            args.expert_amp1,
            args.expert_amp2,
            args.expert_amp3,
            args.expert_amp4,
            args.expert_amp5,
        ]
        expert_config = {
            "PV_EXPERT_INPUT_COLS": PV_EXPERT_INPUT_COLS,
            "LOAD_EXPERT_INPUT_COLS": LOAD_EXPERT_INPUT_COLS,
            "expert_names": EXPERT_NAMES,
            "gate_thresholds": gate_thresholds,
            "future_weather_mode": split_info["future_weather_mode"],
            "future_weather_cols": split_info["future_weather_cols"],
            "pv_expert_input_scaler": standard_scaler_to_dict(pv_expert_input_scaler),
            "load_expert_input_scaler": standard_scaler_to_dict(load_expert_input_scaler),
            "expert_amps": expert_amps,
            "expert_lambda": args.expert_lambda,
            "sign_lambda": args.sign_lambda,
            "corr_lambda": args.corr_lambda,
            "sign_priors": SIGN_PRIORS,
            "expert_dropout": args.expert_dropout,
            "expert_hidden1": args.expert_hidden1,
            "expert_hidden2": args.expert_hidden2,
            "expert_gate_definitions": {
                "m1": "7 <= target_hour < 9 and GHI_t <= G_pv and GHI_f > G_pv and dGHI_f > G_up",
                "m2": "9 <= target_hour < 15 and GHI_f > G_day",
                "m3": "15 <= target_hour < 20 and (GHI_f < G_pv or dGHI_f < -G_down)",
                "m4": "(5 <= target_hour < 8 or 16 <= target_hour < 20) and (dN_hist > 0.5 or dBase > 0.3)",
                "m5": "20 <= target_hour < 23 and (dN_hist < -0.5 or dBase < 0) and (N_t > N_mid or y_base > N_mid)",
            },
            "note": "PV experts and load experts use separated inputs. Expert outputs are bidirectional tanh residual corrections with physical sign-prior regularization.",
        }
        with open(os.path.join(args.save_dir, "expert_config.json"), "w", encoding="utf-8") as f:
            json.dump(expert_config, f, ensure_ascii=False, indent=2)
        config["expert_config"] = expert_config
        with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        expert_model = ExpertCorrectionModel(
            pv_input_dim=len(PV_EXPERT_INPUT_COLS),
            load_input_dim=len(LOAD_EXPERT_INPUT_COLS),
            hidden1=args.expert_hidden1,
            hidden2=args.expert_hidden2,
            dropout=args.expert_dropout,
            amps=expert_amps,
        ).to(device)
        expert_optimizer = torch.optim.Adam(expert_model.parameters(), lr=args.expert_lr)
        best_expert_path = os.path.join(args.save_dir, "best_expert_model.pth")
        final_expert_path = os.path.join(args.save_dir, "final_expert_model.pth")
        expert_log_path = os.path.join(args.save_dir, "expert_training_log.csv")

        expert_logs = []
        best_val_rmse = float("inf")
        best_expert_epoch = 0
        no_improve_expert_epochs = 0

        for epoch in range(1, args.expert_epochs + 1):
            train_stats = run_expert_epoch(
                expert_model,
                expert_train_loader,
                device,
                optimizer=expert_optimizer,
                expert_lambda=args.expert_lambda,
                sign_lambda=args.sign_lambda,
                corr_lambda=args.corr_lambda,
            )
            val_pred = predict_expert(
                expert_model,
                expert_val_loader,
                device,
                expert_lambda=args.expert_lambda,
                sign_lambda=args.sign_lambda,
                corr_lambda=args.corr_lambda,
            )
            val_base_metrics = compute_metrics(val_pred["y_true"], val_pred["y_base"])
            val_expert_metrics = compute_metrics(val_pred["y_true"], val_pred["y_final"])
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_final_loss": train_stats["final_loss"],
                "train_expert_loss": train_stats["expert_loss"],
                "train_sign_loss": train_stats["sign_loss"],
                "train_correction_l1": train_stats["correction_l1"],
                "val_loss": val_pred["loss"],
                "val_final_loss": val_pred["final_loss"],
                "val_expert_loss": val_pred["expert_loss"],
                "val_sign_loss": val_pred["sign_loss"],
                "val_correction_l1": val_pred["correction_l1"],
                "val_RMSE_base": val_base_metrics["RMSE"],
                "val_RMSE_expert": val_expert_metrics["RMSE"],
                "val_MAE_base": val_base_metrics["MAE"],
                "val_MAE_expert": val_expert_metrics["MAE"],
                "val_R2_base": val_base_metrics["R2"],
                "val_R2_expert": val_expert_metrics["R2"],
            }
            expert_logs.append(row)
            pd.DataFrame(expert_logs).to_csv(expert_log_path, index=False, encoding="utf-8-sig")

            print(
                f"Expert Epoch [{epoch:03d}/{args.expert_epochs:03d}] | "
                f"TrainLoss={train_stats['loss']:.6f} | "
                f"ValRMSE base={val_base_metrics['RMSE']:.6f}, expert={val_expert_metrics['RMSE']:.6f}"
            )

            if val_expert_metrics["RMSE"] < best_val_rmse:
                best_val_rmse = val_expert_metrics["RMSE"]
                best_expert_epoch = epoch
                no_improve_expert_epochs = 0
                torch.save(
                    {
                        "model_state_dict": expert_model.state_dict(),
                        "expert_config": expert_config,
                        "epoch": int(epoch),
                        "val_RMSE_expert": float(best_val_rmse),
                    },
                    best_expert_path,
                )
            else:
                no_improve_expert_epochs += 1

            if args.expert_patience > 0 and no_improve_expert_epochs >= args.expert_patience:
                print(
                    f"Expert early stopping at epoch {epoch}. "
                    f"Best validation expert RMSE was {best_val_rmse:.6f} at epoch {best_expert_epoch}."
                )
                break

        torch.save(
            {
                "model_state_dict": expert_model.state_dict(),
                "expert_config": expert_config,
                "epoch": int(expert_logs[-1]["epoch"]),
                "val_RMSE_expert": float(best_val_rmse),
            },
            final_expert_path,
        )

        best_expert_checkpoint = load_checkpoint(best_expert_path, device)
        expert_model.load_state_dict(best_expert_checkpoint["model_state_dict"])
        train_expert_pred = predict_expert(
            expert_model,
            expert_train_eval_loader,
            device,
            expert_lambda=args.expert_lambda,
            sign_lambda=args.sign_lambda,
            corr_lambda=args.corr_lambda,
        )
        val_expert_pred = predict_expert(
            expert_model,
            expert_val_loader,
            device,
            expert_lambda=args.expert_lambda,
            sign_lambda=args.sign_lambda,
            corr_lambda=args.corr_lambda,
        )
        test_expert_pred = predict_expert(
            expert_model,
            expert_test_loader,
            device,
            expert_lambda=args.expert_lambda,
            sign_lambda=args.sign_lambda,
            corr_lambda=args.corr_lambda,
        )

        save_expert_split_outputs(
            args.save_dir,
            "train",
            expert_frames["train"],
            train_expert_pred,
        )
        save_expert_split_outputs(
            args.save_dir,
            "val",
            expert_frames["val"],
            val_expert_pred,
        )
        expert_test_metrics = save_expert_split_outputs(
            args.save_dir,
            "test",
            expert_frames["test"],
            test_expert_pred,
        )
        save_expert_prediction_plot(
            os.path.join(args.save_dir, "test_prediction_expert_corrected.png"),
            test_expert_pred["y_true"],
            test_expert_pred["y_base"],
            test_expert_pred["y_final"],
        )

    print(f"Final best validation MSE loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(
        "Test metrics: "
        f"MAE={test_metrics['MAE']:.6f}, "
        f"RMSE={test_metrics['RMSE']:.6f}, "
        f"R2={test_metrics['R2']:.6f}"
    )
    if expert_test_metrics is not None:
        expert_row = next(row for row in expert_test_metrics if row["model"] == "expert_corrected")
        print(
            "Expert-corrected test metrics: "
            f"MAE={expert_row['MAE']:.6f}, "
            f"RMSE={expert_row['RMSE']:.6f}, "
            f"R2={expert_row['R2']:.6f}"
        )
    print(f"Output directory: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
