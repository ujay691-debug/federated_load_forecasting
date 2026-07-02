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


EXPERIMENT_NAME = "cnn-lstm-本地预测-1"
FEATURE_TO_COLUMN = {
    "net_load": "net_load",
    "ghi": "ghi_wm2",
    "temperature": "temp_c",
    "wind": "wind10m_ms",
}


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
    """Historical multivariate inputs with future net-load targets."""

    def __init__(self, scaled_features, scaled_target, timestamps, seq_len, horizon):
        self.scaled_features = np.asarray(scaled_features, dtype=np.float32)
        self.scaled_target = np.asarray(scaled_target, dtype=np.float32).reshape(-1)
        self.timestamps = np.asarray(timestamps)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        if self.scaled_features.ndim != 2:
            raise ValueError("scaled_features must have shape [N, input_dim].")
        if not (
            len(self.scaled_features) == len(self.scaled_target) == len(self.timestamps)
        ):
            raise ValueError(
                "scaled_features, scaled_target, and timestamps must have the same length."
            )
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")

    def __len__(self):
        return max(0, len(self.scaled_target) - self.seq_len - self.horizon + 1)

    def __getitem__(self, idx):
        x = self.scaled_features[idx : idx + self.seq_len]
        y = self.scaled_target[idx + self.seq_len : idx + self.seq_len + self.horizon]
        timestamp_target = "|".join(
            str(ts) for ts in self.timestamps[idx + self.seq_len : idx + self.seq_len + self.horizon]
        )
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), timestamp_target


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN-LSTM local net-load forecasting with configurable historical inputs."
    )
    parser.add_argument(
        "--data-path",
        default="per_client_merged/client_1_load_weather_30min.csv",
        help="CSV path containing load and any selected weather columns.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join("runs", EXPERIMENT_NAME),
        help="Output directory.",
    )
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--input-features",
        nargs="+",
        choices=list(FEATURE_TO_COLUMN),
        default=["net_load", "ghi", "temperature", "wind"],
        metavar="FEATURE",
        help=(
            "Historical inputs. Available: net_load, ghi, temperature, wind. "
            "Default uses all four features."
        ),
    )
    parser.add_argument("--conv1-channels", type=int, default=32)
    parser.add_argument("--conv2-channels", type=int, default=64)
    parser.add_argument("--lstm-hidden1", type=int, default=48)
    parser.add_argument("--lstm-hidden2", type=int, default=24)
    parser.add_argument("--attn-units", type=int, default=24)
    parser.add_argument("--fc-hidden", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.0)
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


def scaler_to_dict(scaler):
    result = {"type": "sklearn.MinMaxScaler" if MinMaxScaler is not None else "SimpleMinMaxScaler"}
    for name in ["data_min_", "data_max_", "data_range_", "scale_", "min_"]:
        if hasattr(scaler, name):
            result[name.rstrip("_")] = np.asarray(getattr(scaler, name), dtype=float).reshape(-1).tolist()
    if hasattr(scaler, "safe_range_"):
        result["safe_range"] = np.asarray(scaler.safe_range_, dtype=float).reshape(-1).tolist()
    return result


def resolve_input_features(input_features):
    feature_names = list(input_features)
    if not feature_names:
        raise ValueError("At least one --input-features value is required.")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError(f"Duplicate input features are not allowed: {feature_names}")
    unknown = sorted(set(feature_names).difference(FEATURE_TO_COLUMN))
    if unknown:
        raise ValueError(f"Unsupported input features: {', '.join(unknown)}")
    return feature_names, [FEATURE_TO_COLUMN[name] for name in feature_names]


def load_and_prepare_data(
    data_path,
    seq_len,
    horizon,
    train_ratio,
    val_ratio,
    input_features,
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Please confirm --data-path."
        )

    df = pd.read_csv(data_path)
    raw_rows = len(df)
    feature_names, input_columns = resolve_input_features(input_features)
    required = {"timestamp", "gc", "gg"}
    if "ghi" in feature_names:
        required.add("ghi_wm2")
    if "wind" in feature_names:
        required.add("wind10m_ms")
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}.")
    if "temperature" in feature_names and not {
        "temp2m_c",
        "temp2m_k",
    }.intersection(df.columns):
        raise ValueError(
            "Temperature input requires temp2m_c or temp2m_k in the CSV."
        )

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["gc"] = pd.to_numeric(df["gc"], errors="coerce")
    df["gg"] = pd.to_numeric(df["gg"], errors="coerce")
    df["net_load"] = df["gc"] - df["gg"]

    if "ghi" in feature_names:
        df["ghi_wm2"] = pd.to_numeric(df["ghi_wm2"], errors="coerce")
    if "wind" in feature_names:
        df["wind10m_ms"] = pd.to_numeric(df["wind10m_ms"], errors="coerce")
    temperature_source = None
    if "temperature" in feature_names:
        if "temp2m_c" in df.columns:
            df["temp_c"] = pd.to_numeric(df["temp2m_c"], errors="coerce")
            temperature_source = "temp2m_c"
        else:
            df["temp_c"] = pd.to_numeric(df["temp2m_k"], errors="coerce") - 273.15
            temperature_source = "temp2m_k_minus_273.15"

    clean_columns = ["timestamp", "gc", "gg", "net_load", *input_columns]
    clean_columns = list(dict.fromkeys(clean_columns))
    df = df.dropna(subset=clean_columns)
    df = df.sort_values("timestamp").reset_index(drop=True)
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
    train_features_scaled = feature_scaler.fit_transform(
        train_df[input_columns].values
    ).astype(np.float32)
    val_features_scaled = feature_scaler.transform(
        val_df[input_columns].values
    ).astype(np.float32)
    test_features_scaled = feature_scaler.transform(
        test_df[input_columns].values
    ).astype(np.float32)
    train_target_scaled = target_scaler.fit_transform(
        train_df[["net_load"]].values
    ).astype(np.float32).reshape(-1)
    val_target_scaled = target_scaler.transform(
        val_df[["net_load"]].values
    ).astype(np.float32).reshape(-1)
    test_target_scaled = target_scaler.transform(
        test_df[["net_load"]].values
    ).astype(np.float32).reshape(-1)
    all_target_scaled = np.concatenate(
        [train_target_scaled, val_target_scaled, test_target_scaled]
    )

    datasets = {
        "train": NetLoadNextStepDataset(
            train_features_scaled,
            train_target_scaled,
            train_df["timestamp"].values,
            seq_len,
            horizon,
        ),
        "val": NetLoadNextStepDataset(
            val_features_scaled,
            val_target_scaled,
            val_df["timestamp"].values,
            seq_len,
            horizon,
        ),
        "test": NetLoadNextStepDataset(
            test_features_scaled,
            test_target_scaled,
            test_df["timestamp"].values,
            seq_len,
            horizon,
        ),
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
        "input_feature_names": feature_names,
        "input_columns": input_columns,
        "input_dim": int(len(input_columns)),
        "temperature_source": temperature_source,
        "train_target_scaled_min": float(np.min(train_target_scaled)),
        "train_target_scaled_max": float(np.max(train_target_scaled)),
        "all_target_scaled_min": float(np.min(all_target_scaled)),
        "all_target_scaled_max": float(np.max(all_target_scaled)),
        "train_feature_scaled_min": {
            name: float(train_features_scaled[:, idx].min())
            for idx, name in enumerate(feature_names)
        },
        "train_feature_scaled_max": {
            name: float(train_features_scaled[:, idx].max())
            for idx, name in enumerate(feature_names)
        },
    }
    return (
        df,
        datasets,
        target_scaler,
        feature_scaler,
        split_info,
        input_columns,
    )


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    sample_count = 0

    for x, y, _ in loader:
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
        for x, y, ts in loader:
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
    plt.title("CNN-LSTM Local Forecast Train/Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_test_plot(
    path,
    y_true,
    y_pred,
    max_points=300,
    title="CNN-LSTM Local Forecast Test Prediction",
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = min(len(y_true), int(max_points))
    x_axis = np.arange(n)
    plt.figure(figsize=(11, 5))
    plt.plot(x_axis, y_true[:n], label="True net load")
    plt.plot(x_axis, y_pred[:n], label="Predicted net load")
    plt.xlabel("Test sample")
    plt.ylabel("Net load")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_random_train_7day_evaluations(
    save_dir,
    target_scaler,
    pred_scaled,
    true_scaled,
    timestamps_joined,
    seed,
    num_windows=3,
    days=7,
):
    """Save reproducible random, timestamp-continuous training windows.

    The first forecast step is used when horizon > 1, so adjacent samples map
    to adjacent target timestamps without overlapping forecast steps.
    """

    pred_scaled = np.asarray(pred_scaled, dtype=np.float32)
    true_scaled = np.asarray(true_scaled, dtype=np.float32)
    if pred_scaled.ndim != 2 or true_scaled.ndim != 2:
        raise ValueError("Training predictions and targets must have shape [N, horizon].")
    if pred_scaled.shape != true_scaled.shape:
        raise ValueError("Training prediction and target shapes do not match.")

    first_timestamps = np.asarray(
        [str(value).split("|")[0] for value in timestamps_joined], dtype=object
    )
    parsed_timestamps = pd.to_datetime(first_timestamps, errors="coerce")
    if parsed_timestamps.isna().any():
        raise ValueError("Cannot parse one or more training target timestamps.")
    if len(parsed_timestamps) < 2:
        raise ValueError("Not enough training predictions to infer sampling frequency.")

    timestamps_ns = parsed_timestamps.astype("int64").to_numpy()
    diffs_ns = np.diff(timestamps_ns)
    positive_diffs = diffs_ns[diffs_ns > 0]
    if positive_diffs.size == 0:
        raise ValueError("Training target timestamps are not increasing.")
    step_ns = int(np.median(positive_diffs))
    day_ns = 24 * 60 * 60 * 1_000_000_000
    points_per_day = int(round(day_ns / step_ns))
    if points_per_day <= 0 or not np.isclose(
        points_per_day * step_ns, day_ns, rtol=0.0, atol=1_000_000
    ):
        raise ValueError(
            f"Sampling interval {pd.to_timedelta(step_ns, unit='ns')} does not divide one day."
        )
    window_size = int(days) * points_per_day
    if len(parsed_timestamps) < window_size:
        raise ValueError(
            f"Training predictions contain {len(parsed_timestamps)} points, "
            f"fewer than the required {window_size} for {days} days."
        )

    # A valid start has exactly the inferred interval between every adjacent
    # target timestamp in the complete seven-day window.
    bad_step = (diffs_ns != step_ns).astype(np.int64)
    bad_prefix = np.concatenate([[0], np.cumsum(bad_step)])
    candidate_starts = np.arange(len(parsed_timestamps) - window_size + 1)
    bad_counts = (
        bad_prefix[candidate_starts + window_size - 1]
        - bad_prefix[candidate_starts]
    )
    valid_starts = candidate_starts[bad_counts == 0]
    if len(valid_starts) < int(num_windows):
        raise ValueError(
            f"Only {len(valid_starts)} continuous {days}-day training windows are available; "
            f"cannot sample {num_windows}."
        )

    rng = np.random.default_rng(int(seed))
    selected_starts = rng.choice(valid_starts, size=int(num_windows), replace=False)
    pred_step_scaled = pred_scaled[:, 0]
    true_step_scaled = true_scaled[:, 0]
    pred_step_real = inverse_transform_1d(target_scaler, pred_step_scaled)
    true_step_real = inverse_transform_1d(target_scaler, true_step_scaled)

    prediction_frames = []
    metric_rows = []
    for selection_id, start in enumerate(selected_starts, start=1):
        start = int(start)
        end = start + window_size
        window_true = true_step_real[start:end]
        window_pred = pred_step_real[start:end]
        start_timestamp = str(first_timestamps[start])
        end_timestamp = str(first_timestamps[end - 1])
        metric_rows.append(
            {
                "selection_id": selection_id,
                "sample_start_index": start,
                "sample_end_index": end - 1,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "days": int(days),
                "N": int(window_size),
                **compute_metrics(window_true, window_pred),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "selection_id": selection_id,
                    "sample_index": np.arange(start, end),
                    "timestamp": first_timestamps[start:end],
                    "y_true_scaled": true_step_scaled[start:end],
                    "y_pred_scaled": pred_step_scaled[start:end],
                    "y_true": window_true,
                    "y_pred": window_pred,
                }
            )
        )
        save_test_plot(
            os.path.join(
                save_dir, f"train_random_7day_prediction_{selection_id}.png"
            ),
            window_true,
            window_pred,
            max_points=window_size,
            title=(
                f"Train Random 7-Day Window {selection_id}: "
                f"{start_timestamp} to {end_timestamp}"
            ),
        )

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        os.path.join(save_dir, "train_random_7day_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(save_dir, "train_random_7day_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return metric_rows


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    (
        df,
        datasets,
        target_scaler,
        feature_scaler,
        split_info,
        input_columns,
    ) = load_and_prepare_data(
        args.data_path,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        input_features=args.input_features,
    )

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=False
    )
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
    model = CNNLSTMModel(
        input_dim=len(input_columns),
        output_dim=args.horizon,
        cfg=model_cfg,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "experiment_name": EXPERIMENT_NAME,
        "data_path": args.data_path,
        "data_path_abs": os.path.abspath(args.data_path),
        "save_dir": args.save_dir,
        "save_dir_abs": os.path.abspath(args.save_dir),
        "input_feature_names": list(args.input_features),
        "input_columns": input_columns,
        "input_dim": int(len(input_columns)),
        "input_policy": "historical sequences only; no future weather or future net_load inputs",
        "target": "net_load_multi_step_direct",
        "net_load_definition": "gc - gg",
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "method": EXPERIMENT_NAME,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "early_stop_patience": args.patience,
        "optimizer": "Adam",
        "loss": "MSELoss in scaled net_load space",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "model": asdict(model_cfg),
        "split_info": split_info,
        "target_scaler": scaler_to_dict(target_scaler),
        "feature_scaler": scaler_to_dict(feature_scaler),
        "train_random_window_evaluation": {
            "enabled": True,
            "days_per_window": 7,
            "num_random_windows": 3,
            "random_seed": args.seed,
            "forecast_step": 1,
            "requires_continuous_timestamps": True,
        },
    }

    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Data rows: raw={split_info['raw_rows']}, clean={split_info['clean_rows']}")
    print(
        "Train/val/test samples: "
        f"{split_info['train_samples']}, {split_info['val_samples']}, {split_info['test_samples']}"
    )
    print(
        "net_load target scaled range: "
        f"train=[{split_info['train_target_scaled_min']:.6f}, "
        f"{split_info['train_target_scaled_max']:.6f}], "
        f"all=[{split_info['all_target_scaled_min']:.6f}, "
        f"{split_info['all_target_scaled_max']:.6f}]"
    )
    print(f"Input features: {args.input_features} -> columns {input_columns}")
    print(f"Model input dimension: {len(input_columns)}")
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

    train_pred_scaled, train_true_scaled, train_timestamps = collect_predictions(
        model, train_eval_loader, device
    )
    train_window_metrics = save_random_train_7day_evaluations(
        save_dir=args.save_dir,
        target_scaler=target_scaler,
        pred_scaled=train_pred_scaled,
        true_scaled=train_true_scaled,
        timestamps_joined=train_timestamps,
        seed=args.seed,
        num_windows=3,
        days=7,
    )
    for row in train_window_metrics:
        print(
            f"Train random 7-day window {row['selection_id']}: "
            f"{row['start_timestamp']} to {row['end_timestamp']} | "
            f"MAE={row['MAE']:.6f}, RMSE={row['RMSE']:.6f}, R2={row['R2']:.6f}"
        )

    pred_scaled, true_scaled, timestamps_joined = collect_predictions(model, test_loader, device)
    pred_real = inverse_transform_1d(
        target_scaler, pred_scaled.reshape(-1)
    ).reshape(pred_scaled.shape)
    true_real = inverse_transform_1d(
        target_scaler, true_scaled.reshape(-1)
    ).reshape(true_scaled.shape)

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

    print(f"Final best validation MSE loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(
        "Test metrics: "
        f"MAE={test_metrics['MAE']:.6f}, "
        f"RMSE={test_metrics['RMSE']:.6f}, "
        f"R2={test_metrics['R2']:.6f}"
    )
    print(f"Output directory: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
