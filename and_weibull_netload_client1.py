import argparse
import json
import os
import random

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
except ImportError:
    MinMaxScaler = None


class SimpleMinMaxScaler:
    """Fallback scaler used only when sklearn is unavailable."""

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


class NetLoadWindowDataset(Dataset):
    """Builds x_cur, o_cur, y_future windows with batch-first tensors."""

    def __init__(self, scaled_values, time_indices, seq_len):
        self.scaled_values = np.asarray(scaled_values, dtype=np.float32).reshape(-1)
        self.time_indices = np.asarray(time_indices, dtype=np.int64).reshape(-1)
        self.seq_len = int(seq_len)

        if len(self.scaled_values) != len(self.time_indices):
            raise ValueError("scaled_values and time_indices must have the same length.")

    def __len__(self):
        return max(0, len(self.scaled_values) - self.seq_len)

    def __getitem__(self, idx):
        # o_cur uses the global row-based time index: 0, 1, 2, ..., T - 1.
        x_cur = self.scaled_values[idx : idx + self.seq_len]
        o_cur = self.time_indices[idx : idx + self.seq_len].astype(np.float32)
        y_future = self.scaled_values[idx + 1 : idx + self.seq_len + 1]
        target_time_idx = self.time_indices[idx + self.seq_len]

        return (
            torch.from_numpy(x_cur.copy()),
            torch.from_numpy(o_cur.copy()),
            torch.from_numpy(y_future.copy()),
            torch.tensor(target_time_idx, dtype=torch.long),
        )


class WeibullAttentionLSTM(nn.Module):
    """LSTM residual dynamics with trainable Weibull attention over time steps."""

    def __init__(self, seq_len=24, input_size=1, hidden_size=24, num_layers=1, eps=1e-8):
        super().__init__()
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, self.seq_len)
        self.register_buffer("tau", torch.arange(1, self.seq_len + 1, dtype=torch.float32))

        self.raw_kappa = nn.Parameter(self._inverse_softplus(1.5))
        self.raw_lambda = nn.Parameter(self._inverse_softplus(max(1.0, self.seq_len / 2.0)))

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def forward(self, residual_window):
        residual_seq = residual_window.unsqueeze(-1)  # [B, 24] -> [B, 24, 1]
        h, _ = self.lstm(residual_seq)  # [B, 24, 24]

        tau = self.tau.to(device=h.device, dtype=h.dtype)
        kappa = F.softplus(self.raw_kappa) + self.eps
        lambda_ = F.softplus(self.raw_lambda) + self.eps

        scaled_tau = tau / lambda_
        alpha = (kappa / lambda_) * torch.pow(scaled_tau, kappa - 1.0)
        alpha = alpha * torch.exp(-torch.pow(scaled_tau, kappa))
        alpha = alpha / (alpha.sum() + self.eps)

        context = torch.sum(h * alpha.view(1, self.seq_len, 1), dim=1)
        return self.out(context)  # [B, 24]


class ANDWeibullModel(nn.Module):
    """PyTorch implementation of the single-client AND-Weibull net-load system."""

    def __init__(self, seq_len=24, lstm_hidden_units=24):
        super().__init__()
        self.seq_len = int(seq_len)

        # Shape branch: periodic input is o_cur, trend input is x_cur.
        self.W0 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.Phi = nn.Parameter(torch.empty(self.seq_len))
        self.a = nn.Parameter(torch.ones(self.seq_len))
        self.trend_layer = nn.Linear(self.seq_len, self.seq_len)
        self.shape_layer = nn.Linear(self.seq_len * 2, self.seq_len)

        # Autoencoder branch follows Table 1 under the paper setting 24 -> 12 -> 6 -> 12 -> 24.
        ae_hidden = max(1, self.seq_len // 2)
        ae_bottleneck = max(1, self.seq_len // 4)
        self.encoder1 = nn.Linear(self.seq_len, ae_hidden)
        self.bottleneck = nn.Linear(ae_hidden, ae_bottleneck)
        self.decoder1 = nn.Linear(ae_bottleneck, ae_hidden)
        self.decoder2 = nn.Linear(ae_hidden, self.seq_len)

        # One shared Weibull-Attention LSTM module is reused by the prediction and rebuild paths.
        # This is essential: f2 gradients update the same residual dynamics module as f1.
        self.shared_weibull_lstm = WeibullAttentionLSTM(
            seq_len=self.seq_len,
            input_size=1,
            hidden_size=lstm_hidden_units,
            num_layers=1,
            eps=1e-8,
        )

        self.W6 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.W7 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self._init_fourier_like_periodic_layer()

    def _init_fourier_like_periodic_layer(self):
        # Fourier-like initialization from Appendix/Table description.
        # omega_k / 24 keeps the first forward pass numerically steadier with global time_idx inputs.
        with torch.no_grad():
            weight = torch.zeros(self.seq_len, self.seq_len, dtype=torch.float32)
            for k in range(self.seq_len):
                omega_k = 2.0 * np.pi * k / self.seq_len
                weight[k, :].fill_(omega_k / self.seq_len)
            self.W0.weight.copy_(weight)

            phi = [
                np.pi / 2.0 + (k % 2) * np.pi / 2.0
                for k in range(self.seq_len)
            ]
            self.Phi.copy_(torch.tensor(phi, dtype=torch.float32))
            self.a.fill_(1.0)

    def _shape_branch(self, x_cur, o_cur):
        periodic = torch.sin(self.W0(o_cur) + self.Phi) * self.a  # [B, 24]
        trend = self.trend_layer(x_cur)  # [B, 24]
        shape_input = torch.cat([periodic, trend], dim=-1)  # [B, 48]
        return self.shape_layer(shape_input)  # [B, 24]

    def _autoencoder_branch(self, x_cur):
        z = torch.tanh(self.encoder1(x_cur))
        z = torch.tanh(self.bottleneck(z))
        z = torch.tanh(self.decoder1(z))
        return torch.tanh(self.decoder2(z))

    def forward(self, x_cur, o_cur):
        shape = self._shape_branch(x_cur, o_cur)

        ae_recon = self._autoencoder_branch(x_cur)
        residual_cur = x_cur - ae_recon

        residual_future = self.shared_weibull_lstm(residual_cur)
        future_pred = self.W6(shape) + self.W7(residual_future)

        # Appendix C rebuild path: reuse the same shared_weibull_lstm for correction.
        delta = x_cur - future_pred
        correction = self.shared_weibull_lstm(delta)
        rebuild_pred = correction + future_pred

        return future_pred, rebuild_pred, ae_recon


LOG_COLUMNS = [
    "epoch",
    "train_total_loss",
    "train_f1_ste",
    "train_f1_fourier",
    "train_f1_mse",
    "train_f2_ste",
    "train_f2_fourier",
    "train_f2_mse",
    "train_f3_mse",
    "val_total_loss",
    "val_f1_ste",
    "val_f1_fourier",
    "val_f1_mse",
    "val_f2_ste",
    "val_f2_fourier",
    "val_f2_mse",
    "val_f3_mse",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-client AND-Weibull net-load forecasting in PyTorch."
    )
    parser.add_argument(
        "--data-path",
        default="per_client_merged/client_1_load_weather_30min.csv",
        help="CSV path containing timestamp, gc, gg columns.",
    )
    parser.add_argument(
        "--save-dir",
        default="runs/and_weibull_netload_client1",
        help="Directory for configs, checkpoints, logs, predictions, and plots.",
    )
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--feature-mode",
        default="net_load_only",
        choices=["net_load_only"],
        help="Reserved for future feature expansion. Current experiment uses net_load history only.",
    )
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
            value = getattr(scaler, name)
            result[name.rstrip("_")] = np.asarray(value, dtype=float).reshape(-1).tolist()
    if hasattr(scaler, "safe_range_"):
        result["safe_range"] = np.asarray(scaler.safe_range_, dtype=float).reshape(-1).tolist()
    return result


def load_and_prepare_data(data_path, seq_len, train_ratio=0.8, val_ratio=0.1):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"找不到数据文件: {data_path}. 请确认 --data-path 是否指向 client_1 CSV。"
        )

    df = pd.read_csv(data_path)
    raw_rows = len(df)
    required_columns = {"timestamp", "gc", "gg"}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"CSV缺少必需列: {', '.join(missing)}. 必需列为 timestamp、gc、gg。")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["gc"] = pd.to_numeric(df["gc"], errors="coerce")
    df["gg"] = pd.to_numeric(df["gg"], errors="coerce")
    df = df.dropna(subset=["timestamp", "gc", "gg"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["net_load"] = df["gc"] - df["gg"]
    # Global time index after chronological sorting: 0, 1, 2, ..., T - 1.
    df["time_idx"] = np.arange(len(df), dtype=np.int64)

    if len(df) <= seq_len + 1:
        raise ValueError(
            f"有效数据行数 {len(df)} 不足以构造 seq_len={seq_len} 的滑动窗口。"
        )

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 必须为正，且二者之和必须小于 1。")

    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if len(split_df) <= seq_len:
            raise ValueError(
                f"{split_name} 数据段行数 {len(split_df)} 不足以构造 seq_len={seq_len} 的窗口。"
            )

    scaler = make_scaler()
    train_scaled = scaler.fit_transform(train_df[["net_load"]].values).astype(np.float32).reshape(-1)
    val_scaled = scaler.transform(val_df[["net_load"]].values).astype(np.float32).reshape(-1)
    test_scaled = scaler.transform(test_df[["net_load"]].values).astype(np.float32).reshape(-1)

    all_scaled = np.concatenate([train_scaled, val_scaled, test_scaled])

    datasets = {
        "train": NetLoadWindowDataset(train_scaled, train_df["time_idx"].values, seq_len),
        "val": NetLoadWindowDataset(val_scaled, val_df["time_idx"].values, seq_len),
        "test": NetLoadWindowDataset(test_scaled, test_df["time_idx"].values, seq_len),
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
        "train_scaled_min": float(np.min(train_scaled)),
        "train_scaled_max": float(np.max(train_scaled)),
        "all_scaled_min": float(np.min(all_scaled)),
        "all_scaled_max": float(np.max(all_scaled)),
    }

    return df, datasets, scaler, split_info


def ste_loss(y_pred, y_true, alpha=0.10, eps=1e-8):
    """STE = alpha * Fourier amplitude MSE + (1 - alpha) * time-domain MSE."""
    mse = F.mse_loss(y_pred, y_true)

    pred_fft = torch.fft.fft(y_pred, dim=-1)
    true_fft = torch.fft.fft(y_true, dim=-1)
    pred_amp = torch.sqrt(pred_fft.real.pow(2) + pred_fft.imag.pow(2) + eps)
    true_amp = torch.sqrt(true_fft.real.pow(2) + true_fft.imag.pow(2) + eps)
    fourier = F.mse_loss(pred_amp, true_amp)

    total = alpha * fourier + (1.0 - alpha) * mse
    return total, fourier, mse


def empty_meter():
    return {
        "total_loss": 0.0,
        "f1_ste": 0.0,
        "f1_fourier": 0.0,
        "f1_mse": 0.0,
        "f2_ste": 0.0,
        "f2_fourier": 0.0,
        "f2_mse": 0.0,
        "f3_mse": 0.0,
    }


def update_meter(meter, batch_size, total_loss, f1_ste, f1_fourier, f1_mse, f2_ste, f2_fourier, f2_mse, f3_mse):
    meter["total_loss"] += float(total_loss) * batch_size
    meter["f1_ste"] += float(f1_ste) * batch_size
    meter["f1_fourier"] += float(f1_fourier) * batch_size
    meter["f1_mse"] += float(f1_mse) * batch_size
    meter["f2_ste"] += float(f2_ste) * batch_size
    meter["f2_fourier"] += float(f2_fourier) * batch_size
    meter["f2_mse"] += float(f2_mse) * batch_size
    meter["f3_mse"] += float(f3_mse) * batch_size


def finalize_meter(meter, sample_count):
    if sample_count <= 0:
        raise ValueError("DataLoader 没有可用样本，无法计算损失。")
    return {key: value / sample_count for key, value in meter.items()}


def run_epoch(model, data_loader, device, alpha, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    meter = empty_meter()
    sample_count = 0

    for x_cur, o_cur, y_future, _ in data_loader:
        x_cur = x_cur.to(device=device, dtype=torch.float32)
        o_cur = o_cur.to(device=device, dtype=torch.float32)
        y_future = y_future.to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            future_pred, rebuild_pred, ae_recon = model(x_cur, o_cur)

            # f1 forecasts the shifted future window y_future.
            f1_ste, f1_fourier, f1_mse = ste_loss(future_pred, y_future, alpha=alpha)
            # f2 rebuilds the current input window through Appendix C's correction path.
            f2_ste, f2_fourier, f2_mse = ste_loss(rebuild_pred, x_cur, alpha=alpha)
            # f3 trains the autoencoder reconstruction of the current window.
            f3_mse = F.mse_loss(ae_recon, x_cur)

            # Equal weights for f1, f2, f3. A single backward pass over the sum is equivalent
            # to computing each gradient separately and accumulating on shared parameters,
            # matching Appendix D's unified backpropagation idea.
            loss_total = f1_ste + f2_ste + f3_mse

            if is_train:
                loss_total.backward()
                optimizer.step()

        batch_size = x_cur.size(0)
        sample_count += batch_size
        update_meter(
            meter,
            batch_size,
            loss_total.item(),
            f1_ste.item(),
            f1_fourier.item(),
            f1_mse.item(),
            f2_ste.item(),
            f2_fourier.item(),
            f2_mse.item(),
            f3_mse.item(),
        )

    return finalize_meter(meter, sample_count)


def print_epoch_log(epoch, epochs, train_metrics, val_metrics):
    print(
        f"Epoch [{epoch:03d}/{epochs:03d}] | "
        f"TrainTotal={train_metrics['total_loss']:.6f} | "
        f"F1_STE={train_metrics['f1_ste']:.6f} "
        f"F1_Fourier={train_metrics['f1_fourier']:.6f} "
        f"F1_MSE={train_metrics['f1_mse']:.6f} | "
        f"F2_STE={train_metrics['f2_ste']:.6f} "
        f"F2_Fourier={train_metrics['f2_fourier']:.6f} "
        f"F2_MSE={train_metrics['f2_mse']:.6f} | "
        f"F3={train_metrics['f3_mse']:.6f} | "
        f"ValTotal={val_metrics['total_loss']:.6f} | "
        f"ValF1_STE={val_metrics['f1_ste']:.6f} "
        f"ValF1_Fourier={val_metrics['f1_fourier']:.6f} "
        f"ValF1_MSE={val_metrics['f1_mse']:.6f} | "
        f"ValF2_STE={val_metrics['f2_ste']:.6f} "
        f"ValF2_Fourier={val_metrics['f2_fourier']:.6f} "
        f"ValF2_MSE={val_metrics['f2_mse']:.6f} | "
        f"ValF3={val_metrics['f3_mse']:.6f}"
    )


def metrics_to_log_row(epoch, train_metrics, val_metrics):
    return {
        "epoch": epoch,
        "train_total_loss": train_metrics["total_loss"],
        "train_f1_ste": train_metrics["f1_ste"],
        "train_f1_fourier": train_metrics["f1_fourier"],
        "train_f1_mse": train_metrics["f1_mse"],
        "train_f2_ste": train_metrics["f2_ste"],
        "train_f2_fourier": train_metrics["f2_fourier"],
        "train_f2_mse": train_metrics["f2_mse"],
        "train_f3_mse": train_metrics["f3_mse"],
        "val_total_loss": val_metrics["total_loss"],
        "val_f1_ste": val_metrics["f1_ste"],
        "val_f1_fourier": val_metrics["f1_fourier"],
        "val_f1_mse": val_metrics["f1_mse"],
        "val_f2_ste": val_metrics["f2_ste"],
        "val_f2_fourier": val_metrics["f2_fourier"],
        "val_f2_mse": val_metrics["f2_mse"],
        "val_f3_mse": val_metrics["f3_mse"],
    }


def save_training_log(logs, path):
    pd.DataFrame(logs, columns=LOG_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def save_checkpoint(path, model, config, epoch, val_total_loss):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": int(epoch),
            "val_total_loss": float(val_total_loss),
        },
        path,
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inverse_transform_1d(scaler, values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)


def collect_test_predictions(model, data_loader, device):
    model.eval()
    pred_windows = []
    true_windows = []
    target_indices = []

    with torch.no_grad():
        for x_cur, o_cur, y_future, target_time_idx in data_loader:
            x_cur = x_cur.to(device=device, dtype=torch.float32)
            o_cur = o_cur.to(device=device, dtype=torch.float32)
            future_pred, _, _ = model(x_cur, o_cur)

            pred_windows.append(future_pred.cpu().numpy())
            true_windows.append(y_future.numpy())
            target_indices.append(target_time_idx.numpy())

    return (
        np.concatenate(pred_windows, axis=0),
        np.concatenate(true_windows, axis=0),
        np.concatenate(target_indices, axis=0).astype(np.int64),
    )


def compute_real_scale_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_pred - y_true

    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    valid = np.abs(y_true) > eps
    if np.any(valid):
        mape_percent = np.mean(np.abs(errors[valid] / y_true[valid])) * 100.0
    else:
        mape_percent = np.nan

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if ss_tot <= eps else 1.0 - ss_res / ss_tot

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape_percent),
        "R2": float(r2),
    }


def save_test_predictions(path, df, target_indices, pred_scaled, true_scaled, scaler, seq_len):
    y_pred_next_scaled = pred_scaled[:, -1]
    y_true_next_scaled = true_scaled[:, -1]
    y_pred_next = inverse_transform_1d(scaler, y_pred_next_scaled)
    y_true_next = inverse_transform_1d(scaler, y_true_next_scaled)

    timestamp_target = (
        df.iloc[target_indices]["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    )

    data = {
        "timestamp_target": timestamp_target,
        "y_true_next_scaled": y_true_next_scaled,
        "y_pred_next_scaled": y_pred_next_scaled,
        "y_true_next": y_true_next,
        "y_pred_next": y_pred_next,
    }

    for step in range(seq_len):
        data[f"future_pred_step_{step + 1}"] = pred_scaled[:, step]
        data[f"future_true_step_{step + 1}"] = true_scaled[:, step]

    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")
    return y_true_next, y_pred_next


def save_loss_curve(logs, path):
    epochs = [row["epoch"] for row in logs]
    train_loss = [row["train_total_loss"] for row in logs]
    val_loss = [row["val_total_loss"] for row in logs]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, label="Train total loss")
    plt.plot(epochs, val_loss, label="Validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AND-Weibull Train/Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_test_plot(path, y_true, y_pred, max_points=300):
    n = min(len(y_true), int(max_points))
    x_axis = np.arange(n)

    plt.figure(figsize=(11, 5))
    plt.plot(x_axis, y_true[:n], label="True next-step net load")
    plt.plot(x_axis, y_pred[:n], label="Predicted next-step net load")
    plt.xlabel("Test sample")
    plt.ylabel("Net load")
    plt.title("AND-Weibull Test Next-Step Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    df, datasets, scaler, split_info = load_and_prepare_data(
        data_path=args.data_path,
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANDWeibullModel(seq_len=args.seq_len, lstm_hidden_units=24).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "data_path": args.data_path,
        "data_path_abs": os.path.abspath(args.data_path),
        "save_dir": args.save_dir,
        "save_dir_abs": os.path.abspath(args.save_dir),
        "feature_mode": args.feature_mode,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "ste_alpha": args.alpha,
        "early_stop_patience": args.patience,
        "optimizer": "Adam",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "lstm_input_size": 1,
        "lstm_hidden_units": 24,
        "lstm_num_layers": 1,
        "loss_weights": {"f1": 1.0, "f2": 1.0, "f3": 1.0},
        "split_info": split_info,
        "scaler": scaler_to_dict(scaler),
    }

    config_path = os.path.join(args.save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"数据行数: raw={split_info['raw_rows']}, clean={split_info['clean_rows']}")
    print(
        "训练、验证、测试样本数: "
        f"{split_info['train_samples']}, {split_info['val_samples']}, {split_info['test_samples']}"
    )
    print(
        "net_load 归一化范围: "
        f"train=[{split_info['train_scaled_min']:.6f}, {split_info['train_scaled_max']:.6f}], "
        f"all=[{split_info['all_scaled_min']:.6f}, {split_info['all_scaled_max']:.6f}]"
    )
    print(f"模型参数量: {count_parameters(model)}")
    print(f"设备: {device}")

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    logs = []
    training_log_path = os.path.join(args.save_dir, "training_log.csv")
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    final_model_path = os.path.join(args.save_dir, "final_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            device=device,
            alpha=args.alpha,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            alpha=args.alpha,
            optimizer=None,
        )

        print_epoch_log(epoch, args.epochs, train_metrics, val_metrics)

        row = metrics_to_log_row(epoch, train_metrics, val_metrics)
        logs.append(row)
        save_training_log(logs, training_log_path)

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(best_model_path, model, config, epoch, best_val_loss)
        else:
            no_improve_epochs += 1

        if args.patience > 0 and no_improve_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation total loss was {best_val_loss:.6f} at epoch {best_epoch}."
            )
            break

    save_checkpoint(final_model_path, model, config, logs[-1]["epoch"], logs[-1]["val_total_loss"])
    save_loss_curve(logs, os.path.join(args.save_dir, "train_val_loss_curve.png"))

    checkpoint = load_checkpoint(best_model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_scaled, true_scaled, target_indices = collect_test_predictions(model, test_loader, device)
    y_true_next, y_pred_next = save_test_predictions(
        path=os.path.join(args.save_dir, "test_predictions.csv"),
        df=df,
        target_indices=target_indices,
        pred_scaled=pred_scaled,
        true_scaled=true_scaled,
        scaler=scaler,
        seq_len=args.seq_len,
    )

    test_metrics = compute_real_scale_metrics(y_true_next, y_pred_next)
    metrics_row = {"N": int(len(y_true_next))}
    metrics_row.update(test_metrics)
    pd.DataFrame([metrics_row]).to_csv(
        os.path.join(args.save_dir, "test_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    save_test_plot(
        os.path.join(args.save_dir, "test_prediction_next_step.png"),
        y_true_next,
        y_pred_next,
    )

    print(f"最终 best validation total loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(
        "测试集指标: "
        f"MAE={test_metrics['MAE']:.6f}, "
        f"RMSE={test_metrics['RMSE']:.6f}, "
        f"R2={test_metrics['R2']:.6f}"
    )
    print(f"输出目录: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
