"""监督式净负荷分解实验：测试阶段已知 net_load，模型只预测 gg，再由 net_load + gg 重构 gc。"""

import argparse
import copy
import os
import warnings
from typing import Dict, List, Optional, Tuple

from utils.runtime_env import ensure_conda_dll_paths

ensure_conda_dll_paths()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import CFG, PROJECT_ROOT
from models.cnn_lstm import Attention, SamePadMaxPool1d
from utils.data_utils import (
    add_timestamp_occurrence_key,
    build_features,
    ensure_dir,
    fit_and_transform_x,
    save_config,
    set_seed,
    split_df_by_time,
)
from utils.metrics import calc_metrics, plot_round_curve, plot_true_pred, print_metrics, save_metrics_csv


class PVDecompositionDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_gg: np.ndarray,
        y_gc: np.ndarray,
        y_net: np.ndarray,
        day_mask: np.ndarray,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_gg = torch.tensor(y_gg, dtype=torch.float32)
        self.y_gc = torch.tensor(y_gc, dtype=torch.float32)
        self.y_net = torch.tensor(y_net, dtype=torch.float32)
        self.day_mask = torch.tensor(day_mask, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y_gg[idx],
            self.y_gc[idx],
            self.y_net[idx],
            self.day_mask[idx],
        )


class PVOnlyCNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        cfg,
        pv_capacity_eff: float,
        use_pv_gate: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.horizon = horizon
        self.use_attention = cfg.use_attention
        self.use_pv_gate = use_pv_gate
        self.register_buffer("pv_capacity_eff", torch.tensor(float(pv_capacity_eff), dtype=torch.float32))

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

        self.lstm1 = nn.LSTM(
            input_size=cfg.conv2_channels,
            hidden_size=cfg.lstm_hidden1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=cfg.lstm_hidden1,
            hidden_size=cfg.lstm_hidden2,
            batch_first=True,
        )

        head_input_dim = cfg.attn_units if self.use_attention else cfg.lstm_hidden2
        if self.use_attention:
            self.attention = Attention(input_dim=cfg.lstm_hidden2, attn_units=cfg.attn_units)

        self.pv_fc1 = nn.Linear(head_input_dim, cfg.fc_hidden)
        self.pv_fc2 = nn.Linear(cfg.fc_hidden, horizon)

    def extract_features(self, x):
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

        return x

    def _prepare_capacity(self, pv_capacity_eff, raw_gg):
        if pv_capacity_eff is None:
            pv_eff = self.pv_capacity_eff
        else:
            pv_eff = torch.as_tensor(pv_capacity_eff, device=raw_gg.device, dtype=raw_gg.dtype)

        if pv_eff.dim() == 1:
            pv_eff = pv_eff.unsqueeze(1)
        return pv_eff.to(device=raw_gg.device, dtype=raw_gg.dtype)

    def forward(self, x, day_mask=None, pv_capacity_eff=None):
        feat = self.extract_features(x)
        pv_hidden = F.relu(self.pv_fc1(feat))
        raw_gg = self.pv_fc2(pv_hidden)

        if day_mask is None:
            day_mask = torch.ones_like(raw_gg)
        else:
            day_mask = day_mask.to(device=raw_gg.device, dtype=raw_gg.dtype)
        if not self.use_pv_gate:
            day_mask = torch.ones_like(raw_gg)

        pv_eff = self._prepare_capacity(pv_capacity_eff, raw_gg)
        pred_gg = day_mask * pv_eff * torch.sigmoid(raw_gg)

        return {
            "gg": pred_gg,
            "raw_gg": raw_gg,
            "feat": feat,
        }


def get_loss_fn(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss function: {loss_name}")


def make_loader(dataset: PVDecompositionDataset, cfg, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and str(cfg.train.device).startswith("cuda"),
    )


def normalize_device(device_name: str):
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is not available. Falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(device_name)


def read_client_dataframe(csv_path: str, cfg) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find client CSV: {csv_path}")

    dc = cfg.data
    ec = cfg.decomposed_net_load
    required_cols = [dc.datetime_col, ec.load_col, ec.pv_col]

    df = pd.read_csv(csv_path)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df[dc.datetime_col] = pd.to_datetime(df[dc.datetime_col])
    df[ec.load_col] = pd.to_numeric(df[ec.load_col], errors="coerce")
    df[ec.pv_col] = pd.to_numeric(df[ec.pv_col], errors="coerce")
    df = df.sort_values(dc.datetime_col).drop_duplicates(subset=[dc.datetime_col], keep="first")
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    df[ec.net_load_col] = df[ec.load_col].astype(float) - df[ec.pv_col].astype(float)
    return df


def configure_features_for_decomposition(cfg, df: pd.DataFrame) -> None:
    ec = cfg.decomposed_net_load
    fc = cfg.feature
    forbidden_cols = {ec.load_col, ec.pv_col}

    cfg.data.target_col = ec.input_history_col
    cfg.data.net_load_col = ec.net_load_col
    fc.use_target_history = True
    fc.use_slot_sin_cos = ec.use_slot_sin_cos
    fc.use_weekday_sin_cos = ec.use_weekday_sin_cos
    fc.use_month_sin_cos = ec.use_month_sin_cos
    fc.use_is_weekend = ec.use_is_weekend

    if ec.use_temp_c and fc.temp_source_mode.lower() == "auto":
        fc.use_temp_c = fc.temp_c_col in df.columns or fc.temp_k_col in df.columns
    elif ec.use_temp_c and fc.temp_source_mode.lower() == "c":
        fc.use_temp_c = fc.temp_c_col in df.columns
    elif ec.use_temp_c and fc.temp_source_mode.lower() == "k":
        fc.use_temp_c = fc.temp_k_col in df.columns
    else:
        fc.use_temp_c = False
    if ec.use_temp_c and not fc.use_temp_c:
        warnings.warn("Temperature feature is unavailable and will be disabled.", RuntimeWarning)

    fc.use_wind = ec.use_wind and fc.wind_col in df.columns
    if ec.use_wind and not fc.use_wind:
        warnings.warn("Wind feature is unavailable and will be disabled.", RuntimeWarning)

    fc.ghi_col = ec.ghi_col
    fc.use_ghi = ec.use_ghi_feature and ec.ghi_col in df.columns
    if ec.use_ghi_feature and not fc.use_ghi:
        warnings.warn(
            f"{ec.ghi_col} is missing. GHI input will be disabled and day_mask will use hour fallback.",
            RuntimeWarning,
        )

    fc.use_rrp = ec.use_rrp and getattr(fc, "rrp_col", "rrp_aud_per_mwh") in df.columns

    raw_feature_cols = []
    candidate_raw_cols = list(getattr(ec, "raw_feature_cols", [])) + list(getattr(fc, "raw_feature_cols", []))
    for col in candidate_raw_cols:
        if col in forbidden_cols:
            warnings.warn(f"Raw feature '{col}' is forbidden for this experiment and was removed.", RuntimeWarning)
            continue
        if col == ec.net_load_col:
            continue
        if col in df.columns and col not in raw_feature_cols:
            raw_feature_cols.append(col)
    fc.raw_feature_cols = raw_feature_cols


def validate_feature_columns(feature_cols: List[str], cfg) -> None:
    ec = cfg.decomposed_net_load
    forbidden_cols = {ec.load_col, ec.pv_col}
    leaked = [col for col in feature_cols if col in forbidden_cols]
    if leaked:
        raise ValueError(f"Forbidden gc/gg history columns leaked into model inputs: {leaked}")
    if ec.input_history_col != ec.net_load_col:
        raise ValueError("input_history_col must be net_load for this experiment.")
    if cfg.data.target_col != ec.net_load_col:
        raise ValueError("cfg.data.target_col must be net_load while building input history.")


def build_day_mask_and_ghi(df: pd.DataFrame, cfg) -> Tuple[np.ndarray, np.ndarray]:
    ec = cfg.decomposed_net_load
    dt = pd.to_datetime(df[cfg.data.datetime_col])
    hour_fallback = ((dt.dt.hour >= 6) & (dt.dt.hour <= 18)).astype(float).values

    if ec.ghi_col not in df.columns:
        mask = hour_fallback
        ghi_values = np.full(len(df), np.nan, dtype=np.float32)
        return mask.astype(np.float32), ghi_values

    ghi_values = pd.to_numeric(df[ec.ghi_col], errors="coerce").values.astype(np.float32)
    ghi_mask = (ghi_values > ec.ghi_gate_threshold).astype(np.float32)
    mask = np.where(np.isfinite(ghi_values), ghi_mask, hour_fallback).astype(np.float32)
    return mask, ghi_values


def create_decomposition_sequences(
    feature_array: np.ndarray,
    gg_array: np.ndarray,
    gc_array: np.ndarray,
    net_array: np.ndarray,
    day_mask_array: np.ndarray,
    ghi_array: np.ndarray,
    timestamp_array,
    seq_len: int,
    horizon: int,
):
    xs, y_gg, y_gc, y_net, day_masks, ghi_values, timestamps = [], [], [], [], [], [], []
    total_len = len(feature_array)

    for end_idx in range(seq_len, total_len - horizon + 1):
        start_idx = end_idx - seq_len
        xs.append(feature_array[start_idx:end_idx, :])
        y_gg.append(gg_array[end_idx:end_idx + horizon])
        y_gc.append(gc_array[end_idx:end_idx + horizon])
        y_net.append(net_array[end_idx:end_idx + horizon])
        day_masks.append(day_mask_array[end_idx:end_idx + horizon])
        ghi_values.append(ghi_array[end_idx:end_idx + horizon])
        timestamps.append(timestamp_array[end_idx:end_idx + horizon])

    if len(xs) == 0:
        raise ValueError(
            f"Not enough rows to create sequences: total_len={total_len}, seq_len={seq_len}, horizon={horizon}"
        )

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(y_gg, dtype=np.float32).reshape(-1, horizon),
        np.asarray(y_gc, dtype=np.float32).reshape(-1, horizon),
        np.asarray(y_net, dtype=np.float32).reshape(-1, horizon),
        np.asarray(day_masks, dtype=np.float32).reshape(-1, horizon),
        np.asarray(ghi_values, dtype=np.float32).reshape(-1, horizon),
        np.asarray(timestamps),
    )


def build_sequences_for_split(df_scaled: pd.DataFrame, df_raw: pd.DataFrame, feature_cols: List[str], cfg):
    ec = cfg.decomposed_net_load
    day_mask, ghi_values = build_day_mask_and_ghi(df_raw, cfg)
    x = df_scaled[feature_cols].values.astype(np.float32)
    gg = pd.to_numeric(df_raw[ec.pv_col], errors="coerce").values.astype(np.float32)
    gc = pd.to_numeric(df_raw[ec.load_col], errors="coerce").values.astype(np.float32)
    net = pd.to_numeric(df_raw[ec.net_load_col], errors="coerce").values.astype(np.float32)
    timestamps = pd.to_datetime(df_raw[cfg.data.datetime_col]).values

    return create_decomposition_sequences(
        feature_array=x,
        gg_array=gg,
        gc_array=gc,
        net_array=net,
        day_mask_array=day_mask,
        ghi_array=ghi_values,
        timestamp_array=timestamps,
        seq_len=cfg.data.seq_len,
        horizon=cfg.data.horizon,
    )


def compute_effective_pv_capacity(train_df: pd.DataFrame, cfg) -> Dict[str, float]:
    ec = cfg.decomposed_net_load
    gg_values = pd.to_numeric(train_df[ec.pv_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    gg_values = gg_values[gg_values >= 0]

    if len(gg_values) == 0:
        gg_q99 = 0.0
        gg_q995 = 0.0
        gg_max = 0.0
        q_capacity = 1.0
    else:
        gg_q99 = float(gg_values.quantile(0.99))
        gg_q995 = float(gg_values.quantile(ec.pv_capacity_quantile))
        gg_max = float(gg_values.max())
        q_capacity = gg_q995 * ec.pv_capacity_alpha

    raw_capacity = float("inf")
    if ec.capacity_col in train_df.columns:
        cap_values = pd.to_numeric(train_df[ec.capacity_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        cap_values = cap_values.dropna()
        cap_values = cap_values[cap_values > ec.min_pv_capacity_eps]
        if len(cap_values) > 0:
            raw_capacity = float(cap_values.median())

    if not np.isfinite(q_capacity) or q_capacity <= ec.min_pv_capacity_eps:
        q_capacity = gg_max * ec.pv_capacity_alpha if gg_max > ec.min_pv_capacity_eps else 1.0

    if ec.use_effective_pv_capacity:
        pv_capacity_eff = min(raw_capacity, q_capacity) if np.isfinite(raw_capacity) else q_capacity
    else:
        pv_capacity_eff = raw_capacity if np.isfinite(raw_capacity) else q_capacity

    if not np.isfinite(pv_capacity_eff) or pv_capacity_eff <= ec.min_pv_capacity_eps:
        pv_capacity_eff = gg_max if gg_max > ec.min_pv_capacity_eps else 1.0

    return {
        "pv_capacity_eff": float(pv_capacity_eff),
        "raw_pv_capacity": float(raw_capacity),
        "gg_q99": float(gg_q99),
        "gg_q995": float(gg_q995),
        "gg_max": float(gg_max),
        "q_capacity": float(q_capacity),
    }


def save_pv_capacity_summary(capacity_info: Dict[str, float], client_id: int, client_name: str, save_dir: str):
    row = {"client_id": client_id, "client_name": client_name, **capacity_info}
    pd.DataFrame([row]).to_csv(
        os.path.join(save_dir, "pv_capacity_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def run_pv_only_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    cfg,
    train: bool,
    collect_predictions: bool = True,
):
    if train:
        model.train()
    else:
        model.eval()

    ec = cfg.decomposed_net_load
    lambda_gg = float(getattr(ec, "lambda_gg", 1.0))
    lambda_gc = float(getattr(ec, "lambda_gc_reconstruction", 0.0))
    loss_sum = 0.0
    gg_loss_sum = 0.0
    gc_loss_sum = 0.0
    count = 0
    preds = {"gg": [], "gc": []}
    trues = {"gg": [], "gc": [], "net": []}

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_x, y_gg, y_gc, y_net, day_mask in loader:
            batch_x = batch_x.to(device)
            y_gg = y_gg.to(device)
            y_gc = y_gc.to(device)
            y_net = y_net.to(device)
            day_mask = day_mask.to(device)

            if train:
                optimizer.zero_grad()

            out = model(batch_x, day_mask=day_mask)
            pred_gg = out["gg"]
            pred_gc = y_net + pred_gg
            loss_gg = criterion(pred_gg, y_gg)
            loss_gc = criterion(pred_gc, y_gc)
            loss = lambda_gg * loss_gg + lambda_gc * loss_gc

            if train:
                loss.backward()
                optimizer.step()

            batch_size = batch_x.size(0)
            loss_sum += loss.item() * batch_size
            gg_loss_sum += loss_gg.item() * batch_size
            gc_loss_sum += loss_gc.item() * batch_size
            count += batch_size

            if collect_predictions:
                preds["gg"].append(pred_gg.detach().cpu().numpy())
                preds["gc"].append(pred_gc.detach().cpu().numpy())
                trues["gg"].append(y_gg.detach().cpu().numpy())
                trues["gc"].append(y_gc.detach().cpu().numpy())
                trues["net"].append(y_net.detach().cpu().numpy())

    stats = {
        "loss": loss_sum / max(count, 1),
        "gg_loss": gg_loss_sum / max(count, 1),
        "gc_reconstruction_loss": gc_loss_sum / max(count, 1),
    }

    if not collect_predictions:
        return stats, None, None

    preds = {key: np.concatenate(value, axis=0) for key, value in preds.items()}
    trues = {key: np.concatenate(value, axis=0) for key, value in trues.items()}
    return stats, preds, trues


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return calc_metrics(y_true.reshape(-1), y_pred.reshape(-1))


def choose_early_stop_value(early_stop_metric: str, gg_metrics: Dict[str, float], gc_metrics: Dict[str, float]) -> float:
    if early_stop_metric == "gg_RMSE":
        return gg_metrics["RMSE"]
    if early_stop_metric == "gc_RMSE":
        return gc_metrics["RMSE"]
    raise ValueError("early_stop_metric must be 'gg_RMSE' or 'gc_RMSE'.")


def build_prediction_dataframe(
    timestamps,
    y_net_true,
    y_gg_true,
    y_gg_pred,
    y_gc_true,
    y_gc_pred,
    day_mask,
    ghi_values,
    pv_capacity_eff: float,
    horizon: int,
) -> pd.DataFrame:
    data = {
        "timestamp": pd.to_datetime(np.asarray(timestamps).reshape(-1)),
        "y_net_true": y_net_true.reshape(-1),
        "y_gg_true": y_gg_true.reshape(-1),
        "y_gg_pred": y_gg_pred.reshape(-1),
        "y_gc_true": y_gc_true.reshape(-1),
        "y_gc_pred": y_gc_pred.reshape(-1),
        "day_mask": day_mask.reshape(-1),
        "ghi_wm2": ghi_values.reshape(-1),
        "pv_capacity_eff": float(pv_capacity_eff),
    }
    if horizon > 1:
        n_samples = y_net_true.shape[0]
        data["step"] = np.tile(np.arange(1, horizon + 1), n_samples)
    return pd.DataFrame(data)


def save_test_outputs(
    pred_df: pd.DataFrame,
    metrics_gg: Dict[str, float],
    metrics_gc: Dict[str, float],
    save_dir: str,
    cfg,
):
    ec = cfg.decomposed_net_load
    if ec.save_predictions:
        pred_df.to_csv(
            os.path.join(save_dir, "test_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    save_metrics_csv(metrics_gg, os.path.join(save_dir, "test_metrics_gg.csv"))
    save_metrics_csv(metrics_gc, os.path.join(save_dir, "test_metrics_gc.csv"))

    if ec.save_plots:
        plot_true_pred(
            pred_df["y_gg_true"].values,
            pred_df["y_gg_pred"].values,
            save_path=os.path.join(save_dir, "gg_test_prediction.png"),
            title="PV-Only Decomposition GG Test Prediction",
            show_n=300,
        )
        plot_true_pred(
            pred_df["y_gc_true"].values,
            pred_df["y_gc_pred"].values,
            save_path=os.path.join(save_dir, "gc_test_prediction.png"),
            title="PV-Only Decomposition GC Reconstruction",
            show_n=300,
        )


def train_one_client_decomposed(client_id: int, csv_path: str, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    ec = cfg.decomposed_net_load
    client_name = f"client_{client_id}"
    client_dir = os.path.join(ec.save_dir, client_name)
    ensure_dir(client_dir)

    df = read_client_dataframe(csv_path, cfg)
    configure_features_for_decomposition(cfg, df)
    df, feature_cols = build_features(df, cfg)
    validate_feature_columns(feature_cols, cfg)

    train_df, val_df, test_df = split_df_by_time(df, cfg)
    capacity_info = compute_effective_pv_capacity(train_df, cfg)
    save_pv_capacity_summary(capacity_info, client_id, client_name, client_dir)

    train_scaled_df, val_scaled_df, test_scaled_df, _, scale_cols, keep_cols = fit_and_transform_x(
        train_df, val_df, test_df, feature_cols, cfg
    )

    train_seq = build_sequences_for_split(train_scaled_df, train_df, feature_cols, cfg)
    val_seq = build_sequences_for_split(val_scaled_df, val_df, feature_cols, cfg)
    test_seq = build_sequences_for_split(test_scaled_df, test_df, feature_cols, cfg)

    x_train, y_gg_train, y_gc_train, y_net_train, day_train, _, _ = train_seq
    x_val, y_gg_val, y_gc_val, y_net_val, day_val, _, _ = val_seq
    x_test, y_gg_test, y_gc_test, y_net_test, day_test, ghi_test, ts_test = test_seq

    train_loader = make_loader(
        PVDecompositionDataset(x_train, y_gg_train, y_gc_train, y_net_train, day_train),
        cfg,
        shuffle=True,
    )
    val_loader = make_loader(
        PVDecompositionDataset(x_val, y_gg_val, y_gc_val, y_net_val, day_val),
        cfg,
        shuffle=False,
    )
    test_loader = make_loader(
        PVDecompositionDataset(x_test, y_gg_test, y_gc_test, y_net_test, day_test),
        cfg,
        shuffle=False,
    )

    save_config(cfg, client_dir)

    device = normalize_device(cfg.train.device)
    model = PVOnlyCNNLSTMModel(
        input_dim=len(feature_cols),
        horizon=cfg.data.horizon,
        cfg=cfg.model,
        pv_capacity_eff=capacity_info["pv_capacity_eff"],
        use_pv_gate=ec.use_pv_gate,
    ).to(device)
    criterion = get_loss_fn(cfg.train.loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    early_stop_metric = getattr(ec, "early_stop_metric", "gg_RMSE")
    best_early_stop_value = float("inf")
    best_val_gg_rmse = float("inf")
    best_model_path = os.path.join(client_dir, ec.best_model_name)
    final_model_path = os.path.join(client_dir, ec.final_model_name)
    no_improve_epochs = 0
    rows = []

    print("=" * 100)
    print(f"PV-only decomposition training | {client_name}")
    print(f"CSV: {csv_path}")
    print(f"Input features ({len(feature_cols)}): {feature_cols}")
    print(f"Scaled columns: {scale_cols}; kept columns: {keep_cols}")
    print(f"PV effective capacity: {capacity_info['pv_capacity_eff']:.6f}")
    print(f"Train/Val/Test sequences: {len(x_train)}/{len(x_val)}/{len(x_test)}")
    print(f"Early-stop metric: {early_stop_metric}")
    print(f"Device: {device}")
    print("=" * 100)

    for epoch in range(1, cfg.train.epochs + 1):
        train_stats, _, _ = run_pv_only_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            train=True,
            collect_predictions=False,
        )
        val_stats, val_preds, val_trues = run_pv_only_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            cfg=cfg,
            train=False,
            collect_predictions=True,
        )

        val_gg_metrics = metrics_from_arrays(val_trues["gg"], val_preds["gg"])
        val_gc_metrics = metrics_from_arrays(val_trues["gc"], val_preds["gc"])
        best_val_gg_rmse = min(best_val_gg_rmse, val_gg_metrics["RMSE"])
        early_stop_value = choose_early_stop_value(early_stop_metric, val_gg_metrics, val_gc_metrics)

        improved = early_stop_value < best_early_stop_value
        if improved:
            best_early_stop_value = early_stop_value
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epochs += 1

        rows.append({
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_gg_loss": float(train_stats["gg_loss"]),
            "train_gc_reconstruction_loss": float(train_stats["gc_reconstruction_loss"]),
            "val_loss": float(val_stats["loss"]),
            "val_gg_loss": float(val_stats["gg_loss"]),
            "val_gc_reconstruction_loss": float(val_stats["gc_reconstruction_loss"]),
            "val_gg_MAE": val_gg_metrics["MAE"],
            "val_gg_MSE": val_gg_metrics["MSE"],
            "val_gg_RMSE": val_gg_metrics["RMSE"],
            "val_gg_MAPE_percent": val_gg_metrics["MAPE_percent"],
            "val_gg_R2": val_gg_metrics["R2"],
            "val_gc_MAE": val_gc_metrics["MAE"],
            "val_gc_MSE": val_gc_metrics["MSE"],
            "val_gc_RMSE": val_gc_metrics["RMSE"],
            "val_gc_MAPE_percent": val_gc_metrics["MAPE_percent"],
            "val_gc_R2": val_gc_metrics["R2"],
            "best_val_gg_RMSE": best_val_gg_rmse,
        })

        print(
            f"Epoch [{epoch:03d}/{cfg.train.epochs}] | "
            f"TrainLoss: {train_stats['loss']:.6f} | "
            f"ValGGRMSE: {val_gg_metrics['RMSE']:.6f} | "
            f"ValGCRMSE: {val_gc_metrics['RMSE']:.6f} | "
            f"BestMetric: {best_early_stop_value:.6f}"
        )

        patience = cfg.train.early_stop_patience
        if patience and no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)
    torch.save(model.state_dict(), final_model_path)

    log_df = pd.DataFrame(rows)
    log_df.to_csv(os.path.join(client_dir, "training_log.csv"), index=False, encoding="utf-8-sig")
    if ec.save_plots and len(log_df) > 0:
        plot_round_curve(
            log_df["val_gg_RMSE"].values,
            title=f"{client_name} Validation GG RMSE",
            xlabel="Epoch",
            ylabel="RMSE",
            save_path=os.path.join(client_dir, "gg_val_rmse_curve.png"),
        )
        plot_round_curve(
            log_df["val_gc_RMSE"].values,
            title=f"{client_name} Validation GC RMSE",
            xlabel="Epoch",
            ylabel="RMSE",
            save_path=os.path.join(client_dir, "gc_val_rmse_curve.png"),
        )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_stats, test_preds, test_trues = run_pv_only_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        cfg=cfg,
        train=False,
        collect_predictions=True,
    )

    metrics_gg = metrics_from_arrays(test_trues["gg"], test_preds["gg"])
    metrics_gc = metrics_from_arrays(test_trues["gc"], test_preds["gc"])
    print_metrics(metrics_gg, title=f"{client_name} GG Test Metrics")
    print_metrics(metrics_gc, title=f"{client_name} GC Reconstruction Test Metrics")

    pred_df = build_prediction_dataframe(
        timestamps=ts_test,
        y_net_true=test_trues["net"],
        y_gg_true=test_trues["gg"],
        y_gg_pred=test_preds["gg"],
        y_gc_true=test_trues["gc"],
        y_gc_pred=test_preds["gc"],
        day_mask=day_test,
        ghi_values=ghi_test,
        pv_capacity_eff=capacity_info["pv_capacity_eff"],
        horizon=cfg.data.horizon,
    )
    save_test_outputs(pred_df, metrics_gg, metrics_gc, client_dir, cfg)

    summary_row = {
        "client_id": client_id,
        "client_name": client_name,
        "gg_MAE": metrics_gg["MAE"],
        "gg_MSE": metrics_gg["MSE"],
        "gg_RMSE": metrics_gg["RMSE"],
        "gg_MAPE_percent": metrics_gg["MAPE_percent"],
        "gg_R2": metrics_gg["R2"],
        "gc_MAE": metrics_gc["MAE"],
        "gc_MSE": metrics_gc["MSE"],
        "gc_RMSE": metrics_gc["RMSE"],
        "gc_MAPE_percent": metrics_gc["MAPE_percent"],
        "gc_R2": metrics_gc["R2"],
        "pv_capacity_eff": capacity_info["pv_capacity_eff"],
        "raw_pv_capacity": capacity_info["raw_pv_capacity"],
        "gg_q99": capacity_info["gg_q99"],
        "gg_q995": capacity_info["gg_q995"],
        "gg_max": capacity_info["gg_max"],
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "test_loss": float(test_stats["loss"]),
    }

    return {
        "client_id": client_id,
        "client_name": client_name,
        "client_dir": client_dir,
        "pred_df": pred_df,
        "summary_row": summary_row,
        "feature_cols": feature_cols,
    }


def aggregate_regional_component_predictions(results: List[Dict], component: str):
    if component not in {"gg", "gc"}:
        raise ValueError("component must be 'gg' or 'gc'.")

    merged = None
    for result in results:
        cid = result["client_id"]
        true_col = f"y_{component}_true"
        pred_col = f"y_{component}_pred"
        pred_df = result["pred_df"][["timestamp", true_col, pred_col]].copy()
        pred_df = add_timestamp_occurrence_key(pred_df, "timestamp")
        pred_df = pred_df.rename(columns={
            true_col: f"{true_col}_client_{cid}",
            pred_col: f"{pred_col}_client_{cid}",
        })
        if merged is None:
            merged = pred_df
        else:
            merged = pd.merge(merged, pred_df, on=["timestamp", "_timestamp_occurrence"], how="inner")

    if merged is None or len(merged) == 0:
        raise ValueError("Regional aggregation produced no aligned samples.")

    true_prefix = f"y_{component}_true_client_"
    pred_prefix = f"y_{component}_pred_client_"
    true_cols = [col for col in merged.columns if col.startswith(true_prefix)]
    pred_cols = [col for col in merged.columns if col.startswith(pred_prefix)]
    regional_df = pd.DataFrame({"timestamp": merged["timestamp"].copy()})
    regional_df[f"regional_{component}_true"] = merged[true_cols].sum(axis=1)
    regional_df[f"regional_{component}_pred"] = merged[pred_cols].sum(axis=1)
    regional_df = regional_df.sort_values("timestamp").reset_index(drop=True)
    metrics = calc_metrics(
        regional_df[f"regional_{component}_true"].values,
        regional_df[f"regional_{component}_pred"].values,
    )
    return regional_df, metrics


def save_regional_component_outputs(regional_df: pd.DataFrame, metrics: Dict[str, float], component: str, save_dir: str, cfg):
    pred_path = os.path.join(save_dir, f"regional_{component}_test_predictions.csv")
    metrics_path = os.path.join(save_dir, f"regional_{component}_test_metrics.csv")
    plot_path = os.path.join(save_dir, f"regional_{component}_test_prediction.png")
    regional_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    save_metrics_csv(metrics, metrics_path)
    if cfg.decomposed_net_load.save_plots:
        plot_true_pred(
            regional_df[f"regional_{component}_true"].values,
            regional_df[f"regional_{component}_pred"].values,
            save_path=plot_path,
            title=f"Regional {component.upper()} Test Prediction",
            show_n=300,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PV-only supervised net-load decomposition."
    )
    parser.add_argument("--output-root", default=None, help="Output root directory.")
    parser.add_argument("--client-id", type=int, default=None, help="Train only one client id.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override Adam learning rate.")
    parser.add_argument("--ghi-threshold", type=float, default=None, help="Override GHI day-mask threshold.")
    parser.add_argument("--capacity-quantile", type=float, default=None, help="Override PV capacity quantile.")
    parser.add_argument("--capacity-alpha", type=float, default=None, help="Override PV capacity multiplier.")
    parser.add_argument("--lambda-gg", type=float, default=None, help="Override gg loss weight.")
    parser.add_argument(
        "--lambda-gc-reconstruction",
        type=float,
        default=None,
        help="Override optional gc reconstruction loss weight.",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["gg_RMSE", "gc_RMSE"],
        default=None,
        help="Validation metric used for early stopping.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg, args) -> None:
    if args.output_root is not None:
        cfg.decomposed_net_load.save_dir = os.path.abspath(args.output_root)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.patience is not None:
        cfg.train.early_stop_patience = args.patience
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.ghi_threshold is not None:
        cfg.decomposed_net_load.ghi_gate_threshold = args.ghi_threshold
    if args.capacity_quantile is not None:
        cfg.decomposed_net_load.pv_capacity_quantile = args.capacity_quantile
    if args.capacity_alpha is not None:
        cfg.decomposed_net_load.pv_capacity_alpha = args.capacity_alpha
    if args.lambda_gg is not None:
        cfg.decomposed_net_load.lambda_gg = args.lambda_gg
    if args.lambda_gc_reconstruction is not None:
        cfg.decomposed_net_load.lambda_gc_reconstruction = args.lambda_gc_reconstruction
    if args.early_stop_metric is not None:
        cfg.decomposed_net_load.early_stop_metric = args.early_stop_metric


def select_client_jobs(cfg, client_id: Optional[int]) -> List[Tuple[int, str]]:
    client_files = list(getattr(cfg.decomposed_net_load, "client_files", None) or cfg.data.client_files)
    if len(client_files) == 0:
        raise ValueError("No client files configured for PV-only decomposition training.")

    if client_id is not None:
        if client_id < 1 or client_id > len(client_files):
            raise ValueError(f"--client-id must be between 1 and {len(client_files)}")
        return [(client_id, client_files[client_id - 1])]

    if cfg.decomposed_net_load.run_all_clients:
        return [(idx, path) for idx, path in enumerate(client_files, start=1)]

    return [(1, client_files[0])]


def main():
    args = parse_args()
    cfg = copy.deepcopy(CFG)
    apply_cli_overrides(cfg, args)
    set_seed(cfg.train.random_seed)
    ensure_dir(cfg.decomposed_net_load.save_dir)
    save_config(cfg, cfg.decomposed_net_load.save_dir)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root: {cfg.decomposed_net_load.save_dir}")

    jobs = select_client_jobs(cfg, args.client_id)
    results = []
    for client_id, csv_path in jobs:
        result = train_one_client_decomposed(client_id, csv_path, cfg)
        results.append(result)

        summary_df = pd.DataFrame([item["summary_row"] for item in results]).sort_values("client_id")
        summary_df.to_csv(
            os.path.join(cfg.decomposed_net_load.save_dir, "all_clients_test_metrics_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    summary_df = pd.DataFrame([item["summary_row"] for item in results]).sort_values("client_id")
    summary_path = os.path.join(cfg.decomposed_net_load.save_dir, "all_clients_test_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    run_regional = args.client_id is None and cfg.decomposed_net_load.run_all_clients and len(results) > 1
    if run_regional:
        for component in ["gg", "gc"]:
            regional_df, regional_metrics = aggregate_regional_component_predictions(results, component)
            save_regional_component_outputs(regional_df, regional_metrics, component, cfg.decomposed_net_load.save_dir, cfg)
            print_metrics(regional_metrics, title=f"Regional {component.upper()} Test Metrics")

    print(f"\nSummary saved to: {summary_path}")
    print(f"Results directory: {cfg.decomposed_net_load.save_dir}")


if __name__ == "__main__":
    main()
