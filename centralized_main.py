import copy
import os

from utils.runtime_env import ensure_conda_dll_paths

ensure_conda_dll_paths()

import pandas as pd
import torch
from torch.utils.data import DataLoader

from client import get_loss_fn, get_optimizer, run_one_epoch
from config import CFG
from models.cnn_lstm import CNNLSTMModel
from utils.data_utils import (
    SeqDataset,
    add_timestamp_occurrence_key,
    build_centralized_aggregate_dataframe,
    build_features,
    create_sequences,
    ensure_dir,
    fit_and_transform_x,
    fit_and_transform_y,
    inverse_transform_array,
    save_config,
    set_seed,
    split_df_by_time,
)
from utils.metrics import calc_metrics, plot_round_curve, plot_true_pred, print_metrics, save_metrics_csv


def make_loader(x_seq, y_seq, cfg, shuffle=False):
    return DataLoader(
        SeqDataset(x_seq, y_seq),
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and str(cfg.train.device).startswith("cuda"),
    )


def build_centralized_net_load_dataframe(cfg):
    dt_col = cfg.data.datetime_col

    gc_cfg = copy.deepcopy(cfg)
    gc_cfg.data.target_col = "gc"
    gc_df = build_centralized_aggregate_dataframe(gc_cfg.data.client_files, gc_cfg)

    gg_cfg = copy.deepcopy(cfg)
    gg_cfg.data.target_col = "gg"
    gg_df = build_centralized_aggregate_dataframe(gg_cfg.data.client_files, gg_cfg)

    gc_df = add_timestamp_occurrence_key(gc_df, dt_col)
    gg_df = add_timestamp_occurrence_key(gg_df[[dt_col, "gg"]], dt_col)

    merged = pd.merge(
        gc_df,
        gg_df,
        on=[dt_col, "_timestamp_occurrence"],
        how="inner",
    )
    merged[cfg.data.net_load_col] = merged["gc"] - merged["gg"]
    merged = merged.drop(columns=["_timestamp_occurrence"])
    return merged.sort_values(dt_col).reset_index(drop=True)


def main():
    cfg = copy.deepcopy(CFG)

    # This script now runs direct net-load forecasting and removes temperature features.
    cfg.data.target_col = cfg.data.net_load_col
    cfg.feature.use_temp_c = False
    cfg.feature.use_apparent_temp = False
    cfg.centralized.save_dir = os.path.join(cfg.centralized.save_dir, "direct_net_load_no_temp")

    set_seed(cfg.train.random_seed)
    ensure_dir(cfg.centralized.save_dir)
    save_config(cfg, cfg.centralized.save_dir)

    df = build_centralized_net_load_dataframe(cfg)
    df, feature_cols = build_features(df, cfg)

    train_df, val_df, test_df = split_df_by_time(df, cfg)
    train_scaled_df, val_scaled_df, test_scaled_df, _, _, _ = fit_and_transform_x(
        train_df, val_df, test_df, feature_cols, cfg
    )
    y_train_scaled, y_val_scaled, y_test_scaled, y_scaler = fit_and_transform_y(train_df, val_df, test_df, cfg)

    x_train = train_scaled_df[feature_cols].values
    x_val = val_scaled_df[feature_cols].values
    x_test = test_scaled_df[feature_cols].values

    ts_train = pd.to_datetime(train_df[cfg.data.datetime_col]).values
    ts_val = pd.to_datetime(val_df[cfg.data.datetime_col]).values
    ts_test = pd.to_datetime(test_df[cfg.data.datetime_col]).values

    x_train_seq, y_train_seq, _ = create_sequences(
        x_train, y_train_scaled, ts_train, cfg.data.seq_len, cfg.data.horizon
    )
    x_val_seq, y_val_seq, _ = create_sequences(
        x_val, y_val_scaled, ts_val, cfg.data.seq_len, cfg.data.horizon
    )
    x_test_seq, y_test_seq, test_ts_seq = create_sequences(
        x_test, y_test_scaled, ts_test, cfg.data.seq_len, cfg.data.horizon
    )

    train_loader = make_loader(x_train_seq, y_train_seq, cfg, shuffle=True)
    val_loader = make_loader(x_val_seq, y_val_seq, cfg, shuffle=False)
    test_loader = make_loader(x_test_seq, y_test_seq, cfg, shuffle=False)

    device = torch.device(cfg.train.device)
    model = CNNLSTMModel(
        input_dim=len(feature_cols),
        output_dim=cfg.data.horizon,
        cfg=cfg.model,
    ).to(device)

    criterion = get_loss_fn(cfg.train.loss_name)
    optimizer = get_optimizer(cfg.train.optimizer_name, model, cfg.train.lr)

    best_val_rmse = float("inf")
    best_model_path = os.path.join(cfg.centralized.save_dir, cfg.centralized.best_model_name)

    train_loss_curve = []
    val_rmse_curve = []
    val_r2_curve = []

    print("=" * 100)
    print("Centralized direct net-load forecasting")
    print(f"Target column: {cfg.data.target_col}")
    print(f"Input features: {feature_cols}")
    print(f"Device: {cfg.train.device}")
    print("=" * 100)

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, _, _ = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        _, val_pred_scaled, val_true_scaled = run_one_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        val_pred_real = inverse_transform_array(y_scaler, val_pred_scaled)
        val_true_real = inverse_transform_array(y_scaler, val_true_scaled)
        val_metrics = calc_metrics(val_true_real.reshape(-1), val_pred_real.reshape(-1))

        train_loss_curve.append(train_loss)
        val_rmse_curve.append(val_metrics["RMSE"])
        val_r2_curve.append(val_metrics["R2"])

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch [{epoch:03d}/{cfg.train.epochs}] | "
            f"TrainLoss: {train_loss:.6f} | "
            f"ValRMSE: {val_metrics['RMSE']:.6f} | "
            f"ValR2: {val_metrics['R2']:.6f}"
        )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_pred_scaled, test_true_scaled = run_one_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )

    pred_real = inverse_transform_array(y_scaler, test_pred_scaled)
    true_real = inverse_transform_array(y_scaler, test_true_scaled)
    test_metrics = calc_metrics(true_real.reshape(-1), pred_real.reshape(-1))
    print_metrics(test_metrics, title="Centralized Total Net Load Test Metrics")

    pred_df = pd.DataFrame({"timestamp": pd.to_datetime(test_ts_seq[:, 0])})
    for step in range(cfg.data.horizon):
        pred_df[f"y_true_step_{step + 1}"] = true_real[:, step]
        pred_df[f"y_pred_step_{step + 1}"] = pred_real[:, step]

    pred_df.to_csv(
        os.path.join(cfg.centralized.save_dir, "centralized_test_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_metrics_csv(
        test_metrics,
        os.path.join(cfg.centralized.save_dir, "centralized_test_metrics.csv"),
    )

    plot_round_curve(
        train_loss_curve,
        title="Centralized Train Loss",
        xlabel="Epoch",
        ylabel="Loss",
        save_path=os.path.join(cfg.centralized.save_dir, "train_loss_curve.png"),
    )
    plot_round_curve(
        val_rmse_curve,
        title="Centralized Val RMSE",
        xlabel="Epoch",
        ylabel="RMSE",
        save_path=os.path.join(cfg.centralized.save_dir, "val_rmse_curve.png"),
    )
    plot_round_curve(
        val_r2_curve,
        title="Centralized Val R2",
        xlabel="Epoch",
        ylabel="R2",
        save_path=os.path.join(cfg.centralized.save_dir, "val_r2_curve.png"),
    )
    plot_true_pred(
        pred_df["y_true_step_1"].values,
        pred_df["y_pred_step_1"].values,
        save_path=os.path.join(cfg.centralized.save_dir, "centralized_test_prediction.png"),
        title="Centralized Total Net Load Prediction",
        show_n=300,
    )

    print(f"\nBest model saved to: {best_model_path}")
    print(f"Results directory: {cfg.centralized.save_dir}")


if __name__ == "__main__":
    main()
