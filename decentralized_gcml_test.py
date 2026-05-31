import copy
import os

import pandas as pd
import torch

from config import CFG
from decentralized_gcml_main import (
    build_direct_net_load_cfg,
    build_indirect_gc_cfg,
    build_indirect_gg_cfg,
    calc_prediction_metrics,
    clean_test_prediction_df,
    combine_prediction_frames,
    save_net_load_outputs,
    set_default_nine_client_files,
)
from federated_main import build_clients
from utils.data_utils import ensure_dir, set_seed
from utils.metrics import plot_true_pred, print_metrics, save_metrics_csv


def get_receiver_client(cfg):
    clients, _ = build_clients(cfg)
    receiver_id = cfg.decentralized_gcml.receiver_client_id
    for client in clients:
        if client.client_id == receiver_id:
            return client
    raise ValueError(f"receiver_client_id={receiver_id} is not in built clients.")


def load_best_receiver_state(cfg):
    model_path = os.path.join(
        cfg.decentralized_gcml.save_dir,
        cfg.decentralized_gcml.best_receiver_model_name,
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best receiver model not found: {model_path}")
    return torch.load(model_path, map_location=cfg.train.device), model_path


def evaluate_best_receiver_model(cfg, save_dir: str, title_prefix: str):
    ensure_dir(save_dir)
    receiver_client = get_receiver_client(cfg)
    state, model_path = load_best_receiver_state(cfg)

    pred_df, _, loss = receiver_client.evaluate_split(state, split_name="test")
    pred_df = clean_test_prediction_df(pred_df, cfg.data.horizon)
    metrics = calc_prediction_metrics(pred_df, cfg.data.horizon)

    receiver_id = receiver_client.client_id
    pred_df.to_csv(
        os.path.join(save_dir, f"client_{receiver_id}_test_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_metrics_csv(metrics, os.path.join(save_dir, f"client_{receiver_id}_test_metrics.csv"))
    plot_true_pred(
        pred_df["y_true_step_1"].values,
        pred_df["y_pred_step_1"].values,
        save_path=os.path.join(save_dir, f"client_{receiver_id}_test_prediction.png"),
        title=f"{title_prefix} Client {receiver_id} Test Prediction",
        show_n=300,
    )

    summary_df = pd.DataFrame([{
        "client_id": receiver_id,
        "client_name": receiver_client.client_name,
        "loss": float(loss),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE_percent": metrics["MAPE_percent"],
        "R2": metrics["R2"],
        "model_path": model_path,
    }])
    summary_df.to_csv(
        os.path.join(save_dir, "all_clients_test_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print_metrics(metrics, title=f"{title_prefix} Client {receiver_id} Test Metrics")
    return {
        "receiver_client": receiver_client,
        "pred_df": pred_df,
        "metrics": metrics,
        "summary_df": summary_df,
        "model_path": model_path,
        "save_dir": save_dir,
    }


def test_direct_net_load(base_cfg):
    run_cfg = build_direct_net_load_cfg(base_cfg)
    run_cfg.decentralized_gcml.save_dir = os.path.join(base_cfg.decentralized_gcml.save_dir, "direct_net_load")
    save_dir = os.path.join(run_cfg.decentralized_gcml.save_dir, "best_model_test_outputs")
    return evaluate_best_receiver_model(run_cfg, save_dir, "Decentralized GCML Direct Net Load Best Model")


def test_indirect_net_load(base_cfg):
    indirect_root = os.path.join(base_cfg.decentralized_gcml.save_dir, "indirect_net_load")

    gc_cfg = build_indirect_gc_cfg(base_cfg)
    gc_cfg.decentralized_gcml.save_dir = os.path.join(indirect_root, "gc_model")
    gc_result = evaluate_best_receiver_model(
        gc_cfg,
        save_dir=os.path.join(gc_cfg.decentralized_gcml.save_dir, "best_model_test_outputs"),
        title_prefix="Decentralized GCML GC Best Model",
    )

    gg_cfg = build_indirect_gg_cfg(base_cfg)
    gg_cfg.decentralized_gcml.save_dir = os.path.join(indirect_root, "gg_model")
    gg_result = evaluate_best_receiver_model(
        gg_cfg,
        save_dir=os.path.join(gg_cfg.decentralized_gcml.save_dir, "best_model_test_outputs"),
        title_prefix="Decentralized GCML GG Best Model",
    )

    net_load_pred_df = combine_prediction_frames(
        gc_result["pred_df"],
        gg_result["pred_df"],
        horizon=base_cfg.data.horizon,
        op="subtract",
    )
    metrics = calc_prediction_metrics(net_load_pred_df, base_cfg.data.horizon)
    receiver_id = base_cfg.decentralized_gcml.receiver_client_id
    client_name = f"client_{receiver_id}"

    summary_df = pd.DataFrame([{
        "client_id": receiver_id,
        "client_name": client_name,
        "loss": float("nan"),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE_percent": metrics["MAPE_percent"],
        "R2": metrics["R2"],
        "gc_model_path": gc_result["model_path"],
        "gg_model_path": gg_result["model_path"],
    }])

    save_dir = os.path.join(indirect_root, "best_model_test_outputs")
    save_net_load_outputs(
        net_load_client_pred_map={client_name: net_load_pred_df},
        net_load_summary_df=summary_df,
        save_dir=save_dir,
        title_prefix="Decentralized GCML Indirect Net Load Best Model",
        cfg=base_cfg,
    )

    print_metrics(metrics, title=f"Decentralized GCML Indirect Net Load Client {receiver_id} Best Model Test Metrics")
    print(f"\nBest-model test outputs saved to: {save_dir}")
    return {
        "gc_result": gc_result,
        "gg_result": gg_result,
        "pred_df": net_load_pred_df,
        "metrics": metrics,
        "summary_df": summary_df,
        "save_dir": save_dir,
    }


def test_single_target(base_cfg):
    save_dir = os.path.join(base_cfg.decentralized_gcml.save_dir, "best_model_test_outputs")
    return evaluate_best_receiver_model(base_cfg, save_dir, "Decentralized GCML Best Model")


def main():
    cfg = copy.deepcopy(CFG)
    set_default_nine_client_files(cfg)
    set_seed(cfg.train.random_seed)

    if cfg.experiment.task_type == "single_target":
        test_single_target(cfg)
        return

    if cfg.experiment.task_type != "net_load":
        raise ValueError(
            f"Unsupported experiment.task_type={cfg.experiment.task_type}. "
            "Use 'single_target' or 'net_load'."
        )

    method = cfg.experiment.net_load_method.lower()
    if method == "direct":
        test_direct_net_load(cfg)
    elif method == "indirect":
        test_indirect_net_load(cfg)
    else:
        raise ValueError(
            f"Unsupported experiment.net_load_method={cfg.experiment.net_load_method}. "
            "Use 'direct' or 'indirect'."
        )


if __name__ == "__main__":
    main()
