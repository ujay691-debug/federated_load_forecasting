import copy
import os

from utils.runtime_env import ensure_conda_dll_paths

ensure_conda_dll_paths()

import pandas as pd
import torch

from config import CFG
from federated_main import (
    build_clients,
    summarize_clients,
    summarize_h2a_clients,
    evaluate_regional_from_predictions,
    save_regional_outputs,
    save_client_outputs,
    print_metrics,
    build_direct_net_load_cfg,
    build_indirect_gc_cfg,
    build_indirect_gg_cfg,
    combine_prediction_frames,
    calc_prediction_metrics,
    save_combined_net_load_outputs,
)
from h2a_server import H2AServer
from server import FedServer
from utils.data_utils import ensure_dir


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and checkpoint.get("aggregation_method") == "h2a":
        return checkpoint, None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("client_personalizations")
    return checkpoint, None


def load_state_from_path(path, device):
    model_state_dict, _ = load_checkpoint(path, device)
    return model_state_dict


def resolve_best_model_path(run_dir, cfg):
    candidates = []
    checkpoint_dir = getattr(cfg.federated, "checkpoint_dir", None)
    if checkpoint_dir:
        candidates.extend([
            os.path.join(checkpoint_dir, "best_global_checkpoint.pth"),
            os.path.join(checkpoint_dir, cfg.federated.best_model_name),
        ])
    candidates.extend([
        os.path.join(run_dir, "best_global_checkpoint.pth"),
        os.path.join(run_dir, cfg.federated.best_model_name),
    ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def evaluate_saved_federated_model(cfg, model_path, save_dir, title_prefix):
    ensure_dir(save_dir)

    clients, feature_cols_ref = build_clients(cfg)
    state, client_personalizations = load_checkpoint(model_path, cfg.train.device)

    if isinstance(state, dict) and state.get("aggregation_method") == "h2a":
        server = H2AServer(input_dim=len(feature_cols_ref), cfg=cfg, num_clients=len(clients))
        server.load_state_dict(state["h2a_server_state"])
        summary_df, client_test_pred_map = summarize_h2a_clients(
            server, clients, split_name="test", include_predictions=True
        )
    else:
        server = FedServer(input_dim=len(feature_cols_ref), cfg=cfg)
        server.set_global_state(state)
        if client_personalizations is not None:
            for client in clients:
                client.import_personalization(client_personalizations.get(client.client_name))

        summary_df, client_test_pred_map = summarize_clients(
            server, clients, split_name="test", include_predictions=True
        )
    regional_test_df, regional_test_metrics = evaluate_regional_from_predictions(
        client_test_pred_map, clients, cfg
    )

    save_client_outputs(
        client_test_pred_map,
        summary_df,
        save_dir=save_dir,
        split_name="test",
        plot_title_prefix=title_prefix,
    )
    save_regional_outputs(
        regional_test_df,
        regional_test_metrics,
        save_dir=save_dir,
        prefix="regional_test",
        plot_title=f"{title_prefix} Regional Test Prediction",
    )

    print_metrics(regional_test_metrics, title=f"{title_prefix} Regional Test Metrics")
    return summary_df, regional_test_df, regional_test_metrics, client_test_pred_map


def test_direct_net_load_best(base_cfg):
    run_cfg = build_direct_net_load_cfg(base_cfg)
    run_dir = os.path.join(base_cfg.federated.save_dir, "direct_net_load")
    model_path = resolve_best_model_path(run_dir, run_cfg)
    save_dir = os.path.join(run_dir, "best_model_test_outputs")
    return evaluate_saved_federated_model(
        run_cfg,
        model_path=model_path,
        save_dir=save_dir,
        title_prefix="Federated Direct Net Load Best Model",
    )


def test_indirect_net_load_best(base_cfg):
    indirect_root = os.path.join(base_cfg.federated.save_dir, "indirect_net_load")
    gc_cfg = build_indirect_gc_cfg(base_cfg)
    gg_cfg = build_indirect_gg_cfg(base_cfg)
    gc_dir = os.path.join(indirect_root, "gc_model")
    gg_dir = os.path.join(indirect_root, "gg_model")
    checkpoint_root = getattr(base_cfg.federated, "checkpoint_dir", None)
    if checkpoint_root:
        gc_cfg.federated.checkpoint_dir = os.path.join(checkpoint_root, "indirect_net_load", "gc_model")
        gg_cfg.federated.checkpoint_dir = os.path.join(checkpoint_root, "indirect_net_load", "gg_model")

    gc_summary_df, gc_regional_df, gc_regional_metrics, gc_pred_map = evaluate_saved_federated_model(
        gc_cfg,
        model_path=resolve_best_model_path(gc_dir, gc_cfg),
        save_dir=os.path.join(gc_dir, "best_model_test_outputs"),
        title_prefix="Federated GC Best Model",
    )
    gg_summary_df, gg_regional_df, gg_regional_metrics, gg_pred_map = evaluate_saved_federated_model(
        gg_cfg,
        model_path=resolve_best_model_path(gg_dir, gg_cfg),
        save_dir=os.path.join(gg_dir, "best_model_test_outputs"),
        title_prefix="Federated GG Best Model",
    )

    net_load_client_pred_map = {}
    summary_rows = []
    horizon = base_cfg.data.horizon

    for client_name, gc_pred_df in gc_pred_map.items():
        gg_pred_df = gg_pred_map[client_name]
        net_load_pred_df = combine_prediction_frames(
            gc_pred_df, gg_pred_df, horizon=horizon, op="subtract"
        )
        metrics = calc_prediction_metrics(net_load_pred_df, horizon)
        client_id = int(client_name.split("_")[-1])
        summary_rows.append({
            "client_id": client_id,
            "client_name": client_name,
            "loss": float("nan"),
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "R2": metrics["R2"],
        })
        net_load_client_pred_map[client_name] = net_load_pred_df

    net_load_summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    regional_net_load_df = combine_prediction_frames(
        gc_regional_df,
        gg_regional_df,
        horizon=horizon,
        op="subtract",
    )
    regional_net_load_metrics = calc_prediction_metrics(regional_net_load_df, horizon)

    save_dir = os.path.join(indirect_root, "best_model_test_outputs")
    save_combined_net_load_outputs(
        net_load_client_pred_map=net_load_client_pred_map,
        net_load_summary_df=net_load_summary_df,
        regional_net_load_df=regional_net_load_df,
        regional_net_load_metrics=regional_net_load_metrics,
        save_dir=save_dir,
        title_prefix="Federated Indirect Net Load Best Model",
    )

    compare_df = pd.DataFrame([
        {"component": "gc_model_regional", **gc_regional_metrics},
        {"component": "gg_model_regional", **gg_regional_metrics},
        {"component": "indirect_net_load_regional", **regional_net_load_metrics},
    ])
    compare_df.to_csv(
        os.path.join(save_dir, "indirect_net_load_component_compare.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print_metrics(
        regional_net_load_metrics,
        title="Federated Indirect Net Load Best Model Regional Test Metrics",
    )
    print("\nIndirect net load best-model per-client summary:")
    print(net_load_summary_df)

    return {
        "gc_summary_df": gc_summary_df,
        "gg_summary_df": gg_summary_df,
        "regional_test_df": regional_net_load_df,
        "regional_test_metrics": regional_net_load_metrics,
        "client_test_summary_df": net_load_summary_df,
        "client_test_pred_map": net_load_client_pred_map,
    }


def main():
    cfg = copy.deepcopy(CFG)

    if cfg.experiment.task_type == "single_target":
        model_path = resolve_best_model_path(cfg.federated.save_dir, cfg)
        save_dir = os.path.join(cfg.federated.save_dir, "best_model_test_outputs")
        evaluate_saved_federated_model(
            cfg,
            model_path=model_path,
            save_dir=save_dir,
            title_prefix=f"Federated {cfg.data.target_col} Best Model",
        )
        return

    if cfg.experiment.task_type != "net_load":
        raise ValueError(
            f"Unsupported experiment.task_type={cfg.experiment.task_type}. "
            "Use 'single_target' or 'net_load'."
        )

    method = cfg.experiment.net_load_method.lower()
    if method == "direct":
        test_direct_net_load_best(cfg)
    elif method == "indirect":
        test_indirect_net_load_best(cfg)
    else:
        raise ValueError(
            f"Unsupported experiment.net_load_method={cfg.experiment.net_load_method}. "
            "Use 'direct' or 'indirect'."
        )


if __name__ == "__main__":
    main()
