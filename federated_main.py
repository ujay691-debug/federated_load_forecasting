import copy
import math
import os
import random
from typing import Dict, List, Tuple

from utils.runtime_env import ensure_conda_dll_paths

ensure_conda_dll_paths()

import matplotlib.pyplot as plt
import pandas as pd
import torch

from client import FederatedClient
from config import CFG, PROJECT_ROOT
from h2a_server import H2AServer
from server import FedServer
from utils.data_utils import add_timestamp_occurrence_key, ensure_dir, save_config, set_seed
from utils.metrics import calc_metrics, plot_round_curve, plot_true_pred, print_metrics, save_metrics_csv


def select_clients(clients, fraction: float):
    if fraction >= 1.0:
        return clients
    num_selected = max(1, int(round(len(clients) * fraction)))
    return random.sample(clients, num_selected)


def build_clients(cfg) -> Tuple[List[FederatedClient], List[str]]:
    clients = []
    feature_cols_ref = None

    for idx, path in enumerate(cfg.data.client_files, start=1):
        client = FederatedClient(client_id=idx, data_path=path, cfg=cfg)
        if feature_cols_ref is None:
            feature_cols_ref = client.feature_cols
        elif feature_cols_ref != client.feature_cols:
            raise ValueError(
                f"Client {idx} feature columns do not match.\n"
                f"Reference: {feature_cols_ref}\n"
                f"Current: {client.feature_cols}"
            )
        clients.append(client)

    return clients, feature_cols_ref


def merge_regional_predictions(client_pred_dfs, client_ids, horizon: int):
    if len(client_pred_dfs) == 0:
        raise ValueError("No client predictions were provided for regional aggregation.")

    merged = None
    for df, cid in zip(client_pred_dfs, client_ids):
        tmp = add_timestamp_occurrence_key(df, "timestamp")
        rename_map = {}
        for step in range(horizon):
            rename_map[f"y_true_step_{step + 1}"] = f"y_true_step_{step + 1}_client_{cid}"
            rename_map[f"y_pred_step_{step + 1}"] = f"y_pred_step_{step + 1}_client_{cid}"
        tmp = tmp.rename(columns=rename_map)

        if merged is None:
            merged = tmp
        else:
            merged = pd.merge(merged, tmp, on=["timestamp", "_timestamp_occurrence"], how="inner")

    if merged is None or len(merged) == 0:
        raise ValueError("Regional aggregation produced no aligned samples.")

    out = pd.DataFrame({"timestamp": merged["timestamp"].copy()})
    for step in range(horizon):
        true_cols = [c for c in merged.columns if c.startswith(f"y_true_step_{step + 1}_client_")]
        pred_cols = [c for c in merged.columns if c.startswith(f"y_pred_step_{step + 1}_client_")]
        out[f"y_true_step_{step + 1}"] = merged[true_cols].sum(axis=1)
        out[f"y_pred_step_{step + 1}"] = merged[pred_cols].sum(axis=1)

    return out.sort_values("timestamp").reset_index(drop=True)


def combine_prediction_frames(lhs_df: pd.DataFrame, rhs_df: pd.DataFrame, horizon: int, op: str = "subtract") -> pd.DataFrame:
    left = add_timestamp_occurrence_key(lhs_df, "timestamp")
    right = add_timestamp_occurrence_key(rhs_df, "timestamp")
    merged = pd.merge(left, right, on=["timestamp", "_timestamp_occurrence"], how="inner", suffixes=("_lhs", "_rhs"))

    if len(merged) == 0:
        raise ValueError("No aligned samples were found when combining prediction frames.")

    out = pd.DataFrame({"timestamp": merged["timestamp"].copy()})
    for step in range(horizon):
        true_lhs = merged[f"y_true_step_{step + 1}_lhs"]
        true_rhs = merged[f"y_true_step_{step + 1}_rhs"]
        pred_lhs = merged[f"y_pred_step_{step + 1}_lhs"]
        pred_rhs = merged[f"y_pred_step_{step + 1}_rhs"]

        if op == "subtract":
            out[f"y_true_step_{step + 1}"] = true_lhs - true_rhs
            out[f"y_pred_step_{step + 1}"] = pred_lhs - pred_rhs
        elif op == "add":
            out[f"y_true_step_{step + 1}"] = true_lhs + true_rhs
            out[f"y_pred_step_{step + 1}"] = pred_lhs + pred_rhs
        else:
            raise ValueError(f"Unsupported combine operation: {op}")

    return out.sort_values("timestamp").reset_index(drop=True)


def calc_prediction_metrics(pred_df: pd.DataFrame, horizon: int) -> dict:
    true_cols = [f"y_true_step_{i + 1}" for i in range(horizon)]
    pred_cols = [f"y_pred_step_{i + 1}" for i in range(horizon)]
    return calc_metrics(
        pred_df[true_cols].values.reshape(-1),
        pred_df[pred_cols].values.reshape(-1),
    )


def summarize_clients(server, clients, split_name: str = "test", include_predictions: bool = False):
    global_state = server.get_global_state()
    use_personalized_head = getattr(server.cfg.federated, "use_head_personalization", False)
    summary_rows = []
    pred_map = {}

    for client in clients:
        pred_df, metrics, loss = client.evaluate_split(
            global_state,
            split_name=split_name,
            use_personalized_head=use_personalized_head,
        )
        summary_rows.append({
            "client_id": client.client_id,
            "client_name": client.client_name,
            "loss": loss,
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "R2": metrics["R2"],
        })
        if include_predictions:
            pred_map[client.client_name] = pred_df

    summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    if include_predictions:
        return summary_df, pred_map
    return summary_df


def summarize_h2a_clients(server, clients, split_name: str = "test", include_predictions: bool = False):
    summary_rows = []
    pred_map = {}

    for client in clients:
        eval_state = server.get_eval_state(client.client_id)
        pred_df, metrics, loss = client.evaluate_split(
            eval_state,
            split_name=split_name,
            use_personalized_head=False,
        )
        summary_rows.append({
            "client_id": client.client_id,
            "client_name": client.client_name,
            "loss": loss,
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "R2": metrics["R2"],
        })
        if include_predictions:
            pred_map[client.client_name] = pred_df

    summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    if include_predictions:
        return summary_df, pred_map
    return summary_df


def evaluate_regional_from_predictions(client_pred_map: Dict[str, pd.DataFrame], clients, cfg):
    client_pred_dfs = [client_pred_map[client.client_name] for client in clients]
    client_ids = [client.client_id for client in clients]
    regional_df = merge_regional_predictions(client_pred_dfs, client_ids, cfg.data.horizon)
    metrics = calc_prediction_metrics(regional_df, cfg.data.horizon)
    return regional_df, metrics


def plot_client_metric_bar(summary_df, metric_name, save_path, title):
    if metric_name not in summary_df.columns:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["client_name"], summary_df[metric_name].values)
    plt.xlabel("Client")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_client_outputs(client_pred_map: Dict[str, pd.DataFrame], summary_df: pd.DataFrame, save_dir: str, split_name: str, plot_title_prefix: str):
    per_client_dir = os.path.join(save_dir, "per_client_results")
    ensure_dir(per_client_dir)

    for row in summary_df.to_dict("records"):
        client_name = row["client_name"]
        pred_df = client_pred_map[client_name]
        client_dir = os.path.join(per_client_dir, client_name)
        ensure_dir(client_dir)

        pred_csv_path = os.path.join(client_dir, f"{client_name}_{split_name}_predictions.csv")
        metrics_csv_path = os.path.join(client_dir, f"{client_name}_{split_name}_metrics.csv")
        fig_path = os.path.join(client_dir, f"{client_name}_{split_name}_prediction.png")

        metrics = {
            "MAE": row["MAE"],
            "MSE": row["MSE"],
            "RMSE": row["RMSE"],
            "MAPE_percent": row["MAPE_percent"],
            "R2": row["R2"],
        }

        pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
        save_metrics_csv(metrics, metrics_csv_path)
        plot_true_pred(
            pred_df["y_true_step_1"].values,
            pred_df["y_pred_step_1"].values,
            save_path=fig_path,
            title=f"{plot_title_prefix} {client_name} {split_name.title()} Prediction",
            show_n=300,
        )

    summary_path = os.path.join(per_client_dir, f"all_clients_{split_name}_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_client_metric_bar(
        summary_df,
        metric_name="MAE",
        save_path=os.path.join(per_client_dir, f"all_clients_{split_name}_mae_bar.png"),
        title=f"All Clients {split_name.title()} MAE",
    )
    plot_client_metric_bar(
        summary_df,
        metric_name="RMSE",
        save_path=os.path.join(per_client_dir, f"all_clients_{split_name}_rmse_bar.png"),
        title=f"All Clients {split_name.title()} RMSE",
    )
    plot_client_metric_bar(
        summary_df,
        metric_name="R2",
        save_path=os.path.join(per_client_dir, f"all_clients_{split_name}_r2_bar.png"),
        title=f"All Clients {split_name.title()} R2",
    )


def save_regional_outputs(regional_df: pd.DataFrame, metrics: dict, save_dir: str, prefix: str, plot_title: str):
    regional_df.to_csv(
        os.path.join(save_dir, f"{prefix}_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_metrics_csv(metrics, os.path.join(save_dir, f"{prefix}_metrics.csv"))
    plot_true_pred(
        regional_df["y_true_step_1"].values,
        regional_df["y_pred_step_1"].values,
        save_path=os.path.join(save_dir, f"{prefix}_prediction.png"),
        title=plot_title,
        show_n=300,
    )


def save_training_curves(save_dir: str, regional_val_rmse_curve, regional_val_mae_curve, regional_val_r2_curve, avg_client_val_rmse_curve):
    if len(avg_client_val_rmse_curve) > 0:
        plot_round_curve(
            avg_client_val_rmse_curve,
            title="Average Client Validation RMSE Curve",
            xlabel="Evaluation Round",
            ylabel="RMSE",
            save_path=os.path.join(save_dir, "avg_client_val_rmse_curve.png"),
        )

    if len(regional_val_rmse_curve) > 0:
        plot_round_curve(
            regional_val_rmse_curve,
            title="Regional Validation RMSE Curve",
            xlabel="Evaluation Round",
            ylabel="RMSE",
            save_path=os.path.join(save_dir, "regional_val_rmse_curve.png"),
        )
        plot_round_curve(
            regional_val_mae_curve,
            title="Regional Validation MAE Curve",
            xlabel="Evaluation Round",
            ylabel="MAE",
            save_path=os.path.join(save_dir, "regional_val_mae_curve.png"),
        )
        plot_round_curve(
            regional_val_r2_curve,
            title="Regional Validation R2 Curve",
            xlabel="Evaluation Round",
            ylabel="R2",
            save_path=os.path.join(save_dir, "regional_val_r2_curve.png"),
        )


def save_h2a_logs(save_dir: str, alpha_logs, reference_logs, layer_weight_logs, round_summary_logs):
    if len(alpha_logs) > 0:
        pd.DataFrame(alpha_logs).to_csv(
            os.path.join(save_dir, "h2a_alpha_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if len(reference_logs) > 0:
        pd.DataFrame(reference_logs).to_csv(
            os.path.join(save_dir, "h2a_reference_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if len(layer_weight_logs) > 0:
        pd.DataFrame(layer_weight_logs).to_csv(
            os.path.join(save_dir, "h2a_layer_weight_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if len(round_summary_logs) > 0:
        pd.DataFrame(round_summary_logs).to_csv(
            os.path.join(save_dir, "h2a_round_summary_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def save_h2a_importance_matrix(server, save_dir: str, filename: str = "h2a_importance_matrix_final.csv"):
    matrix = server.importance_matrix.detach().cpu().numpy()
    labels = [f"client_{idx}" for idx in range(1, server.num_clients + 1)]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        os.path.join(save_dir, filename),
        encoding="utf-8-sig",
    )


def make_h2a_checkpoint(server, rnd, best_avg_client_val_rmse, regional_val_metrics, avg_client_val_metrics):
    return {
        "aggregation_method": "h2a",
        "h2a_server_state": server.state_dict(),
        "best_round": rnd,
        "best_avg_client_val_rmse": best_avg_client_val_rmse,
        "regional_val_metrics": regional_val_metrics,
        "avg_client_val_metrics": avg_client_val_metrics,
    }


def flatten_head_dict(tensor_dict, head_param_names):
    flat_tensors = []
    meta_list = []
    global_flat_index = 0

    if tensor_dict is None:
        return torch.tensor([], dtype=torch.float32), meta_list

    for name in head_param_names:
        if name not in tensor_dict or tensor_dict[name] is None:
            continue

        flat_tensor = tensor_dict[name].detach().cpu().float().reshape(-1)
        layer_name = name.split(".")[0]
        for local_flat_index in range(flat_tensor.numel()):
            meta_list.append({
                "param_name": name,
                "flat_index_in_param": local_flat_index,
                "global_flat_index": global_flat_index,
                "layer_name": layer_name,
            })
            global_flat_index += 1
        flat_tensors.append(flat_tensor)

    if len(flat_tensors) == 0:
        return torch.tensor([], dtype=torch.float32), meta_list

    return torch.cat(flat_tensors).float(), meta_list


def clean_numeric_values(values):
    if values is None:
        return []
    if torch.is_tensor(values):
        values = values.detach().cpu().reshape(-1).tolist()

    clean_values = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if not math.isnan(value):
            clean_values.append(value)
    return clean_values


def safe_mean(values):
    values = clean_numeric_values(values)
    if len(values) == 0:
        return None
    return float(sum(values) / len(values))


def safe_median(values):
    values = sorted(clean_numeric_values(values))
    if len(values) == 0:
        return None
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def safe_min(values):
    values = clean_numeric_values(values)
    if len(values) == 0:
        return None
    return float(min(values))


def safe_max(values):
    values = clean_numeric_values(values)
    if len(values) == 0:
        return None
    return float(max(values))


def safe_std(values):
    values = clean_numeric_values(values)
    if len(values) == 0:
        return None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(max(variance, 0.0)))


def mask_jaccard(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return None

    mask_a = mask_a.detach().cpu().reshape(-1).bool()
    mask_b = mask_b.detach().cpu().reshape(-1).bool()
    if mask_a.numel() == 0 or mask_a.shape != mask_b.shape:
        return None

    intersection = torch.logical_and(mask_a, mask_b).sum().item()
    union = torch.logical_or(mask_a, mask_b).sum().item()
    if union == 0:
        return None
    return float(intersection / union)


def mask_hamming_change_rate(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return None

    mask_a = mask_a.detach().cpu().reshape(-1)
    mask_b = mask_b.detach().cpu().reshape(-1)
    if mask_a.numel() == 0 or mask_a.shape != mask_b.shape:
        return None

    return float((mask_a != mask_b).float().mean().item())


def tensor_sum_or_zero(values):
    if values is None or values.numel() == 0:
        return 0.0
    return float(values.sum().item())


def top_bottom_importance_means(flat_importance):
    if flat_importance.numel() == 0:
        return None, None

    k = max(1, int(math.ceil(flat_importance.numel() * 0.1)))
    sorted_values = torch.sort(flat_importance.detach().cpu().float().reshape(-1)).values
    bottom_mean = float(sorted_values[:k].mean().item())
    top_mean = float(sorted_values[-k:].mean().item())
    return top_mean, bottom_mean


def build_importance_stats(flat_mask, flat_importance):
    selected_values = flat_importance[flat_mask == 1]
    unselected_values = flat_importance[flat_mask == 0]

    selected_mean = safe_mean(selected_values)
    unselected_mean = safe_mean(unselected_values)
    selected_median = safe_median(selected_values)
    unselected_median = safe_median(unselected_values)
    selected_min = safe_min(selected_values)
    selected_max = safe_max(selected_values)
    unselected_min = safe_min(unselected_values)
    unselected_max = safe_max(unselected_values)
    selected_total = tensor_sum_or_zero(selected_values)
    unselected_total = tensor_sum_or_zero(unselected_values)
    top10_mean, bottom10_mean = top_bottom_importance_means(flat_importance)

    if unselected_values.numel() > 0 and selected_mean is not None and unselected_mean is not None:
        gap_mean_ratio = selected_mean / (unselected_mean + 1e-12)
    else:
        gap_mean_ratio = None

    if unselected_values.numel() > 0 and selected_median is not None and unselected_median is not None:
        gap_median_ratio = selected_median / (unselected_median + 1e-12)
    else:
        gap_median_ratio = None

    if unselected_values.numel() > 0 and selected_min is not None and unselected_max is not None:
        gap_boundary_ratio = selected_min / (unselected_max + 1e-12)
    else:
        gap_boundary_ratio = None

    return {
        "selected_importance_mean": selected_mean,
        "unselected_importance_mean": unselected_mean,
        "selected_importance_median": selected_median,
        "unselected_importance_median": unselected_median,
        "selected_importance_min": selected_min,
        "selected_importance_max": selected_max,
        "unselected_importance_min": unselected_min,
        "unselected_importance_max": unselected_max,
        "importance_threshold": selected_min,
        "importance_gap_mean_ratio": gap_mean_ratio,
        "importance_gap_median_ratio": gap_median_ratio,
        "importance_gap_boundary_ratio": gap_boundary_ratio,
        "importance_selected_total": selected_total,
        "importance_unselected_total": unselected_total,
        "importance_selected_share": selected_total / (selected_total + unselected_total + 1e-12),
        "importance_top10_mean": top10_mean,
        "importance_bottom10_mean": bottom10_mean,
    }


def ordered_head_layer_names(meta_list, head_prefixes):
    layer_names = [meta["layer_name"] for meta in meta_list]
    ordered = []
    for prefix in head_prefixes:
        if prefix in layer_names and prefix not in ordered:
            ordered.append(prefix)
    for layer_name in layer_names:
        if layer_name not in ordered:
            ordered.append(layer_name)
    return ordered


def build_head_personalization_logs(rnd, selected_clients, client_updates, previous_mask_cache, param_frequency_cache):
    head_mask_round_rows = []
    head_layer_mask_rows = []

    for client, update in zip(selected_clients, client_updates):
        head_mask = update.get("head_mask")
        head_importance = update.get("head_importance")
        head_param_names = update.get("head_param_names") or []
        head_prefixes = update.get("head_prefixes") or []
        head_exact_names = update.get("head_exact_names") or []
        use_head_importance_ema = update.get("use_head_importance_ema", False)
        head_importance_ema_beta = update.get("head_importance_ema_beta")

        if not isinstance(head_mask, dict) or not isinstance(head_importance, dict):
            continue
        if len(head_mask) == 0 or len(head_importance) == 0:
            continue

        flat_mask, meta_list = flatten_head_dict(head_mask, head_param_names)
        flat_importance, importance_meta_list = flatten_head_dict(head_importance, head_param_names)
        if flat_mask.numel() == 0 or flat_importance.numel() == 0:
            continue
        if flat_mask.shape != flat_importance.shape or len(meta_list) != len(importance_meta_list):
            continue

        flat_mask = (flat_mask > 0).float()
        tau = getattr(client.cfg.federated, "head_personalization_tau", None)
        prev_mask = previous_mask_cache.get(client.client_name)
        selected_count = int(flat_mask.sum().item())
        num_head_params = int(flat_mask.numel())
        stats = build_importance_stats(flat_mask, flat_importance)

        head_mask_round_rows.append({
            "round": rnd,
            "client_id": client.client_id,
            "client_name": client.client_name,
            "tau": tau,
            "head_prefixes": ",".join(str(prefix) for prefix in head_prefixes),
            "head_exact_names": ",".join(str(name) for name in head_exact_names),
            "num_head_params": num_head_params,
            "num_selected_params": selected_count,
            "selected_ratio": selected_count / num_head_params if num_head_params > 0 else None,
            "mask_jaccard_with_prev_round": mask_jaccard(prev_mask, flat_mask),
            "mask_hamming_change_rate": mask_hamming_change_rate(prev_mask, flat_mask),
            "use_head_importance_ema": use_head_importance_ema,
            "head_importance_ema_beta": head_importance_ema_beta,
            **stats,
        })

        for layer_name in ordered_head_layer_names(meta_list, head_prefixes):
            layer_indices = [
                meta_idx for meta_idx, meta in enumerate(meta_list)
                if meta["layer_name"] == layer_name
            ]
            if len(layer_indices) == 0:
                continue

            index_tensor = torch.tensor(layer_indices, dtype=torch.long)
            layer_mask = flat_mask[index_tensor]
            layer_importance = flat_importance[index_tensor]
            prev_layer_mask = None
            if prev_mask is not None and prev_mask.numel() == flat_mask.numel():
                prev_layer_mask = prev_mask.detach().cpu().reshape(-1)[index_tensor]

            layer_stats = build_importance_stats(layer_mask, layer_importance)
            num_layer_params = int(layer_mask.numel())
            num_selected_layer_params = int(layer_mask.sum().item())
            head_layer_mask_rows.append({
                "round": rnd,
                "client_id": client.client_id,
                "client_name": client.client_name,
                "tau": tau,
                "layer_name": layer_name,
                "num_layer_params": num_layer_params,
                "num_selected_params": num_selected_layer_params,
                "selected_ratio": num_selected_layer_params / num_layer_params if num_layer_params > 0 else None,
                "mask_jaccard_with_prev_round_layer": mask_jaccard(prev_layer_mask, layer_mask),
                "mask_hamming_change_rate_layer": mask_hamming_change_rate(prev_layer_mask, layer_mask),
                "selected_importance_mean": layer_stats["selected_importance_mean"],
                "unselected_importance_mean": layer_stats["unselected_importance_mean"],
                "selected_importance_median": layer_stats["selected_importance_median"],
                "unselected_importance_median": layer_stats["unselected_importance_median"],
                "selected_importance_min": layer_stats["selected_importance_min"],
                "selected_importance_max": layer_stats["selected_importance_max"],
                "unselected_importance_min": layer_stats["unselected_importance_min"],
                "unselected_importance_max": layer_stats["unselected_importance_max"],
                "importance_gap_mean_ratio": layer_stats["importance_gap_mean_ratio"],
                "importance_gap_boundary_ratio": layer_stats["importance_gap_boundary_ratio"],
                "importance_selected_share": layer_stats["importance_selected_share"],
                "use_head_importance_ema": use_head_importance_ema,
                "head_importance_ema_beta": head_importance_ema_beta,
            })

        sorted_indices = torch.argsort(flat_importance, descending=True)
        ranks = torch.empty_like(sorted_indices, dtype=torch.long)
        ranks[sorted_indices] = torch.arange(1, flat_importance.numel() + 1, dtype=torch.long)
        for meta_idx, meta in enumerate(meta_list):
            key = (client.client_name, meta["param_name"], meta["flat_index_in_param"])
            if key not in param_frequency_cache:
                param_frequency_cache[key] = {
                    "client_id": client.client_id,
                    "client_name": client.client_name,
                    "param_name": meta["param_name"],
                    "flat_index_in_param": meta["flat_index_in_param"],
                    "layer_name": meta["layer_name"],
                    "selected_count": 0,
                    "total_rounds": 0,
                    "importance_sum": 0.0,
                    "importance_sq_sum": 0.0,
                    "rank_sum": 0.0,
                }

            importance_value = float(flat_importance[meta_idx].item())
            cache_item = param_frequency_cache[key]
            cache_item["selected_count"] += int(flat_mask[meta_idx].item())
            cache_item["total_rounds"] += 1
            cache_item["importance_sum"] += importance_value
            cache_item["importance_sq_sum"] += importance_value ** 2
            cache_item["rank_sum"] += float(ranks[meta_idx].item())

        previous_mask_cache[client.client_name] = flat_mask.detach().cpu().clone()

    return head_mask_round_rows, head_layer_mask_rows


def finalize_head_param_frequency_logs(param_frequency_cache):
    rows = []
    for item in param_frequency_cache.values():
        total_rounds = int(item["total_rounds"])
        if total_rounds <= 0:
            continue

        importance_mean = item["importance_sum"] / total_rounds
        importance_sq_mean = item["importance_sq_sum"] / total_rounds
        importance_variance = max(importance_sq_mean - importance_mean ** 2, 0.0)
        rows.append({
            "client_id": item["client_id"],
            "client_name": item["client_name"],
            "layer_name": item["layer_name"],
            "param_name": item["param_name"],
            "flat_index_in_param": item["flat_index_in_param"],
            "selected_count": item["selected_count"],
            "total_rounds": total_rounds,
            "selected_frequency": item["selected_count"] / total_rounds,
            "importance_mean": importance_mean,
            "importance_std": math.sqrt(importance_variance),
            "importance_rank_mean": item["rank_sum"] / total_rounds,
        })

    return sorted(
        rows,
        key=lambda row: (
            row["client_id"],
            row["layer_name"],
            row["param_name"],
            row["flat_index_in_param"],
        ),
    )


def format_optional_metric(value, digits=4):
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def train_federated_model(cfg, save_dir: str, run_label: str):
    run_cfg = copy.deepcopy(cfg)
    run_cfg.federated.save_dir = save_dir

    set_seed(run_cfg.train.random_seed)
    ensure_dir(run_cfg.federated.save_dir)
    checkpoint_dir = getattr(run_cfg.federated, "checkpoint_dir", None) or run_cfg.federated.save_dir
    ensure_dir(checkpoint_dir)
    save_config(run_cfg, run_cfg.federated.save_dir)
    if checkpoint_dir != run_cfg.federated.save_dir:
        save_config(run_cfg, checkpoint_dir)

    clients, feature_cols_ref = build_clients(run_cfg)
    aggregation_method = getattr(run_cfg.federated, "aggregation_method", "fedavg").lower()
    if aggregation_method not in ("fedavg", "h2a"):
        raise ValueError("cfg.federated.aggregation_method must be 'fedavg' or 'h2a'.")
    if aggregation_method == "h2a":
        server = H2AServer(input_dim=len(feature_cols_ref), cfg=run_cfg, num_clients=len(clients))
    else:
        server = FedServer(input_dim=len(feature_cols_ref), cfg=run_cfg)

    round_logs = []
    regional_val_rmse_curve = []
    regional_val_mae_curve = []
    regional_val_r2_curve = []
    avg_client_val_rmse_curve = []
    h2a_alpha_logs = []
    h2a_reference_logs = []
    h2a_layer_weight_logs = []
    h2a_round_summary_logs = []
    fedavg_rc_rg_client_logs = []
    head_mask_round_logs = []
    head_layer_mask_logs = []
    head_param_frequency_cache = {}
    previous_head_mask_cache = {}

    best_avg_client_val_rmse = float("inf")
    best_model_path = os.path.join(checkpoint_dir, run_cfg.federated.best_model_name)
    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        getattr(run_cfg.federated, "best_checkpoint_name", "best_global_checkpoint.pth"),
    )
    final_model_path = os.path.join(
        checkpoint_dir,
        getattr(run_cfg.federated, "final_model_name", "final_global_model.pth"),
    )
    best_model_saved = False
    best_checkpoint_saved = False
    best_client_personalizations = None
    early_stop_patience = getattr(run_cfg.federated, "early_stop_patience", 0)
    no_improve_rounds = 0

    print("=" * 100)
    print(f"{run_label} training started")
    print(f"Task target: {run_cfg.data.target_col}")
    print(f"Input features: {feature_cols_ref}")
    print(f"Clients: {len(clients)}")
    print(f"Aggregation: {aggregation_method}")
    print(f"Device: {run_cfg.train.device}")
    print("=" * 100)

    for rnd in range(1, run_cfg.federated.rounds + 1):
        should_stop = False
        selected_clients = select_clients(clients, run_cfg.federated.client_fraction)
        client_updates = []
        h2a_alpha_values = []
        h2a_distance_values = []
        h2a_meta_losses = []
        h2a_ref_summary_items = []
        h2a_ref_counts = []
        use_rc = (
            getattr(run_cfg.federated, "use_rc_regularization", False)
            if aggregation_method == "fedavg" else False
        )
        rc_lambda = getattr(run_cfg.federated, "rc_lambda", 0.0) if use_rc else 0.0
        head_warmup_rounds = int(getattr(run_cfg.federated, "head_personalization_warmup_rounds", 0))
        head_mask_update_interval = int(getattr(run_cfg.federated, "head_mask_update_interval", 1))
        use_head_cfg = (
            getattr(run_cfg.federated, "use_head_personalization", False)
            if aggregation_method == "fedavg" else False
        )
        use_head_this_round = bool(use_head_cfg and rnd > head_warmup_rounds)
        if use_head_this_round:
            after_warmup_round_idx = rnd - head_warmup_rounds
            update_head_mask_this_round = (
                after_warmup_round_idx == 1
                or head_mask_update_interval <= 1
                or ((after_warmup_round_idx - 1) % head_mask_update_interval == 0)
            )
        else:
            update_head_mask_this_round = False
        round_head_rows = []
        layer_head_rows = []

        if aggregation_method == "h2a":
            server.begin_round()
            for client in selected_clients:
                graph_state, init_state, h2a_info = server.build_personalized_state(
                    client.client_id,
                    training=True,
                )
                ref_client_ids = h2a_info["ref_client_ids"]
                h2a_ref_counts.append(len(ref_client_ids))
                h2a_ref_summary_items.append(
                    f"{client.client_name}->{ref_client_ids}"
                )
                importance_values_for_refs = [
                    float(server.importance_matrix[client.client_id - 1, ref_id - 1].item())
                    for ref_id in ref_client_ids
                ]
                update = client.local_update(
                    global_state_dict=init_state,
                    local_epochs=run_cfg.federated.local_epochs,
                    global_rc=None,
                    rc_lambda=0.0,
                )
                meta_loss = server.meta_update(
                    client.client_id,
                    graph_state,
                    update["delta_state"],
                )
                server.queue_client_state(client.client_id, update["state_dict"])
                server.update_importance(
                    client.client_id,
                    ref_client_ids,
                    h2a_info["weights"],
                )
                client_updates.append(update)
                h2a_alpha_values.append(float(h2a_info["alpha"]))
                h2a_distance_values.append(float(h2a_info["distance_avg"]))
                h2a_meta_losses.append(float(meta_loss))

                h2a_alpha_logs.append({
                    "round": rnd,
                    "client_id": client.client_id,
                    "client_name": client.client_name,
                    "alpha": float(h2a_info["alpha"]),
                    "distance_avg": float(h2a_info["distance_avg"]),
                    "gamma": float(h2a_info["gamma"]),
                    "ref_client_ids": ref_client_ids,
                    "train_loss": float(update["train_loss"]),
                    "val_loss": float(update["val_loss"]),
                    "meta_loss": float(meta_loss),
                })
                h2a_reference_logs.append({
                    "round": rnd,
                    "client_id": client.client_id,
                    "client_name": client.client_name,
                    "reference_mode": getattr(run_cfg.federated, "h2a_reference_mode", "adaptive"),
                    "ref_client_ids": ref_client_ids,
                    "aggregation_client_ids": ref_client_ids,
                    "num_refs": len(ref_client_ids),
                    "importance_values_for_refs": importance_values_for_refs,
                })

                weights = h2a_info["weights"]
                self_weights = h2a_info["self_weights"]
                for layer_idx, layer_name in enumerate(h2a_info["feature_layer_names"]):
                    self_weight = float(self_weights[layer_idx].item())
                    for ref_pos, ref_id in enumerate(ref_client_ids):
                        h2a_layer_weight_logs.append({
                            "round": rnd,
                            "client_id": client.client_id,
                            "layer_name": layer_name,
                            "ref_client_id": ref_id,
                            "weight": float(weights[ref_pos, layer_idx].item()),
                            "self_weight": self_weight,
                            "alpha": float(h2a_info["alpha"]),
                        })
            server.commit_round()
        else:
            for client in selected_clients:
                update = client.local_update(
                    global_state_dict=server.get_global_state(),
                    local_epochs=run_cfg.federated.local_epochs,
                    global_rc=server.get_global_rc(),
                    rc_lambda=rc_lambda,
                    enable_head_personalization=use_head_this_round,
                    update_head_mask=update_head_mask_this_round,
                )
                client_updates.append(update)

            server.aggregate(client_updates)
            if use_head_this_round:
                round_head_rows, layer_head_rows = build_head_personalization_logs(
                    rnd=rnd,
                    selected_clients=selected_clients,
                    client_updates=client_updates,
                    previous_mask_cache=previous_head_mask_cache,
                    param_frequency_cache=head_param_frequency_cache,
                )
                head_mask_round_logs.extend(round_head_rows)
                head_layer_mask_logs.extend(layer_head_rows)

        avg_local_train_task_loss = sum(item["train_task_loss"] for item in client_updates) / len(client_updates)
        avg_local_train_rc_loss = sum(item["train_rc_loss"] for item in client_updates) / len(client_updates)
        avg_local_train_weighted_rc_loss = sum(item["train_weighted_rc_loss"] for item in client_updates) / len(client_updates)
        avg_local_train_total_loss = sum(item["train_total_loss"] for item in client_updates) / len(client_updates)
        avg_local_train_rc_to_task_ratio = sum(item["train_rc_to_task_ratio"] for item in client_updates) / len(client_updates)
        round_rc_to_task_ratio = avg_local_train_weighted_rc_loss / (avg_local_train_task_loss + 1e-12)
        avg_personalized_head_ratio = sum(item["personalized_head_ratio"] for item in client_updates) / len(client_updates)
        avg_num_personalized_head_params = sum(item["num_personalized_head_params"] for item in client_updates) / len(client_updates)
        avg_num_head_params = sum(item["num_head_params"] for item in client_updates) / len(client_updates)
        avg_head_importance_mean = sum(item["head_importance_mean"] for item in client_updates) / len(client_updates)
        avg_train_loss = avg_local_train_task_loss
        avg_val_loss = sum(item["val_loss"] for item in client_updates) / len(client_updates)
        global_rc = server.get_global_rc() if aggregation_method == "fedavg" else None
        global_rc_norm = None if global_rc is None else global_rc.norm().item()
        fedavg_rc_rg_l2_norms = []
        if aggregation_method == "fedavg" and global_rc is not None:
            global_rc_cpu = global_rc.detach().cpu().float()
            global_rc_l2_norm = float(torch.norm(global_rc_cpu, p=2).item())
            for client, item in zip(selected_clients, client_updates):
                local_rc = item.get("local_rc")
                if local_rc is None:
                    continue

                local_rc_cpu = local_rc.detach().cpu().float()
                rc_rg_delta = local_rc_cpu - global_rc_cpu
                rc_rg_l2_norm = float(torch.norm(rc_rg_delta, p=2).item())
                fedavg_rc_rg_l2_norms.append(rc_rg_l2_norm)
                fedavg_rc_rg_client_logs.append({
                    "round": rnd,
                    "client_id": client.client_id,
                    "client_name": client.client_name,
                    "num_samples": item["num_samples"],
                    "local_rc_norm": float(torch.norm(local_rc_cpu, p=2).item()),
                    "global_rc_norm": global_rc_l2_norm,
                    "rc_rg_l2_norm": rc_rg_l2_norm,
                    "rc_rg_mean_abs": float(torch.mean(torch.abs(rc_rg_delta)).item()),
                    "use_rc_regularization": getattr(run_cfg.federated, "use_rc_regularization", False),
                    "rc_lambda": getattr(run_cfg.federated, "rc_lambda", 0.0),
                })
        avg_fedavg_rc_rg_l2_norm = (
            sum(fedavg_rc_rg_l2_norms) / len(fedavg_rc_rg_l2_norms)
            if len(fedavg_rc_rg_l2_norms) > 0 else None
        )
        min_fedavg_rc_rg_l2_norm = (
            min(fedavg_rc_rg_l2_norms)
            if len(fedavg_rc_rg_l2_norms) > 0 else None
        )
        max_fedavg_rc_rg_l2_norm = (
            max(fedavg_rc_rg_l2_norms)
            if len(fedavg_rc_rg_l2_norms) > 0 else None
        )
        head_mask_jaccard_mean = safe_mean(
            row.get("mask_jaccard_with_prev_round") for row in round_head_rows
        )
        head_mask_hamming_change_rate_mean = safe_mean(
            row.get("mask_hamming_change_rate") for row in round_head_rows
        )
        head_selected_importance_mean_avg = safe_mean(
            row.get("selected_importance_mean") for row in round_head_rows
        )
        head_unselected_importance_mean_avg = safe_mean(
            row.get("unselected_importance_mean") for row in round_head_rows
        )
        head_importance_gap_mean_ratio_avg = safe_mean(
            row.get("importance_gap_mean_ratio") for row in round_head_rows
        )
        head_importance_gap_boundary_ratio_avg = safe_mean(
            row.get("importance_gap_boundary_ratio") for row in round_head_rows
        )
        head_importance_selected_share_avg = safe_mean(
            row.get("importance_selected_share") for row in round_head_rows
        )
        head_info = f" | HeadPersRatio={avg_personalized_head_ratio:.4f}" if use_head_this_round else ""
        head_diag_text = ""
        if use_head_this_round:
            head_diag_text = (
                f" | HeadJaccard: {format_optional_metric(head_mask_jaccard_mean)}"
                f" | HeadChange: {format_optional_metric(head_mask_hamming_change_rate_mean)}"
                f" | HeadGapMean: {format_optional_metric(head_importance_gap_mean_ratio_avg)}"
                f" | HeadSelectedShare: {format_optional_metric(head_importance_selected_share_avg)}"
            )
        h2a_alpha_mean = (
            sum(h2a_alpha_values) / len(h2a_alpha_values)
            if len(h2a_alpha_values) > 0 else None
        )
        h2a_alpha_min = min(h2a_alpha_values) if len(h2a_alpha_values) > 0 else None
        h2a_alpha_max = max(h2a_alpha_values) if len(h2a_alpha_values) > 0 else None
        h2a_distance_mean = (
            sum(h2a_distance_values) / len(h2a_distance_values)
            if len(h2a_distance_values) > 0 else None
        )
        h2a_meta_loss_mean = (
            sum(h2a_meta_losses) / len(h2a_meta_losses)
            if len(h2a_meta_losses) > 0 else None
        )
        h2a_info_text = (
            f" | H2AAlpha={h2a_alpha_mean:.4f}" if h2a_alpha_mean is not None else ""
        )

        row = {
            "round": rnd,
            "selected_clients": len(selected_clients),
            "avg_local_train_loss": avg_train_loss,
            "avg_local_val_loss": avg_val_loss,
            "avg_local_train_task_loss": avg_local_train_task_loss,
            "avg_local_train_rc_loss": avg_local_train_rc_loss,
            "avg_local_train_weighted_rc_loss": avg_local_train_weighted_rc_loss,
            "avg_local_train_total_loss": avg_local_train_total_loss,
            "avg_local_train_rc_to_task_ratio": avg_local_train_rc_to_task_ratio,
            "round_rc_to_task_ratio": round_rc_to_task_ratio,
            "global_rc_norm": global_rc_norm,
            "avg_client_rc_rg_l2_norm": avg_fedavg_rc_rg_l2_norm,
            "min_client_rc_rg_l2_norm": min_fedavg_rc_rg_l2_norm,
            "max_client_rc_rg_l2_norm": max_fedavg_rc_rg_l2_norm,
            "avg_personalized_head_ratio": avg_personalized_head_ratio,
            "avg_num_personalized_head_params": avg_num_personalized_head_params,
            "avg_num_head_params": avg_num_head_params,
            "avg_head_importance_mean": avg_head_importance_mean,
            "head_personalization_enabled_this_round": use_head_this_round,
            "head_personalization_warmup_rounds": head_warmup_rounds,
            "head_mask_update_interval": head_mask_update_interval,
            "head_mask_updated_this_round": update_head_mask_this_round,
            "head_param_prefixes": ",".join(getattr(run_cfg.federated, "head_param_prefixes", ["fc1", "fc2"])),
            "head_param_exact_names": ",".join(getattr(run_cfg.federated, "head_param_exact_names", [])),
            "head_personalization_tau": getattr(run_cfg.federated, "head_personalization_tau", 0.0),
            "use_head_importance_ema": getattr(run_cfg.federated, "use_head_importance_ema", False),
            "head_importance_ema_beta": (
                getattr(run_cfg.federated, "head_importance_ema_beta", None)
                if getattr(run_cfg.federated, "use_head_importance_ema", False) else None
            ),
        }
        if aggregation_method == "h2a":
            row.update({
                "h2a_alpha_mean": h2a_alpha_mean,
                "h2a_alpha_min": h2a_alpha_min,
                "h2a_alpha_max": h2a_alpha_max,
                "h2a_distance_mean": h2a_distance_mean,
                "h2a_meta_loss_mean": h2a_meta_loss_mean,
                "h2a_num_refs": max(h2a_ref_counts) if len(h2a_ref_counts) > 0 else None,
                "h2a_reference_mode": getattr(run_cfg.federated, "h2a_reference_mode", "adaptive"),
                "h2a_reference_summary": "; ".join(h2a_ref_summary_items),
            })
        if use_head_this_round:
            row.update({
                "head_mask_jaccard_mean": head_mask_jaccard_mean,
                "head_mask_hamming_change_rate_mean": head_mask_hamming_change_rate_mean,
                "head_selected_importance_mean_avg": head_selected_importance_mean_avg,
                "head_unselected_importance_mean_avg": head_unselected_importance_mean_avg,
                "head_importance_gap_mean_ratio_avg": head_importance_gap_mean_ratio_avg,
                "head_importance_gap_boundary_ratio_avg": head_importance_gap_boundary_ratio_avg,
                "head_importance_selected_share_avg": head_importance_selected_share_avg,
            })

        if rnd % run_cfg.federated.eval_every == 0:
            if aggregation_method == "h2a":
                client_val_summary_df, client_val_pred_map = summarize_h2a_clients(
                    server, clients, split_name="val", include_predictions=True
                )
            else:
                client_val_summary_df, client_val_pred_map = summarize_clients(
                    server, clients, split_name="val", include_predictions=True
                )
            regional_val_df, regional_val_metrics = evaluate_regional_from_predictions(
                client_val_pred_map, clients, run_cfg
            )
            avg_client_val_rmse = float(client_val_summary_df["RMSE"].mean())
            avg_client_val_mae = float(client_val_summary_df["MAE"].mean())
            avg_client_val_mse = float(client_val_summary_df["MSE"].mean())
            avg_client_val_r2 = float(client_val_summary_df["R2"].mean())

            row["regional_val_loss"] = float(client_val_summary_df["loss"].mean())
            row.update({
                "regional_val_MAE": regional_val_metrics["MAE"],
                "regional_val_MSE": regional_val_metrics["MSE"],
                "regional_val_RMSE": regional_val_metrics["RMSE"],
                "regional_val_MAPE_percent": regional_val_metrics["MAPE_percent"],
                "regional_val_R2": regional_val_metrics["R2"],
                "avg_client_val_MAE": avg_client_val_mae,
                "avg_client_val_MSE": avg_client_val_mse,
                "avg_client_val_RMSE": avg_client_val_rmse,
                "avg_client_val_R2": avg_client_val_r2,
                "regional_val_samples": len(regional_val_df),
            })

            regional_val_mae_curve.append(regional_val_metrics["MAE"])
            regional_val_rmse_curve.append(regional_val_metrics["RMSE"])
            regional_val_r2_curve.append(regional_val_metrics["R2"])
            avg_client_val_rmse_curve.append(avg_client_val_rmse)

            if avg_client_val_rmse < best_avg_client_val_rmse:
                best_avg_client_val_rmse = avg_client_val_rmse
                no_improve_rounds = 0
                avg_client_val_metrics = {
                    "MAE": avg_client_val_mae,
                    "MSE": avg_client_val_mse,
                    "RMSE": avg_client_val_rmse,
                    "R2": avg_client_val_r2,
                }
                if aggregation_method == "h2a":
                    checkpoint_payload = make_h2a_checkpoint(
                        server,
                        rnd,
                        best_avg_client_val_rmse,
                        regional_val_metrics,
                        avg_client_val_metrics,
                    )
                    torch.save(checkpoint_payload, best_model_path)
                    torch.save(checkpoint_payload, best_checkpoint_path)
                else:
                    torch.save(server.get_global_state(), best_model_path)
                    best_client_personalizations = {
                        client.client_name: client.export_personalization()
                        for client in clients
                    }
                    torch.save({
                        "model_state_dict": server.get_global_state(),
                        "global_rc": server.get_global_rc(),
                        "client_personalizations": best_client_personalizations,
                        "best_round": rnd,
                        "best_avg_client_val_rmse": best_avg_client_val_rmse,
                        "regional_val_metrics": regional_val_metrics,
                        "avg_client_val_metrics": avg_client_val_metrics,
                        "aggregation_method": "fedavg",
                        "use_rc_regularization": getattr(run_cfg.federated, "use_rc_regularization", False),
                        "rc_lambda": getattr(run_cfg.federated, "rc_lambda", 0.0),
                        "use_head_personalization": getattr(run_cfg.federated, "use_head_personalization", False),
                        "head_personalization_tau": getattr(run_cfg.federated, "head_personalization_tau", 0.0),
                        "head_param_prefixes": getattr(run_cfg.federated, "head_param_prefixes", ["fc1", "fc2"]),
                    }, best_checkpoint_path)
                best_model_saved = True
                best_checkpoint_saved = True
            else:
                no_improve_rounds += 1
                if early_stop_patience and no_improve_rounds >= early_stop_patience:
                    should_stop = True

            if use_rc:
                global_rc_norm_str = "None" if global_rc_norm is None else f"{global_rc_norm:.6f}"
                print(
                    f"Round [{rnd:03d}/{run_cfg.federated.rounds}] | "
                    f"AvgTaskLoss: {avg_local_train_task_loss:.6f} | "
                    f"AvgRCLoss: {avg_local_train_rc_loss:.6f} | "
                    f"AvgWeightedRCLoss: {avg_local_train_weighted_rc_loss:.6f} | "
                    f"RCratio={round_rc_to_task_ratio:.4f}{head_info}{head_diag_text} | "
                    f"GlobalRCNorm: {global_rc_norm_str} | "
                    f"AvgLocalValLoss: {avg_val_loss:.6f} | "
                    f"AvgClientValRMSE: {avg_client_val_rmse:.6f} | "
                    f"RegionalValRMSE: {regional_val_metrics['RMSE']:.6f}"
                )
            else:
                print(
                    f"Round [{rnd:03d}/{run_cfg.federated.rounds}] | "
                    f"AvgLocalTrainLoss: {avg_train_loss:.6f}{head_info}{head_diag_text}{h2a_info_text} | "
                    f"AvgLocalValLoss: {avg_val_loss:.6f} | "
                    f"AvgClientValRMSE: {avg_client_val_rmse:.6f} | "
                    f"RegionalValRMSE: {regional_val_metrics['RMSE']:.6f} | "
                    f"RegionalValR2: {regional_val_metrics['R2']:.6f}"
                )
        else:
            if use_rc:
                global_rc_norm_str = "None" if global_rc_norm is None else f"{global_rc_norm:.6f}"
                print(
                    f"Round [{rnd:03d}/{run_cfg.federated.rounds}] | "
                    f"AvgTaskLoss: {avg_local_train_task_loss:.6f} | "
                    f"AvgRCLoss: {avg_local_train_rc_loss:.6f} | "
                    f"AvgWeightedRCLoss: {avg_local_train_weighted_rc_loss:.6f} | "
                    f"RCratio={round_rc_to_task_ratio:.4f}{head_info}{head_diag_text} | "
                    f"GlobalRCNorm: {global_rc_norm_str} | "
                    f"AvgLocalValLoss: {avg_val_loss:.6f}"
                )
            else:
                print(
                    f"Round [{rnd:03d}/{run_cfg.federated.rounds}] | "
                    f"AvgLocalTrainLoss: {avg_train_loss:.6f}{head_info}{head_diag_text}{h2a_info_text} | "
                    f"AvgLocalValLoss: {avg_val_loss:.6f}"
                )

        if aggregation_method == "h2a":
            print(f"H2A aggregation refs [{rnd:03d}]: {'; '.join(h2a_ref_summary_items)}")

        round_logs.append(row)
        if aggregation_method == "h2a":
            h2a_round_summary_logs.append({
                "round": rnd,
                "h2a_alpha_mean": h2a_alpha_mean,
                "h2a_alpha_min": h2a_alpha_min,
                "h2a_alpha_max": h2a_alpha_max,
                "h2a_distance_mean": h2a_distance_mean,
                "h2a_meta_loss_mean": h2a_meta_loss_mean,
                "avg_local_train_loss": avg_train_loss,
                "avg_local_val_loss": avg_val_loss,
                "avg_client_val_RMSE": row.get("avg_client_val_RMSE"),
                "regional_val_RMSE": row.get("regional_val_RMSE"),
            })
            save_h2a_logs(
                run_cfg.federated.save_dir,
                h2a_alpha_logs,
                h2a_reference_logs,
                h2a_layer_weight_logs,
                h2a_round_summary_logs,
            )
            if rnd % run_cfg.federated.eval_every == 0:
                save_h2a_importance_matrix(
                    server,
                    run_cfg.federated.save_dir,
                    filename=f"h2a_importance_matrix_round_{rnd:03d}.csv",
                )
        if should_stop:
            print(
                f"Early stopping at round {rnd}: "
                f"no AvgClientValRMSE improvement for {early_stop_patience} evaluations."
            )
            break

    if aggregation_method == "h2a":
        final_payload = make_h2a_checkpoint(
            server,
            None,
            None if best_avg_client_val_rmse == float("inf") else best_avg_client_val_rmse,
            None,
            None,
        )
        torch.save(final_payload, final_model_path)
        if not best_model_saved:
            torch.save(final_payload, best_model_path)
        if not best_checkpoint_saved:
            torch.save(final_payload, best_checkpoint_path)
    else:
        torch.save(server.get_global_state(), final_model_path)

        if not best_model_saved:
            torch.save(server.get_global_state(), best_model_path)
            best_client_personalizations = {
                client.client_name: client.export_personalization()
                for client in clients
            }

        if not best_checkpoint_saved:
            if best_client_personalizations is None:
                best_client_personalizations = {
                    client.client_name: client.export_personalization()
                    for client in clients
                }
            torch.save({
                "model_state_dict": server.get_global_state(),
                "global_rc": server.get_global_rc(),
                "client_personalizations": best_client_personalizations,
                "best_round": None,
                "best_avg_client_val_rmse": None if best_avg_client_val_rmse == float("inf") else best_avg_client_val_rmse,
                "regional_val_metrics": None,
                "avg_client_val_metrics": None,
                "aggregation_method": "fedavg",
                "use_rc_regularization": getattr(run_cfg.federated, "use_rc_regularization", False),
                "rc_lambda": getattr(run_cfg.federated, "rc_lambda", 0.0),
                "use_head_personalization": getattr(run_cfg.federated, "use_head_personalization", False),
                "head_personalization_tau": getattr(run_cfg.federated, "head_personalization_tau", 0.0),
                "head_param_prefixes": getattr(run_cfg.federated, "head_param_prefixes", ["fc1", "fc2"]),
            }, best_checkpoint_path)

    if aggregation_method == "h2a":
        best_payload = torch.load(best_checkpoint_path, map_location=run_cfg.train.device)
        server.load_state_dict(best_payload["h2a_server_state"])
        client_test_summary_df, client_test_pred_map = summarize_h2a_clients(
            server, clients, split_name="test", include_predictions=True
        )
    else:
        server.set_global_state(torch.load(best_model_path, map_location=run_cfg.train.device))
        if getattr(run_cfg.federated, "use_head_personalization", False) and best_client_personalizations is not None:
            for client in clients:
                client.import_personalization(best_client_personalizations.get(client.client_name))

        client_test_summary_df, client_test_pred_map = summarize_clients(
            server, clients, split_name="test", include_predictions=True
        )
    regional_test_df, regional_test_metrics = evaluate_regional_from_predictions(
        client_test_pred_map, clients, run_cfg
    )

    print_metrics(regional_test_metrics, title=f"{run_label} Regional Test Metrics")

    pd.DataFrame(round_logs).to_csv(
        os.path.join(run_cfg.federated.save_dir, "federated_round_logs.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    if aggregation_method == "fedavg":
        pd.DataFrame(
            fedavg_rc_rg_client_logs,
            columns=[
                "round",
                "client_id",
                "client_name",
                "num_samples",
                "local_rc_norm",
                "global_rc_norm",
                "rc_rg_l2_norm",
                "rc_rg_mean_abs",
                "use_rc_regularization",
                "rc_lambda",
            ],
        ).to_csv(
            os.path.join(run_cfg.federated.save_dir, "fedavg_rc_rg_client_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if len(head_mask_round_logs) > 0:
        pd.DataFrame(head_mask_round_logs).to_csv(
            os.path.join(run_cfg.federated.save_dir, "head_mask_round_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if len(head_layer_mask_logs) > 0:
        pd.DataFrame(head_layer_mask_logs).to_csv(
            os.path.join(run_cfg.federated.save_dir, "head_layer_mask_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    head_param_frequency_rows = finalize_head_param_frequency_logs(head_param_frequency_cache)
    if len(head_param_frequency_rows) > 0:
        pd.DataFrame(head_param_frequency_rows).to_csv(
            os.path.join(run_cfg.federated.save_dir, "head_param_selection_frequency.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    save_regional_outputs(
        regional_test_df,
        regional_test_metrics,
        run_cfg.federated.save_dir,
        prefix="regional_test",
        plot_title=f"{run_label} Regional Test Prediction",
    )
    save_training_curves(
        run_cfg.federated.save_dir,
        regional_val_rmse_curve,
        regional_val_mae_curve,
        regional_val_r2_curve,
        avg_client_val_rmse_curve,
    )
    save_client_outputs(
        client_test_pred_map,
        client_test_summary_df,
        save_dir=run_cfg.federated.save_dir,
        split_name="test",
        plot_title_prefix=run_label,
    )
    if aggregation_method == "h2a":
        save_h2a_importance_matrix(server, run_cfg.federated.save_dir)

    print("\nPer-client test summary:")
    print(client_test_summary_df)
    print(f"\nBest model saved to: {best_model_path}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Results directory: {run_cfg.federated.save_dir}")

    return {
        "cfg": run_cfg,
        "clients": clients,
        "server": server,
        "best_model_path": best_model_path,
        "best_checkpoint_path": best_checkpoint_path,
        "final_model_path": final_model_path,
        "save_dir": run_cfg.federated.save_dir,
        "checkpoint_dir": checkpoint_dir,
        "regional_test_df": regional_test_df,
        "regional_test_metrics": regional_test_metrics,
        "client_test_summary_df": client_test_summary_df,
        "client_test_pred_map": client_test_pred_map,
    }


def build_direct_net_load_cfg(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.data.target_col = cfg.data.net_load_col
    cfg.feature.use_target_history = True
    cfg.feature.raw_feature_cols = []
    return cfg


def build_indirect_gc_cfg(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.data.target_col = "gc"
    cfg.feature = copy.deepcopy(getattr(base_cfg, "gc_feature", base_cfg.feature))
    return cfg


def build_indirect_gg_cfg(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.data.target_col = "gg"
    cfg.feature = copy.deepcopy(getattr(base_cfg, "gg_feature", base_cfg.feature))
    return cfg


def configure_fedavg_baseline_gg_clients_1_8(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.data.client_files = [
        os.path.join(PROJECT_ROOT, "per_client_merged", f"client_{client_id}_load_weather_30min.csv")
        for client_id in range(1, 9)
    ]
    cfg.experiment.task_type = "single_target"
    cfg.data.target_col = "gg"
    cfg.feature = copy.deepcopy(getattr(cfg, "gg_feature", cfg.feature))

    cfg.federated.aggregation_method = "fedavg"
    cfg.federated.rounds = 20
    cfg.federated.local_epochs = 1
    cfg.federated.client_fraction = 1.0
    cfg.federated.eval_every = 1
    cfg.federated.early_stop_patience = 0
    cfg.federated.use_rc_regularization = False
    cfg.federated.rc_lambda = 0.0
    cfg.federated.use_head_personalization = False
    cfg.federated.checkpoint_dir = None
    cfg.federated.save_dir = os.path.join(PROJECT_ROOT, "runs", "fedavg_baseline_gg_clients_1_8")
    cfg.federated.best_model_name = "best_fedavg_gg_clients_1_8_model.pth"
    cfg.federated.best_checkpoint_name = "best_fedavg_gg_clients_1_8_checkpoint.pth"
    cfg.federated.final_model_name = "final_fedavg_gg_clients_1_8_model.pth"
    return cfg


def save_combined_net_load_outputs(net_load_client_pred_map, net_load_summary_df, regional_net_load_df, regional_net_load_metrics, save_dir: str, title_prefix: str):
    ensure_dir(save_dir)
    save_regional_outputs(
        regional_net_load_df,
        regional_net_load_metrics,
        save_dir=save_dir,
        prefix="regional_test",
        plot_title=f"{title_prefix} Regional Net Load Test Prediction",
    )
    save_client_outputs(
        net_load_client_pred_map,
        net_load_summary_df,
        save_dir=save_dir,
        split_name="test",
        plot_title_prefix=title_prefix,
    )


def run_direct_net_load(base_cfg):
    run_cfg = build_direct_net_load_cfg(base_cfg)
    save_dir = os.path.join(base_cfg.federated.save_dir, "direct_net_load")
    checkpoint_root = getattr(base_cfg.federated, "checkpoint_dir", None)
    if checkpoint_root:
        run_cfg.federated.checkpoint_dir = os.path.join(checkpoint_root, "direct_net_load")
    return train_federated_model(run_cfg, save_dir=save_dir, run_label="Federated Direct Net Load")


def run_indirect_net_load(base_cfg):
    indirect_root = os.path.join(base_cfg.federated.save_dir, "indirect_net_load")
    checkpoint_root = getattr(base_cfg.federated, "checkpoint_dir", None)
    indirect_checkpoint_root = (
        os.path.join(checkpoint_root, "indirect_net_load")
        if checkpoint_root else None
    )
    ensure_dir(indirect_root)
    if indirect_checkpoint_root:
        ensure_dir(indirect_checkpoint_root)

    gc_cfg = build_indirect_gc_cfg(base_cfg)
    if indirect_checkpoint_root:
        gc_cfg.federated.checkpoint_dir = os.path.join(indirect_checkpoint_root, "gc_model")
    gc_result = train_federated_model(
        gc_cfg,
        save_dir=os.path.join(indirect_root, "gc_model"),
        run_label="Federated GC Model",
    )
    gg_cfg = build_indirect_gg_cfg(base_cfg)
    if indirect_checkpoint_root:
        gg_cfg.federated.checkpoint_dir = os.path.join(indirect_checkpoint_root, "gg_model")
    gg_result = train_federated_model(
        gg_cfg,
        save_dir=os.path.join(indirect_root, "gg_model"),
        run_label="Federated GG Model",
    )

    net_load_client_pred_map = {}
    summary_rows = []
    horizon = base_cfg.data.horizon

    for client_name, gc_pred_df in gc_result["client_test_pred_map"].items():
        gg_pred_df = gg_result["client_test_pred_map"][client_name]
        net_load_pred_df = combine_prediction_frames(gc_pred_df, gg_pred_df, horizon=horizon, op="subtract")
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
        gc_result["regional_test_df"],
        gg_result["regional_test_df"],
        horizon=horizon,
        op="subtract",
    )
    regional_net_load_metrics = calc_prediction_metrics(regional_net_load_df, horizon)

    save_combined_net_load_outputs(
        net_load_client_pred_map=net_load_client_pred_map,
        net_load_summary_df=net_load_summary_df,
        regional_net_load_df=regional_net_load_df,
        regional_net_load_metrics=regional_net_load_metrics,
        save_dir=indirect_root,
        title_prefix="Federated Indirect Net Load",
    )

    compare_df = pd.DataFrame([
        {"component": "gc_model_regional", **gc_result["regional_test_metrics"]},
        {"component": "gg_model_regional", **gg_result["regional_test_metrics"]},
        {"component": "indirect_net_load_regional", **regional_net_load_metrics},
    ])
    compare_df.to_csv(
        os.path.join(indirect_root, "indirect_net_load_component_compare.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print_metrics(regional_net_load_metrics, title="Federated Indirect Net Load Regional Test Metrics")
    print("\nIndirect net load per-client summary:")
    print(net_load_summary_df)
    print(f"\nIndirect net load results directory: {indirect_root}")

    return {
        "gc_result": gc_result,
        "gg_result": gg_result,
        "regional_test_df": regional_net_load_df,
        "regional_test_metrics": regional_net_load_metrics,
        "client_test_summary_df": net_load_summary_df,
        "client_test_pred_map": net_load_client_pred_map,
        "save_dir": indirect_root,
    }


def main():
    cfg = copy.deepcopy(CFG)
    if cfg.data.target_col == "gc":
        cfg.feature = copy.deepcopy(getattr(cfg, "gc_feature", cfg.feature))
    elif cfg.data.target_col == "gg":
        cfg.feature = copy.deepcopy(getattr(cfg, "gg_feature", cfg.feature))

    if cfg.experiment.task_type == "single_target":
        ema_label = (
            f" EMA={cfg.federated.head_importance_ema_beta:g}"
            if getattr(cfg.federated, "use_head_importance_ema", False) else ""
        )
        train_federated_model(
            cfg,
            save_dir=cfg.federated.save_dir,
            run_label=(
                f"{cfg.federated.aggregation_method.upper()} {cfg.data.target_col.upper()} "
                f"+ RC + fc1 Head tau={cfg.federated.head_personalization_tau:g} "
                f"warmup={cfg.federated.head_personalization_warmup_rounds}{ema_label}"
            ),
        )
        return

    if cfg.experiment.task_type != "net_load":
        raise ValueError(
            f"Unsupported experiment.task_type={cfg.experiment.task_type}. "
            "Use 'single_target' or 'net_load'."
        )

    print("Net load is defined as gc - gg.")
    method = cfg.experiment.net_load_method.lower()
    if method == "direct":
        run_direct_net_load(cfg)
    elif method == "indirect":
        run_indirect_net_load(cfg)
    else:
        raise ValueError(
            f"Unsupported experiment.net_load_method={cfg.experiment.net_load_method}. "
            "Use 'direct' or 'indirect'."
        )


if __name__ == "__main__":
    main()
