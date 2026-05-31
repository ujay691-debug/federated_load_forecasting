import copy
import os
import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
from client import get_optimizer
from config import CFG, PROJECT_ROOT
from federated_main import build_clients
from models.cnn_lstm import CNNLSTMModel
from utils.data_utils import add_timestamp_occurrence_key, ensure_dir, save_config, set_seed
from utils.metrics import calc_metrics, plot_true_pred, print_metrics, save_metrics_csv


def build_model(input_dim, output_dim, cfg):
    return CNNLSTMModel(
        input_dim=input_dim,
        output_dim=output_dim,
        cfg=cfg.model,
    ).to(torch.device(cfg.train.device))


def clone_state_dict(state_dict):
    return {
        key: value.clone().detach() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }


def compute_prediction_loss(pred_student, y_true, loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return torch.mean((pred_student - y_true) ** 2)
    if loss_name == "mae":
        return torch.mean(torch.abs(pred_student - y_true))
    raise ValueError(f"Unsupported loss_name={loss_name}. Use 'mse' or 'mae'.")


def compute_advantage_weighted_transfer_loss(
    pred_student,
    pred_teacher,
    y_true,
    rho,
    r_max,
    a_max,
    eps,
):
    if r_max <= 0:
        raise ValueError("r_max must be positive.")

    pred_teacher_detached = pred_teacher.detach()
    e_student = (pred_student - y_true) ** 2
    e_teacher = (pred_teacher_detached - y_true) ** 2

    r = (e_student - e_teacher) / (e_student + eps)
    a = torch.clamp((r - rho) / r_max, min=0.0, max=a_max)

    numerator = torch.sum(a * (pred_student - pred_teacher_detached) ** 2)
    denominator = torch.sum(a) + eps
    transfer_loss = numerator / denominator
    active_ratio = (a.detach() > 0).float().mean()
    return transfer_loss, active_ratio


def compute_contrastive_regression_transfer_loss(
    pred_student,
    pred_teacher,
    y_true,
    rho,
    r_max,
    a_max,
    repulsion_margin,
    eps,
):
    if r_max <= 0:
        raise ValueError("r_max must be positive.")

    pred_teacher_detached = pred_teacher.detach()
    e_student = (pred_student - y_true) ** 2
    e_teacher = (pred_teacher_detached - y_true) ** 2

    teacher_better_mask = e_teacher < e_student
    teacher_worse_mask = e_teacher > e_student

    r_pos = (e_student - e_teacher) / (e_student + eps)
    a_pos = torch.clamp((r_pos - rho) / r_max, min=0.0, max=a_max)
    positive_loss = torch.sum(a_pos * (pred_student - pred_teacher_detached) ** 2) / (torch.sum(a_pos) + eps)

    r_neg = (e_teacher - e_student) / (e_student + eps)
    a_neg = torch.clamp((r_neg - rho) / r_max, min=0.0, max=a_max)
    distance = torch.abs(pred_student - pred_teacher_detached)
    repulsion_loss = torch.sum(a_neg * torch.relu(repulsion_margin - distance) ** 2) / (torch.sum(a_neg) + eps)

    stats_dict = {
        "positive_active_ratio": float((a_pos.detach() > 0).float().mean().item()),
        "repulsion_active_ratio": float((a_neg.detach() > 0).float().mean().item()),
        "teacher_better_ratio": float(teacher_better_mask.detach().float().mean().item()),
        "teacher_worse_ratio": float(teacher_worse_mask.detach().float().mean().item()),
        "positive_weight_mean": float(a_pos.detach().mean().item()),
        "negative_weight_mean": float(a_neg.detach().mean().item()),
    }
    return positive_loss, repulsion_loss, stats_dict


def summarize_scalar_values(values):
    if len(values) == 0:
        return {
            "avg": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }

    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "avg": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def build_alpha_grid(alpha_grid_step: float):
    if alpha_grid_step <= 0 or alpha_grid_step > 1:
        raise ValueError("alpha_grid_step must be in (0, 1].")

    alpha_grid = []
    alpha = 0.0
    while alpha < 1.0 - 1e-12:
        alpha_grid.append(round(alpha, 10))
        alpha += alpha_grid_step
    if not alpha_grid or alpha_grid[-1] != 1.0:
        alpha_grid.append(1.0)
    return alpha_grid


def weighted_merge_state_dicts(state_a, state_b, weight_a, weight_b):
    denom = float(weight_a) + float(weight_b)
    if denom <= 0:
        raise ValueError("The sum of merge weights must be positive.")

    merged = {}
    for key, tensor_a in state_a.items():
        tensor_b = state_b[key]
        if torch.is_tensor(tensor_a) and torch.is_floating_point(tensor_a):
            tensor_b = tensor_b.to(device=tensor_a.device, dtype=tensor_a.dtype)
            merged[key] = (
                (float(weight_a) * tensor_a + float(weight_b) * tensor_b) / denom
            ).clone().detach()
        elif torch.is_tensor(tensor_a):
            merged[key] = tensor_a.clone().detach()
        else:
            merged[key] = copy.deepcopy(tensor_a)
    return merged


def clean_test_prediction_df(pred_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = pred_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    value_cols = []
    for step in range(horizon):
        value_cols.extend([f"y_true_step_{step + 1}", f"y_pred_step_{step + 1}"])

    # Remove DST-related empty timestamps/values and repeated wall-clock timestamps
    # before saving test predictions or computing final test metrics.
    out = out.dropna(subset=["timestamp"] + value_cols)
    out = out.drop_duplicates(subset=["timestamp"], keep="first")
    return out.sort_values("timestamp").reset_index(drop=True)


def calc_prediction_metrics(pred_df: pd.DataFrame, horizon: int) -> dict:
    true_cols = [f"y_true_step_{i + 1}" for i in range(horizon)]
    pred_cols = [f"y_pred_step_{i + 1}" for i in range(horizon)]
    return calc_metrics(
        pred_df[true_cols].values.reshape(-1),
        pred_df[pred_cols].values.reshape(-1),
    )


def build_prediction_error_df(pred_df: pd.DataFrame, horizon: int, eps: float = 1e-8) -> pd.DataFrame:
    error_df = pd.DataFrame({"timestamp": pred_df["timestamp"].copy()})
    for step in range(horizon):
        true_col = f"y_true_step_{step + 1}"
        pred_col = f"y_pred_step_{step + 1}"
        error = pred_df[pred_col] - pred_df[true_col]
        abs_error = error.abs()
        error_df[true_col] = pred_df[true_col]
        error_df[pred_col] = pred_df[pred_col]
        error_df[f"error_step_{step + 1}"] = error
        error_df[f"abs_error_step_{step + 1}"] = abs_error
        error_df[f"squared_error_step_{step + 1}"] = error ** 2
        error_df[f"abs_percent_error_step_{step + 1}"] = (
            abs_error / pred_df[true_col].abs().clip(lower=eps) * 100.0
        )
    return error_df


def combine_prediction_frames(lhs_df: pd.DataFrame, rhs_df: pd.DataFrame, horizon: int, op: str = "subtract") -> pd.DataFrame:
    left = add_timestamp_occurrence_key(clean_test_prediction_df(lhs_df, horizon), "timestamp")
    right = add_timestamp_occurrence_key(clean_test_prediction_df(rhs_df, horizon), "timestamp")
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

    return clean_test_prediction_df(out, horizon)


def train_student_without_teacher(
    student_model,
    train_loader,
    cfg,
):
    dcfg = cfg.decentralized_gcml
    device = torch.device(cfg.train.device)

    student_model.to(device)
    student_model.train()

    for param in student_model.parameters():
        param.requires_grad = True

    optimizer = get_optimizer(cfg.train.optimizer_name, student_model, cfg.train.lr)

    pred_loss_sum = 0.0
    total_loss_sum = 0.0
    sample_count = 0
    num_batches = 0

    for _ in range(dcfg.transfer_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.size(0)

            optimizer.zero_grad()
            pred_student = student_model(batch_x)
            pred_loss = compute_prediction_loss(pred_student, batch_y, cfg.train.loss_name)
            total_loss = pred_loss

            total_loss.backward()
            optimizer.step()

            pred_loss_sum += float(pred_loss.item()) * batch_size
            total_loss_sum += float(total_loss.item()) * batch_size
            sample_count += batch_size
            num_batches += 1

    denom = max(sample_count, 1)
    log_dict = {
        "pred_loss_avg": pred_loss_sum / denom,
        "total_loss_avg": total_loss_sum / denom,
        "num_batches": num_batches,
    }
    return clone_state_dict(student_model.state_dict()), log_dict


def train_student_with_teacher(
    student_model,
    teacher_model,
    train_loader,
    cfg,
):
    dcfg = cfg.decentralized_gcml
    device = torch.device(cfg.train.device)

    student_model.to(device)
    teacher_model.to(device)
    student_model.train()
    teacher_model.eval()

    for param in student_model.parameters():
        param.requires_grad = True
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = get_optimizer(cfg.train.optimizer_name, student_model, cfg.train.lr)

    pred_loss_sum = 0.0
    positive_loss_sum = 0.0
    repulsion_loss_sum = 0.0
    total_loss_sum = 0.0
    sample_count = 0
    teacher_better_ratio_values = []
    teacher_worse_ratio_values = []
    positive_active_ratio_values = []
    repulsion_active_ratio_values = []
    positive_weight_mean_values = []
    negative_weight_mean_values = []

    for _ in range(dcfg.transfer_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.size(0)

            optimizer.zero_grad()
            pred_student = student_model(batch_x)
            with torch.no_grad():
                pred_teacher = teacher_model(batch_x)

            pred_loss = compute_prediction_loss(pred_student, batch_y, cfg.train.loss_name)
            positive_loss, repulsion_loss, stats_dict = compute_contrastive_regression_transfer_loss(
                pred_student=pred_student,
                pred_teacher=pred_teacher,
                y_true=batch_y,
                rho=dcfg.rho,
                r_max=dcfg.r_max,
                a_max=dcfg.a_max,
                repulsion_margin=dcfg.repulsion_margin,
                eps=dcfg.eps,
            )
            total_loss = (
                pred_loss
                + dcfg.lambda_transfer * positive_loss
                + dcfg.repulsion_lambda * repulsion_loss
            )

            total_loss.backward()
            optimizer.step()

            pred_loss_sum += float(pred_loss.item()) * batch_size
            positive_loss_sum += float(positive_loss.item()) * batch_size
            repulsion_loss_sum += float(repulsion_loss.item()) * batch_size
            total_loss_sum += float(total_loss.item()) * batch_size
            sample_count += batch_size

            teacher_better_ratio_values.append(stats_dict["teacher_better_ratio"])
            teacher_worse_ratio_values.append(stats_dict["teacher_worse_ratio"])
            positive_active_ratio_values.append(stats_dict["positive_active_ratio"])
            repulsion_active_ratio_values.append(stats_dict["repulsion_active_ratio"])
            positive_weight_mean_values.append(stats_dict["positive_weight_mean"])
            negative_weight_mean_values.append(stats_dict["negative_weight_mean"])

    denom = max(sample_count, 1)
    teacher_better_ratio_stats = summarize_scalar_values(teacher_better_ratio_values)
    teacher_worse_ratio_stats = summarize_scalar_values(teacher_worse_ratio_values)
    positive_active_ratio_stats = summarize_scalar_values(positive_active_ratio_values)
    repulsion_active_ratio_stats = summarize_scalar_values(repulsion_active_ratio_values)
    positive_weight_mean_stats = summarize_scalar_values(positive_weight_mean_values)
    negative_weight_mean_stats = summarize_scalar_values(negative_weight_mean_values)
    log_dict = {
        "pred_loss_avg": pred_loss_sum / denom,
        "positive_loss_avg": positive_loss_sum / denom,
        "repulsion_loss_avg": repulsion_loss_sum / denom,
        "total_loss_avg": total_loss_sum / denom,
        "teacher_better_ratio_avg": teacher_better_ratio_stats["avg"],
        "teacher_better_ratio_min": teacher_better_ratio_stats["min"],
        "teacher_better_ratio_max": teacher_better_ratio_stats["max"],
        "teacher_better_ratio_std": teacher_better_ratio_stats["std"],
        "teacher_worse_ratio_avg": teacher_worse_ratio_stats["avg"],
        "teacher_worse_ratio_min": teacher_worse_ratio_stats["min"],
        "teacher_worse_ratio_max": teacher_worse_ratio_stats["max"],
        "teacher_worse_ratio_std": teacher_worse_ratio_stats["std"],
        "positive_active_ratio_avg": positive_active_ratio_stats["avg"],
        "positive_active_ratio_min": positive_active_ratio_stats["min"],
        "positive_active_ratio_max": positive_active_ratio_stats["max"],
        "positive_active_ratio_std": positive_active_ratio_stats["std"],
        "repulsion_active_ratio_avg": repulsion_active_ratio_stats["avg"],
        "repulsion_active_ratio_min": repulsion_active_ratio_stats["min"],
        "repulsion_active_ratio_max": repulsion_active_ratio_stats["max"],
        "repulsion_active_ratio_std": repulsion_active_ratio_stats["std"],
        "positive_weight_mean_avg": positive_weight_mean_stats["avg"],
        "negative_weight_mean_avg": negative_weight_mean_stats["avg"],
        "num_batches": len(teacher_better_ratio_values),
    }
    return clone_state_dict(student_model.state_dict()), log_dict


def run_receiver_gcml_update(
    receiver_client,
    receiver_loc_state,
    sender_loc_state,
    cfg,
    sender_client=None,
):
    dcfg = cfg.decentralized_gcml
    receiver_id = receiver_client.client_id
    sender_id = sender_client.client_id if sender_client is not None else (1 if receiver_id == 2 else 2)

    input_dim = len(receiver_client.feature_cols)
    output_dim = cfg.data.horizon
    base_state = clone_state_dict(receiver_loc_state)

    local_continue_model = build_model(input_dim=input_dim, output_dim=output_dim, cfg=cfg)
    local_continue_model.load_state_dict(copy.deepcopy(base_state))
    local_continue_state, local_continue_train_log = train_student_without_teacher(
        student_model=local_continue_model,
        train_loader=receiver_client.data["train_loader"],
        cfg=cfg,
    )

    teacher_guided_model = build_model(input_dim=input_dim, output_dim=output_dim, cfg=cfg)
    sender_teacher_model = build_model(input_dim=input_dim, output_dim=output_dim, cfg=cfg)
    teacher_guided_model.load_state_dict(copy.deepcopy(base_state))
    sender_teacher_model.load_state_dict(copy.deepcopy(sender_loc_state))
    teacher_guided_state, teacher_guided_train_log = train_student_with_teacher(
        student_model=teacher_guided_model,
        teacher_model=sender_teacher_model,
        train_loader=receiver_client.data["train_loader"],
        cfg=cfg,
    )

    _, _, val_loss_receiver_loc = receiver_client.evaluate_split(base_state, split_name="val")
    _, _, val_loss_local_continue = receiver_client.evaluate_split(local_continue_state, split_name="val")
    _, _, val_loss_teacher_guided = receiver_client.evaluate_split(teacher_guided_state, split_name="val")

    best_alpha = None
    same_origin_merge_state = None
    val_loss_same_origin_merge = float("inf")
    for alpha in build_alpha_grid(dcfg.alpha_grid_step):
        merged_state = weighted_merge_state_dicts(
            local_continue_state,
            teacher_guided_state,
            weight_a=1.0 - alpha,
            weight_b=alpha,
        )
        _, _, alpha_val_loss = receiver_client.evaluate_split(merged_state, split_name="val")
        alpha_val_loss = float(alpha_val_loss)
        if alpha_val_loss < val_loss_same_origin_merge:
            best_alpha = float(alpha)
            val_loss_same_origin_merge = alpha_val_loss
            same_origin_merge_state = clone_state_dict(merged_state)

    if same_origin_merge_state is None:
        raise RuntimeError("Alpha grid search did not produce a same-origin merge state.")

    candidate_losses = {
        "receiver_loc": float(val_loss_receiver_loc),
        "local_continue": float(val_loss_local_continue),
        "teacher_guided": float(val_loss_teacher_guided),
        "same_origin_merge": float(val_loss_same_origin_merge),
    }
    candidate_states = {
        "receiver_loc": base_state,
        "local_continue": local_continue_state,
        "teacher_guided": teacher_guided_state,
        "same_origin_merge": same_origin_merge_state,
    }

    if getattr(dcfg, "enable_merge_rollback", True):
        selected_model = min(candidate_losses, key=candidate_losses.get)
    else:
        selected_model = "same_origin_merge"
    final_state = clone_state_dict(candidate_states[selected_model])

    rollback_to_local = selected_model == "receiver_loc"
    accept_local_continue = selected_model == "local_continue"
    accept_teacher_guided = selected_model == "teacher_guided"
    accept_same_origin_merge = selected_model == "same_origin_merge"

    log_dict = {
        "receiver_id": receiver_id,
        "sender_id": sender_id,
        "selected_sender_id": sender_id,
        "selected_model": selected_model,
        "selected_val_loss": candidate_losses[selected_model],
        "val_loss_receiver_loc": candidate_losses["receiver_loc"],
        "val_loss_local_continue": candidate_losses["local_continue"],
        "val_loss_teacher_guided": candidate_losses["teacher_guided"],
        "val_loss_same_origin_merge": candidate_losses["same_origin_merge"],
        "best_alpha": float(best_alpha),
        "rollback_to_local": bool(rollback_to_local),
        "accept_local_continue": bool(accept_local_continue),
        "accept_teacher_guided": bool(accept_teacher_guided),
        "accept_same_origin_merge": bool(accept_same_origin_merge),
        "delta_local_continue_vs_loc": candidate_losses["local_continue"] - candidate_losses["receiver_loc"],
        "delta_teacher_guided_vs_loc": candidate_losses["teacher_guided"] - candidate_losses["receiver_loc"],
        "delta_teacher_guided_vs_local_continue": (
            candidate_losses["teacher_guided"] - candidate_losses["local_continue"]
        ),
        "delta_same_origin_merge_vs_loc": candidate_losses["same_origin_merge"] - candidate_losses["receiver_loc"],
        "delta_same_origin_merge_vs_local_continue": (
            candidate_losses["same_origin_merge"] - candidate_losses["local_continue"]
        ),
        "delta_same_origin_merge_vs_teacher_guided": (
            candidate_losses["same_origin_merge"] - candidate_losses["teacher_guided"]
        ),
        "local_continue_pred_loss_avg": local_continue_train_log["pred_loss_avg"],
        "local_continue_total_loss_avg": local_continue_train_log["total_loss_avg"],
        "local_continue_num_batches": local_continue_train_log["num_batches"],
        "teacher_guided_pred_loss_avg": teacher_guided_train_log["pred_loss_avg"],
        "teacher_guided_positive_loss_avg": teacher_guided_train_log["positive_loss_avg"],
        "teacher_guided_repulsion_loss_avg": teacher_guided_train_log["repulsion_loss_avg"],
        "teacher_guided_total_loss_avg": teacher_guided_train_log["total_loss_avg"],
        "teacher_guided_teacher_better_ratio_avg": teacher_guided_train_log["teacher_better_ratio_avg"],
        "teacher_guided_teacher_better_ratio_min": teacher_guided_train_log["teacher_better_ratio_min"],
        "teacher_guided_teacher_better_ratio_max": teacher_guided_train_log["teacher_better_ratio_max"],
        "teacher_guided_teacher_better_ratio_std": teacher_guided_train_log["teacher_better_ratio_std"],
        "teacher_guided_teacher_worse_ratio_avg": teacher_guided_train_log["teacher_worse_ratio_avg"],
        "teacher_guided_teacher_worse_ratio_min": teacher_guided_train_log["teacher_worse_ratio_min"],
        "teacher_guided_teacher_worse_ratio_max": teacher_guided_train_log["teacher_worse_ratio_max"],
        "teacher_guided_teacher_worse_ratio_std": teacher_guided_train_log["teacher_worse_ratio_std"],
        "teacher_guided_positive_active_ratio_avg": teacher_guided_train_log["positive_active_ratio_avg"],
        "teacher_guided_positive_active_ratio_min": teacher_guided_train_log["positive_active_ratio_min"],
        "teacher_guided_positive_active_ratio_max": teacher_guided_train_log["positive_active_ratio_max"],
        "teacher_guided_positive_active_ratio_std": teacher_guided_train_log["positive_active_ratio_std"],
        "teacher_guided_repulsion_active_ratio_avg": teacher_guided_train_log["repulsion_active_ratio_avg"],
        "teacher_guided_repulsion_active_ratio_min": teacher_guided_train_log["repulsion_active_ratio_min"],
        "teacher_guided_repulsion_active_ratio_max": teacher_guided_train_log["repulsion_active_ratio_max"],
        "teacher_guided_repulsion_active_ratio_std": teacher_guided_train_log["repulsion_active_ratio_std"],
        "teacher_guided_positive_weight_mean_avg": teacher_guided_train_log["positive_weight_mean_avg"],
        "teacher_guided_negative_weight_mean_avg": teacher_guided_train_log["negative_weight_mean_avg"],
        "teacher_guided_num_batches": teacher_guided_train_log["num_batches"],
    }
    return final_state, log_dict


def save_personalized_test_outputs(clients, client_states, save_dir: str, cfg):
    summary_rows = []
    client_pred_map = {}
    for client, state in zip(clients, client_states):
        pred_df, _, loss = client.evaluate_split(state, split_name="test")
        pred_df = clean_test_prediction_df(pred_df, cfg.data.horizon)
        metrics = calc_prediction_metrics(pred_df, cfg.data.horizon)
        client_id = client.client_id

        pred_df.to_csv(
            os.path.join(save_dir, f"client_{client_id}_test_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        error_df = build_prediction_error_df(pred_df, cfg.data.horizon, eps=cfg.decentralized_gcml.eps)
        error_df.to_csv(
            os.path.join(save_dir, f"client_{client_id}_test_prediction_errors.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        save_metrics_csv(metrics, os.path.join(save_dir, f"client_{client_id}_test_metrics.csv"))
        plot_true_pred(
            pred_df["y_true_step_1"].values,
            pred_df["y_pred_step_1"].values,
            save_path=os.path.join(save_dir, f"client_{client_id}_test_prediction.png"),
            title=f"Decentralized GCML Client {client_id} Test Prediction",
            show_n=300,
        )

        row = {
            "client_id": client_id,
            "client_name": client.client_name,
            "loss": float(loss),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "R2": metrics["R2"],
        }
        summary_rows.append(row)
        client_pred_map[client.client_name] = pred_df
        print_metrics(metrics, title=f"Decentralized GCML Client {client_id} Test Metrics")

    summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    summary_df.to_csv(
        os.path.join(save_dir, "all_clients_test_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return summary_df, client_pred_map


def save_best_if_needed(client_id, client_state, val_loss, best_val_losses: Dict[int, float], cfg):
    if val_loss >= best_val_losses[client_id]:
        return

    best_val_losses[client_id] = float(val_loss)
    if client_id == 1:
        model_name = cfg.decentralized_gcml.best_model_name_client1
    elif client_id == 2:
        model_name = cfg.decentralized_gcml.best_model_name_client2
    else:
        model_name = f"best_client_{client_id}_model.pth"

    torch.save(
        clone_state_dict(client_state),
        os.path.join(cfg.decentralized_gcml.save_dir, model_name),
    )


def print_round_log(rnd, total_rounds, direction, log_dict):
    print(f"Round [{rnd:03d}/{total_rounds}]")
    print(f"direction: {direction}")
    print(f"selected_model: {log_dict['selected_model']}")
    print(f"rollback_to_local: {log_dict['rollback_to_local']}")
    print(f"accept_local_continue: {log_dict['accept_local_continue']}")
    print(f"accept_teacher_guided: {log_dict['accept_teacher_guided']}")
    print(f"accept_same_origin_merge: {log_dict['accept_same_origin_merge']}")
    print(f"receiver val local loss: {log_dict['val_loss_receiver_loc']:.6f}")
    print(f"local continue val loss: {log_dict['val_loss_local_continue']:.6f}")
    print(f"teacher guided val loss: {log_dict['val_loss_teacher_guided']:.6f}")
    print(f"same origin merge val loss: {log_dict['val_loss_same_origin_merge']:.6f}")
    print(f"selected val loss: {log_dict['selected_val_loss']:.6f}")
    print(f"best alpha: {log_dict['best_alpha']:.2f}")
    print(f"delta local continue vs loc: {log_dict['delta_local_continue_vs_loc']:.6f}")
    print(f"delta teacher guided vs local continue: {log_dict['delta_teacher_guided_vs_local_continue']:.6f}")
    print(f"delta teacher guided vs loc: {log_dict['delta_teacher_guided_vs_loc']:.6f}")
    print(f"delta same origin merge vs teacher guided: {log_dict['delta_same_origin_merge_vs_teacher_guided']:.6f}")
    print(f"teacher better ratio avg: {log_dict['teacher_guided_teacher_better_ratio_avg']:.6f}")
    print(f"teacher worse ratio avg: {log_dict['teacher_guided_teacher_worse_ratio_avg']:.6f}")
    print(f"positive active ratio avg: {log_dict['teacher_guided_positive_active_ratio_avg']:.6f}")
    print(f"repulsion active ratio avg: {log_dict['teacher_guided_repulsion_active_ratio_avg']:.6f}")
    print(f"positive loss avg: {log_dict['teacher_guided_positive_loss_avg']:.6f}")
    print(f"repulsion loss avg: {log_dict['teacher_guided_repulsion_loss_avg']:.6f}")


ROUND_LOG_CORE_COLUMNS = [
    "round",
    "global_round",
    "pair_index",
    "pair_id",
    "direction",
    "receiver_id",
    "sender_id",
    "selected_sender_id",
    "selected_model",
    "receiver_local_update_executed",
    "receiver_local_train_loss",
    "receiver_local_val_loss",
    "sender_local_train_loss",
    "sender_local_val_loss",
    "val_loss_receiver_loc",
    "val_loss_local_continue",
    "val_loss_teacher_guided",
    "val_loss_same_origin_merge",
    "selected_val_loss",
    "best_alpha",
    "delta_local_continue_vs_loc",
    "delta_teacher_guided_vs_loc",
    "delta_teacher_guided_vs_local_continue",
    "delta_same_origin_merge_vs_loc",
    "delta_same_origin_merge_vs_local_continue",
    "delta_same_origin_merge_vs_teacher_guided",
    "rollback_to_local",
    "accept_local_continue",
    "accept_teacher_guided",
    "accept_same_origin_merge",
    "teacher_guided_teacher_better_ratio_avg",
    "teacher_guided_teacher_worse_ratio_avg",
    "teacher_guided_positive_active_ratio_avg",
    "teacher_guided_repulsion_active_ratio_avg",
    "teacher_guided_positive_loss_avg",
    "teacher_guided_repulsion_loss_avg",
    "teacher_guided_total_loss_avg",
]


def save_round_logs_csv(round_logs, save_path: str) -> pd.DataFrame:
    round_logs_df = pd.DataFrame(round_logs)
    for column in ROUND_LOG_CORE_COLUMNS:
        if column not in round_logs_df.columns:
            round_logs_df[column] = pd.NA

    ordered_columns = ROUND_LOG_CORE_COLUMNS + [
        column for column in round_logs_df.columns if column not in ROUND_LOG_CORE_COLUMNS
    ]
    round_logs_df = round_logs_df[ordered_columns]
    round_logs_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return round_logs_df


def _safe_numeric_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def save_sender_effect_summary(round_logs, save_dir: str) -> pd.DataFrame:
    round_logs_df = pd.DataFrame(round_logs)
    summary_path = os.path.join(save_dir, "sender_effect_summary.csv")
    if round_logs_df.empty or "selected_sender_id" not in round_logs_df.columns:
        summary_df = pd.DataFrame()
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary_df

    summary_rows = []
    for sender_id, sender_df in round_logs_df.groupby("selected_sender_id"):
        num_selected = len(sender_df)
        selected_model = sender_df["selected_model"].astype(str)
        num_selected_receiver_loc = int((selected_model == "receiver_loc").sum())
        num_selected_local_continue = int((selected_model == "local_continue").sum())
        num_selected_teacher_guided = int((selected_model == "teacher_guided").sum())
        num_selected_same_origin_merge = int((selected_model == "same_origin_merge").sum())
        num_rollback_to_local = int(sender_df["rollback_to_local"].astype(bool).sum())

        summary_rows.append({
            "selected_sender_id": sender_id,
            "num_selected": num_selected,
            "num_selected_receiver_loc": num_selected_receiver_loc,
            "num_selected_local_continue": num_selected_local_continue,
            "num_selected_teacher_guided": num_selected_teacher_guided,
            "num_selected_same_origin_merge": num_selected_same_origin_merge,
            "rollback_rate": num_rollback_to_local / max(num_selected, 1),
            "local_continue_accept_rate": num_selected_local_continue / max(num_selected, 1),
            "teacher_guided_accept_rate": num_selected_teacher_guided / max(num_selected, 1),
            "same_origin_merge_accept_rate": num_selected_same_origin_merge / max(num_selected, 1),
            "mean_best_alpha": _safe_numeric_mean(sender_df, "best_alpha"),
            "mean_delta_local_continue_vs_loc": _safe_numeric_mean(sender_df, "delta_local_continue_vs_loc"),
            "mean_delta_teacher_guided_vs_local_continue": _safe_numeric_mean(
                sender_df, "delta_teacher_guided_vs_local_continue"
            ),
            "mean_delta_teacher_guided_vs_loc": _safe_numeric_mean(sender_df, "delta_teacher_guided_vs_loc"),
            "mean_delta_same_origin_merge_vs_teacher_guided": _safe_numeric_mean(
                sender_df, "delta_same_origin_merge_vs_teacher_guided"
            ),
            "mean_delta_same_origin_merge_vs_loc": _safe_numeric_mean(sender_df, "delta_same_origin_merge_vs_loc"),
            "mean_teacher_better_ratio_avg": _safe_numeric_mean(
                sender_df, "teacher_guided_teacher_better_ratio_avg"
            ),
            "mean_teacher_worse_ratio_avg": _safe_numeric_mean(
                sender_df, "teacher_guided_teacher_worse_ratio_avg"
            ),
            "mean_positive_active_ratio_avg": _safe_numeric_mean(
                sender_df, "teacher_guided_positive_active_ratio_avg"
            ),
            "mean_repulsion_active_ratio_avg": _safe_numeric_mean(
                sender_df, "teacher_guided_repulsion_active_ratio_avg"
            ),
            "mean_positive_loss_avg": _safe_numeric_mean(sender_df, "teacher_guided_positive_loss_avg"),
            "mean_repulsion_loss_avg": _safe_numeric_mean(sender_df, "teacher_guided_repulsion_loss_avg"),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("selected_sender_id").reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_df


def make_disjoint_pairs(active_client_ids: List[int], global_round: int, cfg):
    ids = list(active_client_ids)
    if len(ids) == 0:
        raise ValueError("active_client_ids must not be empty.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"active_client_ids contains duplicate client ids: {ids}")

    mode = getattr(cfg.decentralized_gcml, "pair_schedule_mode", "round_robin_disjoint").lower()
    if mode == "random_disjoint":
        shuffled_ids = ids[:]
        seed = int(cfg.train.random_seed) + int(global_round)
        random.Random(seed).shuffle(shuffled_ids)

        idle_client_ids = []
        if len(shuffled_ids) % 2 == 1:
            idle_client_ids.append(shuffled_ids.pop())

        pairs = [
            (shuffled_ids[idx], shuffled_ids[idx + 1])
            for idx in range(0, len(shuffled_ids), 2)
        ]
    elif mode == "round_robin_disjoint":
        if len(ids) % 2 == 1:
            roster = [None] + ids
        else:
            roster = ids[:]

        n = len(roster)
        if n == 1:
            return [], [roster[0]]

        fixed = roster[0]
        rotating = roster[1:]
        shift = (int(global_round) - 1) % max(n - 1, 1)
        if shift:
            rotating = rotating[-shift:] + rotating[:-shift]
        arranged = [fixed] + rotating

        pairs = []
        idle_client_ids = []
        for idx in range(n // 2):
            left = arranged[idx]
            right = arranged[n - 1 - idx]
            if left is None and right is None:
                continue
            if left is None:
                idle_client_ids.append(right)
                continue
            if right is None:
                idle_client_ids.append(left)
                continue
            pairs.append((left, right))
    else:
        raise ValueError(
            f"Unsupported pair_schedule_mode={mode}. "
            "Use 'round_robin_disjoint' or 'random_disjoint'."
        )

    paired_ids = [cid for pair in pairs for cid in pair]
    duplicate_paired_ids = sorted({cid for cid in paired_ids if paired_ids.count(cid) > 1})
    if duplicate_paired_ids:
        raise RuntimeError(
            f"Pair schedule is not disjoint in round {global_round}: {pairs}. "
            f"Duplicate ids: {duplicate_paired_ids}"
        )
    unknown_ids = sorted(set(paired_ids + idle_client_ids) - set(ids))
    if unknown_ids:
        raise RuntimeError(f"Pair schedule produced unknown client ids: {unknown_ids}")

    return pairs, idle_client_ids


def save_sender_receiver_effect_summary(round_logs, save_dir: str) -> pd.DataFrame:
    round_logs_df = pd.DataFrame(round_logs)
    summary_path = os.path.join(save_dir, "sender_receiver_effect_summary.csv")
    if round_logs_df.empty or "receiver_id" not in round_logs_df.columns:
        summary_df = pd.DataFrame()
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary_df

    if "sender_id" not in round_logs_df.columns and "selected_sender_id" in round_logs_df.columns:
        round_logs_df["sender_id"] = round_logs_df["selected_sender_id"]

    summary_rows = []
    for (receiver_id, sender_id), group_df in round_logs_df.groupby(["receiver_id", "sender_id"]):
        selected_model = group_df["selected_model"].astype(str)
        num_events = len(group_df)
        num_selected_receiver_loc = int((selected_model == "receiver_loc").sum())
        num_selected_local_continue = int((selected_model == "local_continue").sum())
        num_selected_teacher_guided = int((selected_model == "teacher_guided").sum())
        num_selected_same_origin_merge = int((selected_model == "same_origin_merge").sum())

        summary_rows.append({
            "receiver_id": receiver_id,
            "sender_id": sender_id,
            "num_events": num_events,
            "num_selected_receiver_loc": num_selected_receiver_loc,
            "num_selected_local_continue": num_selected_local_continue,
            "num_selected_teacher_guided": num_selected_teacher_guided,
            "num_selected_same_origin_merge": num_selected_same_origin_merge,
            "rollback_rate": num_selected_receiver_loc / max(num_events, 1),
            "local_continue_accept_rate": num_selected_local_continue / max(num_events, 1),
            "teacher_guided_accept_rate": num_selected_teacher_guided / max(num_events, 1),
            "same_origin_merge_accept_rate": num_selected_same_origin_merge / max(num_events, 1),
            "mean_best_alpha": _safe_numeric_mean(group_df, "best_alpha"),
            "mean_selected_val_loss": _safe_numeric_mean(group_df, "selected_val_loss"),
            "mean_delta_local_continue_vs_loc": _safe_numeric_mean(group_df, "delta_local_continue_vs_loc"),
            "mean_delta_teacher_guided_vs_loc": _safe_numeric_mean(group_df, "delta_teacher_guided_vs_loc"),
            "mean_delta_teacher_guided_vs_local_continue": _safe_numeric_mean(
                group_df, "delta_teacher_guided_vs_local_continue"
            ),
            "mean_delta_same_origin_merge_vs_loc": _safe_numeric_mean(group_df, "delta_same_origin_merge_vs_loc"),
            "mean_delta_same_origin_merge_vs_local_continue": _safe_numeric_mean(
                group_df, "delta_same_origin_merge_vs_local_continue"
            ),
            "mean_delta_same_origin_merge_vs_teacher_guided": _safe_numeric_mean(
                group_df, "delta_same_origin_merge_vs_teacher_guided"
            ),
            "mean_teacher_better_ratio_avg": _safe_numeric_mean(
                group_df, "teacher_guided_teacher_better_ratio_avg"
            ),
            "mean_teacher_worse_ratio_avg": _safe_numeric_mean(
                group_df, "teacher_guided_teacher_worse_ratio_avg"
            ),
            "mean_positive_active_ratio_avg": _safe_numeric_mean(
                group_df, "teacher_guided_positive_active_ratio_avg"
            ),
            "mean_repulsion_active_ratio_avg": _safe_numeric_mean(
                group_df, "teacher_guided_repulsion_active_ratio_avg"
            ),
            "mean_positive_loss_avg": _safe_numeric_mean(group_df, "teacher_guided_positive_loss_avg"),
            "mean_repulsion_loss_avg": _safe_numeric_mean(group_df, "teacher_guided_repulsion_loss_avg"),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["receiver_id", "sender_id"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_df


def save_client_effect_summary(round_logs, save_dir: str) -> pd.DataFrame:
    round_logs_df = pd.DataFrame(round_logs)
    summary_path = os.path.join(save_dir, "client_effect_summary.csv")
    if round_logs_df.empty or "receiver_id" not in round_logs_df.columns:
        summary_df = pd.DataFrame()
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        return summary_df

    summary_rows = []
    for receiver_id, receiver_df in round_logs_df.groupby("receiver_id"):
        selected_model = receiver_df["selected_model"].astype(str)
        num_events = len(receiver_df)
        num_rollback_to_local = int((selected_model == "receiver_loc").sum())
        num_selected_local_continue = int((selected_model == "local_continue").sum())
        num_selected_teacher_guided = int((selected_model == "teacher_guided").sum())
        num_selected_same_origin_merge = int((selected_model == "same_origin_merge").sum())

        sort_columns = [
            column
            for column in ["global_round", "round", "pair_index"]
            if column in receiver_df.columns
        ]
        ordered_df = receiver_df.sort_values(sort_columns) if sort_columns else receiver_df
        selected_val_losses = pd.to_numeric(ordered_df["selected_val_loss"], errors="coerce").dropna()
        final_selected_val_loss = (
            float(selected_val_losses.iloc[-1])
            if len(selected_val_losses) > 0
            else float("nan")
        )

        summary_rows.append({
            "receiver_id": receiver_id,
            "num_events": num_events,
            "num_rollback_to_local": num_rollback_to_local,
            "num_selected_local_continue": num_selected_local_continue,
            "num_selected_teacher_guided": num_selected_teacher_guided,
            "num_selected_same_origin_merge": num_selected_same_origin_merge,
            "rollback_rate": num_rollback_to_local / max(num_events, 1),
            "same_origin_merge_accept_rate": num_selected_same_origin_merge / max(num_events, 1),
            "teacher_guided_accept_rate": num_selected_teacher_guided / max(num_events, 1),
            "local_continue_accept_rate": num_selected_local_continue / max(num_events, 1),
            "best_val_loss": _safe_numeric_mean(
                pd.DataFrame({"selected_val_loss": [selected_val_losses.min()]}),
                "selected_val_loss",
            ),
            "final_selected_val_loss": final_selected_val_loss,
            "mean_selected_val_loss": _safe_numeric_mean(receiver_df, "selected_val_loss"),
            "mean_delta_teacher_guided_vs_local_continue": _safe_numeric_mean(
                receiver_df, "delta_teacher_guided_vs_local_continue"
            ),
            "mean_delta_same_origin_merge_vs_local_continue": _safe_numeric_mean(
                receiver_df, "delta_same_origin_merge_vs_local_continue"
            ),
            "mean_delta_same_origin_merge_vs_teacher_guided": _safe_numeric_mean(
                receiver_df, "delta_same_origin_merge_vs_teacher_guided"
            ),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("receiver_id").reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_df


def save_all_clients_best_test_outputs(
    clients,
    save_dir: str,
    cfg,
    best_state_paths=None,
    best_states=None,
):
    ensure_dir(save_dir)
    summary_rows = []
    client_pred_map = {}

    for client in sorted(clients, key=lambda item: item.client_id):
        client_id = client.client_id
        model_path = None
        if best_states is not None and client_id in best_states:
            best_state = clone_state_dict(best_states[client_id])
        else:
            if best_state_paths is None or client_id not in best_state_paths:
                raise ValueError(f"No best model state/path was provided for client_id={client_id}.")
            model_path = best_state_paths[client_id]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Best model not found for client_id={client_id}: {model_path}")
            best_state = torch.load(model_path, map_location=cfg.train.device)

        pred_df, _, loss = client.evaluate_split(best_state, split_name="test")
        pred_df = clean_test_prediction_df(pred_df, cfg.data.horizon)
        metrics = calc_prediction_metrics(pred_df, cfg.data.horizon)

        pred_df.to_csv(
            os.path.join(save_dir, f"client_{client_id}_test_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        error_df = build_prediction_error_df(pred_df, cfg.data.horizon, eps=cfg.decentralized_gcml.eps)
        error_df.to_csv(
            os.path.join(save_dir, f"client_{client_id}_test_prediction_errors.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        save_metrics_csv(metrics, os.path.join(save_dir, f"client_{client_id}_test_metrics.csv"))
        plot_true_pred(
            pred_df["y_true_step_1"].values,
            pred_df["y_pred_step_1"].values,
            save_path=os.path.join(save_dir, f"client_{client_id}_test_prediction.png"),
            title=f"Decentralized GCML Client {client_id} Best Test Prediction",
            show_n=300,
        )

        summary_rows.append({
            "client_id": client_id,
            "client_name": client.client_name,
            "loss": float(loss),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "R2": metrics["R2"],
            "model_path": model_path if model_path is not None else "",
        })
        client_pred_map[client.client_name] = pred_df
        print_metrics(metrics, title=f"Decentralized GCML Client {client_id} Best Test Metrics")

    summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    summary_df.to_csv(
        os.path.join(save_dir, "all_clients_test_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return summary_df, client_pred_map


def train_decentralized_gcml_all_clients_disjoint_bidirectional(cfg):
    dcfg = cfg.decentralized_gcml
    if not getattr(dcfg, "bidirectional_pair_update", True):
        raise ValueError(
            "train_decentralized_gcml_all_clients_disjoint_bidirectional requires "
            "bidirectional_pair_update=True."
        )

    ensure_dir(dcfg.save_dir)
    save_config(cfg, dcfg.save_dir)

    clients, feature_cols_ref = build_clients(cfg)
    client_map = {client.client_id: client for client in clients}
    active_client_ids = list(dcfg.active_client_ids)
    if len(active_client_ids) < 2:
        raise ValueError("At least two active clients are required.")
    if len(set(active_client_ids)) != len(active_client_ids):
        raise ValueError(f"active_client_ids contains duplicates: {active_client_ids}")

    missing_client_ids = [cid for cid in active_client_ids if cid not in client_map]
    if missing_client_ids:
        raise ValueError(
            f"active_client_ids contains ids not in built clients: {missing_client_ids}. "
            f"Built clients: {sorted(client_map.keys())}"
        )

    input_dim = len(feature_cols_ref)
    output_dim = cfg.data.horizon
    client_states = {}
    for client_id in active_client_ids:
        init_model = build_model(input_dim=input_dim, output_dim=output_dim, cfg=cfg)
        client_states[client_id] = clone_state_dict(init_model.state_dict())

    best_val_losses = {client_id: float("inf") for client_id in active_client_ids}
    best_state_paths = {
        client_id: os.path.join(
            dcfg.save_dir,
            dcfg.best_model_name_template.format(client_id=client_id),
        )
        for client_id in active_client_ids
    }
    best_model_saved = {client_id: False for client_id in active_client_ids}
    round_logs = []
    pair_logs = []
    total_rounds = int(getattr(dcfg, "global_rounds", dcfg.rounds))
    warmup_local_epochs = int(getattr(dcfg, "warmup_local_epochs", 0))
    if total_rounds < 1:
        raise ValueError("global_rounds must be at least 1 for all-client GCML training.")

    print("=" * 100)
    print("Decentralized GCML all-client disjoint bidirectional training started")
    print(f"Training mode: {dcfg.training_mode}")
    print(f"Active clients: {active_client_ids}")
    print(f"Global rounds: {total_rounds}")
    print(f"Warmup local epochs: {getattr(dcfg, 'warmup_local_epochs', 0)}")
    print(f"Pair schedule mode: {dcfg.pair_schedule_mode}")
    print(f"Input features: {feature_cols_ref}")
    print(f"Clients built: {len(clients)}")
    print(f"Device: {cfg.train.device}")
    print("=" * 100)

    warmup_logs = []
    if warmup_local_epochs > 0:
        print("=" * 100)
        print(f"Warmup local training before global mutual learning: {warmup_local_epochs} epoch(s)")
        print("=" * 100)

        for client_id in active_client_ids:
            client = client_map[client_id]
            update = client.local_update(
                global_state_dict=client_states[client_id],
                local_epochs=warmup_local_epochs,
            )
            client_states[client_id] = clone_state_dict(update["state_dict"])

            warmup_row = {
                "client_id": client_id,
                "client_name": client.client_name,
                "warmup_local_epochs": warmup_local_epochs,
                "warmup_train_loss": float(update["train_loss"]),
                "warmup_val_loss": float(update["val_loss"]),
            }
            warmup_logs.append(warmup_row)

            print(
                f"Warmup Client{client_id} | "
                f"TrainLoss: {float(update['train_loss']):.6f} | "
                f"ValLoss: {float(update['val_loss']):.6f}"
            )

        pd.DataFrame(warmup_logs).to_csv(
            os.path.join(dcfg.save_dir, "warmup_local_training_logs.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    for global_round in range(1, total_rounds + 1):
        round_snapshot = {
            client_id: clone_state_dict(client_states[client_id])
            for client_id in active_client_ids
        }
        pairs, idle_client_ids = make_disjoint_pairs(active_client_ids, global_round, cfg)
        pending_states = {
            client_id: clone_state_dict(round_snapshot[client_id])
            for client_id in active_client_ids
        }
        current_round_logs = []
        idle_client_ids_text = ",".join(str(cid) for cid in idle_client_ids)

        for pair_index, (a_id, b_id) in enumerate(pairs, start=1):
            client_a = client_map[a_id]
            client_b = client_map[b_id]

            a_final_state, a_log = run_receiver_gcml_update(
                receiver_client=client_a,
                receiver_loc_state=round_snapshot[a_id],
                sender_loc_state=round_snapshot[b_id],
                cfg=cfg,
                sender_client=client_b,
            )
            b_final_state, b_log = run_receiver_gcml_update(
                receiver_client=client_b,
                receiver_loc_state=round_snapshot[b_id],
                sender_loc_state=round_snapshot[a_id],
                cfg=cfg,
                sender_client=client_a,
            )

            pending_states[a_id] = clone_state_dict(a_final_state)
            pending_states[b_id] = clone_state_dict(b_final_state)

            pair_id = f"{a_id}-{b_id}"
            a_log.update({
                "round": global_round,
                "global_round": global_round,
                "pair_index": pair_index,
                "pair_id": pair_id,
                "receiver_id": a_id,
                "sender_id": b_id,
                "selected_sender_id": b_id,
                "direction": f"Client{b_id} -> Client{a_id}",
                "idle_client_ids": idle_client_ids_text,
            })
            b_log.update({
                "round": global_round,
                "global_round": global_round,
                "pair_index": pair_index,
                "pair_id": pair_id,
                "receiver_id": b_id,
                "sender_id": a_id,
                "selected_sender_id": a_id,
                "direction": f"Client{a_id} -> Client{b_id}",
                "idle_client_ids": idle_client_ids_text,
            })

            round_logs.append(a_log)
            round_logs.append(b_log)
            current_round_logs.append(a_log)
            current_round_logs.append(b_log)

            pair_logs.append({
                "global_round": global_round,
                "pair_index": pair_index,
                "client_a_id": a_id,
                "client_b_id": b_id,
                "a_selected_model": a_log["selected_model"],
                "b_selected_model": b_log["selected_model"],
                "a_selected_val_loss": a_log["selected_val_loss"],
                "b_selected_val_loss": b_log["selected_val_loss"],
                "a_best_alpha": a_log["best_alpha"],
                "b_best_alpha": b_log["best_alpha"],
                "idle_client_ids": idle_client_ids_text,
            })

        expected_receiver_events = len(active_client_ids) - len(idle_client_ids)
        if len(current_round_logs) != expected_receiver_events:
            raise RuntimeError(
                f"Round {global_round} produced {len(current_round_logs)} receiver events; "
                f"expected {expected_receiver_events}."
            )

        for client_id in active_client_ids:
            client_states[client_id] = clone_state_dict(pending_states[client_id])

        current_logs_by_receiver = {
            int(log["receiver_id"]): log
            for log in current_round_logs
        }
        for client_id in active_client_ids:
            log = current_logs_by_receiver.get(client_id)
            if log is None:
                continue
            selected_val_loss = float(log["selected_val_loss"])
            if selected_val_loss < best_val_losses[client_id]:
                best_val_losses[client_id] = selected_val_loss
                torch.save(clone_state_dict(client_states[client_id]), best_state_paths[client_id])
                best_model_saved[client_id] = True

        selected_models = [str(log["selected_model"]) for log in current_round_logs]
        selected_val_losses = [
            float(log["selected_val_loss"])
            for log in current_round_logs
        ]
        num_receiver_events = len(current_round_logs)
        num_same_origin_merge = selected_models.count("same_origin_merge")
        num_teacher_guided = selected_models.count("teacher_guided")
        num_local_continue = selected_models.count("local_continue")
        num_rollback_to_local = selected_models.count("receiver_loc")
        avg_selected_val_loss = (
            sum(selected_val_losses) / max(len(selected_val_losses), 1)
            if selected_val_losses
            else float("nan")
        )
        merge_accept_rate = num_same_origin_merge / max(num_receiver_events, 1)
        rollback_rate = num_rollback_to_local / max(num_receiver_events, 1)

        print(f"Global Round [{global_round:03d}/{total_rounds}]")
        print(f"pairs: {pairs}")
        if idle_client_ids:
            print(f"idle_client_ids: {idle_client_ids}")
        print(f"avg_selected_val_loss: {avg_selected_val_loss:.6f}")
        print(f"num_receiver_events: {num_receiver_events}")
        print(f"num_same_origin_merge: {num_same_origin_merge}")
        print(f"num_teacher_guided: {num_teacher_guided}")
        print(f"num_local_continue: {num_local_continue}")
        print(f"num_rollback_to_local: {num_rollback_to_local}")
        print(f"merge_accept_rate: {merge_accept_rate:.6f}")
        print(f"rollback_rate: {rollback_rate:.6f}")

    if len(active_client_ids) % 2 == 0:
        expected_total_receiver_events = total_rounds * len(active_client_ids)
        if len(round_logs) != expected_total_receiver_events:
            raise RuntimeError(
                f"Total receiver logs mismatch: got {len(round_logs)}, "
                f"expected {expected_total_receiver_events}."
            )

    for client_id in active_client_ids:
        final_path = os.path.join(dcfg.save_dir, f"final_client_{client_id}_model.pth")
        torch.save(clone_state_dict(client_states[client_id]), final_path)
        if not best_model_saved[client_id]:
            _, _, val_loss = client_map[client_id].evaluate_split(client_states[client_id], split_name="val")
            best_val_losses[client_id] = float(val_loss)
            torch.save(clone_state_dict(client_states[client_id]), best_state_paths[client_id])
            best_model_saved[client_id] = True

    round_logs_df = save_round_logs_csv(
        round_logs,
        os.path.join(dcfg.save_dir, "decentralized_gcml_round_logs.csv"),
    )
    pair_logs_df = pd.DataFrame(pair_logs)
    pair_logs_df.to_csv(
        os.path.join(dcfg.save_dir, "decentralized_gcml_pair_logs.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    sender_receiver_effect_summary_df = save_sender_receiver_effect_summary(round_logs, dcfg.save_dir)
    client_effect_summary_df = save_client_effect_summary(round_logs, dcfg.save_dir)

    active_clients = [client_map[client_id] for client_id in active_client_ids]
    test_summary_df, client_test_pred_map = save_all_clients_best_test_outputs(
        clients=active_clients,
        best_state_paths=best_state_paths,
        save_dir=dcfg.save_dir,
        cfg=cfg,
    )

    print("\nAll-client best-model test summary:")
    print(test_summary_df)
    print("\nSender-receiver effect summary:")
    print(sender_receiver_effect_summary_df)
    print("\nClient effect summary:")
    print(client_effect_summary_df)
    print(f"\nReceiver log rows: {len(round_logs)}")
    print(f"Results directory: {dcfg.save_dir}")

    return {
        "cfg": cfg,
        "clients": clients,
        "active_client_ids": active_client_ids,
        "client_states": client_states,
        "best_val_losses": best_val_losses,
        "best_state_paths": best_state_paths,
        "round_logs": round_logs,
        "round_logs_df": round_logs_df,
        "pair_logs_df": pair_logs_df,
        "sender_receiver_effect_summary_df": sender_receiver_effect_summary_df,
        "client_effect_summary_df": client_effect_summary_df,
        "client_test_summary_df": test_summary_df,
        "client_test_pred_map": client_test_pred_map,
        "save_dir": dcfg.save_dir,
    }


def train_decentralized_gcml_two_clients(cfg):
    dcfg = cfg.decentralized_gcml
    if len(cfg.data.client_files) != 2:
        raise ValueError(
            "Decentralized GCML currently supports exactly two clients. "
            f"Got {len(cfg.data.client_files)} client files."
        )
    if dcfg.pair_mode.lower() != "alternate":
        raise ValueError("Decentralized GCML currently supports pair_mode='alternate' only.")

    ensure_dir(dcfg.save_dir)
    save_config(cfg, dcfg.save_dir)

    clients, feature_cols_ref = build_clients(cfg)
    client1, client2 = clients
    input_dim = len(feature_cols_ref)
    output_dim = cfg.data.horizon

    init_model_client1 = build_model(input_dim=input_dim, output_dim=output_dim, cfg=cfg)
    init_model_client2 = build_model(input_dim=input_dim, output_dim=output_dim, cfg=cfg)
    client1_state = clone_state_dict(init_model_client1.state_dict())
    client2_state = clone_state_dict(init_model_client2.state_dict())

    round_logs = []
    best_val_losses = {1: float("inf"), 2: float("inf")}

    print("=" * 100)
    print("Decentralized GCML training started")
    print("Method: Gossip sender-receiver conditional mutual learning with rollback")
    print(f"Input features: {feature_cols_ref}")
    print(f"Clients: {len(clients)}")
    print(f"Device: {cfg.train.device}")
    print("=" * 100)

    for rnd in range(1, dcfg.rounds + 1):
        update1 = client1.local_update(
            global_state_dict=client1_state,
            local_epochs=dcfg.local_epochs,
        )
        update2 = client2.local_update(
            global_state_dict=client2_state,
            local_epochs=dcfg.local_epochs,
        )

        client1_loc_state = update1["state_dict"]
        client2_loc_state = update2["state_dict"]

        if rnd % 2 == 1:
            direction = "Client2 -> Client1"
            receiver_final_state, log_dict = run_receiver_gcml_update(
                receiver_client=client1,
                receiver_loc_state=client1_loc_state,
                sender_loc_state=client2_loc_state,
                cfg=cfg,
            )
            client1_state = receiver_final_state
            client2_state = clone_state_dict(client2_loc_state)
            client1_current_val_loss = log_dict["selected_val_loss"]
            client2_current_val_loss = update2["val_loss"]
        else:
            direction = "Client1 -> Client2"
            receiver_final_state, log_dict = run_receiver_gcml_update(
                receiver_client=client2,
                receiver_loc_state=client2_loc_state,
                sender_loc_state=client1_loc_state,
                cfg=cfg,
            )
            client1_state = clone_state_dict(client1_loc_state)
            client2_state = receiver_final_state
            client1_current_val_loss = update1["val_loss"]
            client2_current_val_loss = log_dict["selected_val_loss"]

        save_best_if_needed(1, client1_state, client1_current_val_loss, best_val_losses, cfg)
        save_best_if_needed(2, client2_state, client2_current_val_loss, best_val_losses, cfg)

        row = {
            "round": rnd,
            "direction": direction,
            "client1_local_train_loss": float(update1["train_loss"]),
            "client1_local_val_loss": float(update1["val_loss"]),
            "client2_local_train_loss": float(update2["train_loss"]),
            "client2_local_val_loss": float(update2["val_loss"]),
        }
        row.update(log_dict)
        round_logs.append(row)

        print_round_log(rnd, dcfg.rounds, direction, log_dict)

    torch.save(clone_state_dict(client1_state), os.path.join(dcfg.save_dir, "final_client_1_model.pth"))
    torch.save(clone_state_dict(client2_state), os.path.join(dcfg.save_dir, "final_client_2_model.pth"))

    save_round_logs_csv(
        round_logs,
        os.path.join(dcfg.save_dir, "decentralized_gcml_round_logs.csv"),
    )

    test_summary_df, client_test_pred_map = save_personalized_test_outputs(
        clients=clients,
        client_states=[client1_state, client2_state],
        save_dir=dcfg.save_dir,
        cfg=cfg,
    )

    print("\nPer-client test summary:")
    print(test_summary_df)
    print(f"\nResults directory: {dcfg.save_dir}")

    return {
        "cfg": cfg,
        "clients": clients,
        "client1_state": client1_state,
        "client2_state": client2_state,
        "round_logs": round_logs,
        "client_test_summary_df": test_summary_df,
        "client_test_pred_map": client_test_pred_map,
        "save_dir": dcfg.save_dir,
    }


def train_decentralized_gcml_receiver_with_random_senders(cfg):
    dcfg = cfg.decentralized_gcml
    if len(cfg.data.client_files) < 2:
        raise ValueError("At least two clients are required for receiver-sender GCML training.")

    ensure_dir(dcfg.save_dir)
    save_config(cfg, dcfg.save_dir)

    clients, feature_cols_ref = build_clients(cfg)
    client_map = {client.client_id: client for client in clients}
    receiver_id = dcfg.receiver_client_id
    if receiver_id not in client_map:
        raise ValueError(f"receiver_client_id={receiver_id} is not in built clients: {sorted(client_map.keys())}")

    receiver_client = client_map[receiver_id]
    sender_candidate_ids = list(dcfg.sender_candidate_client_ids)
    if receiver_id in sender_candidate_ids:
        raise ValueError("sender_candidate_client_ids must not include receiver_client_id.")

    missing_sender_ids = [cid for cid in sender_candidate_ids if cid not in client_map]
    if missing_sender_ids:
        raise ValueError(
            f"sender_candidate_client_ids contains ids not in built clients: {missing_sender_ids}. "
            f"Built clients: {sorted(client_map.keys())}"
        )

    sender_clients = [client_map[cid] for cid in sender_candidate_ids]
    if len(sender_clients) == 0:
        raise ValueError("No sender clients are available.")

    sender_selection_mode = getattr(dcfg, "sender_selection_mode", "round_robin").lower()
    if sender_selection_mode not in {"random", "round_robin"}:
        raise ValueError(
            f"Unsupported sender_selection_mode={sender_selection_mode}. "
            "Use 'random' or 'round_robin'."
        )

    input_dim = len(feature_cols_ref)
    output_dim = cfg.data.horizon
    client_states = {}
    for client in clients:
        init_model = build_model(input_dim=input_dim, output_dim=output_dim, cfg=cfg)
        client_states[client.client_id] = clone_state_dict(init_model.state_dict())

    round_logs = []
    best_receiver_val_loss = float("inf")
    best_model_path = os.path.join(dcfg.save_dir, dcfg.best_receiver_model_name)

    print("=" * 100)
    print("Decentralized GCML receiver-focused training started")
    print(f"Receiver: Client{receiver_id}")
    print(f"Sender pool: {[client.client_id for client in sender_clients]}")
    print(
        "Method: fixed receiver GCML with sender -> receiver transfer, "
        f"selection_mode={sender_selection_mode}"
    )
    print(f"Input features: {feature_cols_ref}")
    print(f"Clients: {len(clients)}")
    print(f"Device: {cfg.train.device}")
    print("=" * 100)

    for rnd in range(1, dcfg.rounds + 1):
        if sender_selection_mode == "random":
            sender_client = random.choice(sender_clients)
        else:
            sender_client = sender_clients[(rnd - 1) % len(sender_clients)]
        sender_id = sender_client.client_id

        if rnd == 1:
            receiver_update = receiver_client.local_update(
                global_state_dict=client_states[receiver_id],
                local_epochs=dcfg.local_epochs,
            )
            receiver_loc_state = receiver_update["state_dict"]
            receiver_local_train_loss = float(receiver_update["train_loss"])
            receiver_local_val_loss = float(receiver_update["val_loss"])
            receiver_local_update_executed = True
        else:
            # After the first round, the fixed receiver starts from its previous
            # personalized state and is updated only by receiver-side GCML.
            receiver_loc_state = clone_state_dict(client_states[receiver_id])
            _, _, receiver_local_val_loss_eval = receiver_client.evaluate_split(receiver_loc_state, split_name="val")
            receiver_local_train_loss = float("nan")
            receiver_local_val_loss = float(receiver_local_val_loss_eval)
            receiver_local_update_executed = False

        sender_update = sender_client.local_update(
            global_state_dict=client_states[sender_id],
            local_epochs=dcfg.local_epochs,
        )

        sender_loc_state = sender_update["state_dict"]

        receiver_final_state, log_dict = run_receiver_gcml_update(
            receiver_client=receiver_client,
            receiver_loc_state=receiver_loc_state,
            sender_loc_state=sender_loc_state,
            cfg=cfg,
            sender_client=sender_client,
        )

        # Only the receiver accepts the selected branch state. The selected sender
        # keeps its own local model; teacher-guided updates never flow back.
        client_states[receiver_id] = receiver_final_state
        client_states[sender_id] = clone_state_dict(sender_loc_state)

        receiver_current_val_loss = log_dict["selected_val_loss"]
        if receiver_current_val_loss < best_receiver_val_loss:
            best_receiver_val_loss = float(receiver_current_val_loss)
            torch.save(clone_state_dict(client_states[receiver_id]), best_model_path)

        direction = f"Client{sender_id} -> Client{receiver_id}"
        row = {
            "round": rnd,
            "direction": direction,
            "receiver_id": receiver_id,
            "selected_sender_id": sender_id,
            "receiver_local_update_executed": receiver_local_update_executed,
            "receiver_local_train_loss": receiver_local_train_loss,
            "receiver_local_val_loss": receiver_local_val_loss,
            "sender_local_train_loss": float(sender_update["train_loss"]),
            "sender_local_val_loss": float(sender_update["val_loss"]),
        }
        row.update(log_dict)
        round_logs.append(row)

        print_round_log(rnd, dcfg.rounds, direction, log_dict)

    if not os.path.exists(best_model_path):
        torch.save(clone_state_dict(client_states[receiver_id]), best_model_path)

    final_model_path = os.path.join(dcfg.save_dir, f"final_receiver_client_{receiver_id}_model.pth")
    torch.save(clone_state_dict(client_states[receiver_id]), final_model_path)

    round_logs_df = save_round_logs_csv(
        round_logs,
        os.path.join(dcfg.save_dir, "decentralized_gcml_round_logs.csv"),
    )
    sender_effect_summary_df = save_sender_effect_summary(round_logs, dcfg.save_dir)

    best_receiver_state = torch.load(best_model_path, map_location=cfg.train.device)
    test_summary_df, client_test_pred_map = save_personalized_test_outputs(
        clients=[receiver_client],
        client_states=[best_receiver_state],
        save_dir=dcfg.save_dir,
        cfg=cfg,
    )

    print("\nReceiver test summary using best receiver model:")
    print(test_summary_df)
    print("\nSender effect summary:")
    print(sender_effect_summary_df)
    print(f"\nBest receiver model saved to: {best_model_path}")
    print(f"Final receiver model saved to: {final_model_path}")
    print(f"Results directory: {dcfg.save_dir}")

    return {
        "cfg": cfg,
        "clients": clients,
        "receiver_client": receiver_client,
        "receiver_state": best_receiver_state,
        "receiver_id": receiver_id,
        "round_logs": round_logs,
        "round_logs_df": round_logs_df,
        "sender_effect_summary_df": sender_effect_summary_df,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "client_test_summary_df": test_summary_df,
        "client_test_pred_map": client_test_pred_map,
        "save_dir": dcfg.save_dir,
    }


def set_default_nine_client_files(cfg):
    cfg.data.client_files = [
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_1_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_2_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_3_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_4_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_5_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_6_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_7_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_8_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_9_load_weather_30min.csv"),
    ]
    return cfg


def build_direct_net_load_cfg(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.data.target_col = cfg.data.net_load_col
    cfg.feature.use_target_history = True
    cfg.feature.raw_feature_cols = []
    return cfg


def build_indirect_gc_cfg(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.data.target_col = "gc"

    # GC model features: adjust this block when grid consumption should use
    # a different feature set from generation.
    cfg.feature.use_target_history = True
    cfg.feature.raw_feature_cols = []
    cfg.feature.use_slot_sin_cos = True
    cfg.feature.use_weekday_sin_cos = True
    cfg.feature.use_month_sin_cos = True
    cfg.feature.use_is_weekend = True
    cfg.feature.use_is_holiday = False
    cfg.feature.use_temp_c = True
    cfg.feature.use_rh = False
    cfg.feature.use_wind = True
    cfg.feature.use_ghi = False
    cfg.feature.use_apparent_temp = False
    return cfg


def build_indirect_gg_cfg(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg.data.target_col = "gg"

    # GG model features: this block is intentionally separate from GC.
    # Edit these switches to use the feature set you want for generation.
    cfg.feature.use_target_history = True
    cfg.feature.raw_feature_cols = []
    cfg.feature.use_slot_sin_cos = False
    cfg.feature.use_weekday_sin_cos = False
    cfg.feature.use_month_sin_cos = False
    cfg.feature.use_is_weekend = False
    cfg.feature.use_is_holiday = False
    cfg.feature.use_temp_c = True
    cfg.feature.use_rh = False
    cfg.feature.use_wind = True
    cfg.feature.use_ghi = True
    cfg.feature.use_apparent_temp = False
    return cfg


def save_net_load_outputs(net_load_client_pred_map, net_load_summary_df, save_dir: str, title_prefix: str, cfg):
    ensure_dir(save_dir)

    for row in net_load_summary_df.to_dict("records"):
        client_id = row["client_id"]
        client_name = row["client_name"]
        pred_df = net_load_client_pred_map[client_name]
        metrics = {
            "MAE": row["MAE"],
            "RMSE": row["RMSE"],
            "MAPE_percent": row["MAPE_percent"],
            "R2": row["R2"],
        }

        pred_df.to_csv(
            os.path.join(save_dir, f"client_{client_id}_test_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        error_df = build_prediction_error_df(pred_df, cfg.data.horizon, eps=cfg.decentralized_gcml.eps)
        error_df.to_csv(
            os.path.join(save_dir, f"client_{client_id}_test_prediction_errors.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        save_metrics_csv(metrics, os.path.join(save_dir, f"client_{client_id}_test_metrics.csv"))
        plot_true_pred(
            pred_df["y_true_step_1"].values,
            pred_df["y_pred_step_1"].values,
            save_path=os.path.join(save_dir, f"client_{client_id}_test_prediction.png"),
            title=f"{title_prefix} Client {client_id} Test Prediction",
            show_n=300,
        )

    net_load_summary_df.to_csv(
        os.path.join(save_dir, "all_clients_test_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def load_existing_client_test_predictions(prediction_dir: str, active_client_ids: List[int], cfg):
    pred_map = {}
    summary_rows = []
    for client_id in active_client_ids:
        client_name = f"client_{client_id}"
        pred_path = os.path.join(prediction_dir, f"client_{client_id}_test_predictions.csv")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(
                f"Existing GC prediction file not found for Client{client_id}: {pred_path}"
            )

        pred_df = pd.read_csv(pred_path)
        pred_df = clean_test_prediction_df(pred_df, cfg.data.horizon)
        metrics = calc_prediction_metrics(pred_df, cfg.data.horizon)
        pred_map[client_name] = pred_df
        summary_rows.append({
            "client_id": client_id,
            "client_name": client_name,
            "loss": float("nan"),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "R2": metrics["R2"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    return pred_map, summary_df


def save_net_load_from_existing_gc_predictions(base_cfg, gg_result, existing_gc_prediction_dir: str, save_dir: str):
    ensure_dir(save_dir)
    active_client_ids = list(base_cfg.decentralized_gcml.active_client_ids)
    horizon = base_cfg.data.horizon

    gc_client_pred_map, gc_summary_df = load_existing_client_test_predictions(
        prediction_dir=existing_gc_prediction_dir,
        active_client_ids=active_client_ids,
        cfg=base_cfg,
    )

    gg_client_pred_map = gg_result["client_test_pred_map"]
    net_load_client_pred_map = {}
    summary_rows = []

    for client_id in active_client_ids:
        client_name = f"client_{client_id}"
        if client_name not in gg_client_pred_map:
            raise KeyError(f"GG predictions do not contain {client_name}.")

        gc_pred_df = gc_client_pred_map[client_name]
        gg_pred_df = gg_client_pred_map[client_name]
        net_load_pred_df = combine_prediction_frames(gc_pred_df, gg_pred_df, horizon=horizon, op="subtract")
        metrics = calc_prediction_metrics(net_load_pred_df, horizon)

        summary_rows.append({
            "client_id": client_id,
            "client_name": client_name,
            "loss": float("nan"),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "R2": metrics["R2"],
        })
        net_load_client_pred_map[client_name] = net_load_pred_df
        print_metrics(
            metrics,
            title=(
                "Decentralized GCML Net Load from Existing GC + New GG "
                f"Client {client_id} Test Metrics"
            ),
        )

    net_load_summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    save_net_load_outputs(
        net_load_client_pred_map=net_load_client_pred_map,
        net_load_summary_df=net_load_summary_df,
        save_dir=save_dir,
        title_prefix="Decentralized GCML Net Load from Existing GC + New GG",
        cfg=base_cfg,
    )

    gg_summary_df = gg_result["client_test_summary_df"]
    compare_df = pd.DataFrame([
        {"component": "existing_gc_model_avg_client", **{
            "MAE": float(gc_summary_df["MAE"].mean()),
            "RMSE": float(gc_summary_df["RMSE"].mean()),
            "MAPE_percent": float(gc_summary_df["MAPE_percent"].mean()),
            "R2": float(gc_summary_df["R2"].mean()),
        }},
        {"component": "new_gg_model_avg_client", **{
            "MAE": float(gg_summary_df["MAE"].mean()),
            "RMSE": float(gg_summary_df["RMSE"].mean()),
            "MAPE_percent": float(gg_summary_df["MAPE_percent"].mean()),
            "R2": float(gg_summary_df["R2"].mean()),
        }},
        {"component": "net_load_from_existing_gc_and_new_gg_avg_client", **{
            "MAE": float(net_load_summary_df["MAE"].mean()),
            "RMSE": float(net_load_summary_df["RMSE"].mean()),
            "MAPE_percent": float(net_load_summary_df["MAPE_percent"].mean()),
            "R2": float(net_load_summary_df["R2"].mean()),
        }},
    ])
    compare_df.to_csv(
        os.path.join(save_dir, "net_load_component_compare.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    gc_summary_df.to_csv(
        os.path.join(save_dir, "existing_gc_test_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\nNet load summary from existing GC + new GG:")
    print(net_load_summary_df)
    print(f"\nNet load results directory: {save_dir}")

    return {
        "gc_client_test_summary_df": gc_summary_df,
        "gg_client_test_summary_df": gg_summary_df,
        "client_test_summary_df": net_load_summary_df,
        "client_test_pred_map": net_load_client_pred_map,
        "component_compare_df": compare_df,
        "save_dir": save_dir,
    }


def run_all_clients_gg_and_net_load_from_existing_gc(base_cfg):
    root_save_dir = base_cfg.decentralized_gcml.save_dir
    existing_gc_prediction_dir = getattr(base_cfg.decentralized_gcml, "existing_gc_prediction_dir", None)
    if not existing_gc_prediction_dir:
        existing_gc_prediction_dir = root_save_dir

    gg_cfg = build_indirect_gg_cfg(base_cfg)
    gg_cfg.decentralized_gcml.save_dir = os.path.join(
        root_save_dir,
        base_cfg.decentralized_gcml.gg_model_subdir,
    )

    gg_result = train_decentralized_gcml_all_clients_disjoint_bidirectional(gg_cfg)

    net_load_save_dir = os.path.join(
        root_save_dir,
        base_cfg.decentralized_gcml.net_load_from_existing_gc_subdir,
    )
    net_load_result = save_net_load_from_existing_gc_predictions(
        base_cfg=base_cfg,
        gg_result=gg_result,
        existing_gc_prediction_dir=existing_gc_prediction_dir,
        save_dir=net_load_save_dir,
    )

    return {
        "gg_result": gg_result,
        "net_load_result": net_load_result,
        "existing_gc_prediction_dir": existing_gc_prediction_dir,
        "gg_save_dir": gg_cfg.decentralized_gcml.save_dir,
        "net_load_save_dir": net_load_save_dir,
    }


def run_direct_net_load(base_cfg):
    run_cfg = build_direct_net_load_cfg(base_cfg)
    run_cfg.decentralized_gcml.save_dir = os.path.join(base_cfg.decentralized_gcml.save_dir, "direct_net_load")
    return train_decentralized_gcml_receiver_with_random_senders(run_cfg)


def run_indirect_net_load(base_cfg):
    indirect_root = os.path.join(base_cfg.decentralized_gcml.save_dir, "indirect_net_load")
    ensure_dir(indirect_root)

    gc_cfg = build_indirect_gc_cfg(base_cfg)
    gc_cfg.decentralized_gcml.save_dir = os.path.join(indirect_root, "gc_model")
    gc_result = train_decentralized_gcml_receiver_with_random_senders(gc_cfg)

    gg_cfg = build_indirect_gg_cfg(base_cfg)
    gg_cfg.decentralized_gcml.save_dir = os.path.join(indirect_root, "gg_model")
    gg_result = train_decentralized_gcml_receiver_with_random_senders(gg_cfg)

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
            "RMSE": metrics["RMSE"],
            "MAPE_percent": metrics["MAPE_percent"],
            "R2": metrics["R2"],
        })
        net_load_client_pred_map[client_name] = net_load_pred_df
        print_metrics(metrics, title=f"Decentralized GCML Indirect Net Load Client {client_id} Test Metrics")

    net_load_summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    save_net_load_outputs(
        net_load_client_pred_map=net_load_client_pred_map,
        net_load_summary_df=net_load_summary_df,
        save_dir=indirect_root,
        title_prefix="Decentralized GCML Indirect Net Load",
        cfg=base_cfg,
    )

    compare_df = pd.DataFrame([
        {"component": "gc_model_avg_client", **{
            "MAE": float(gc_result["client_test_summary_df"]["MAE"].mean()),
            "RMSE": float(gc_result["client_test_summary_df"]["RMSE"].mean()),
            "MAPE_percent": float(gc_result["client_test_summary_df"]["MAPE_percent"].mean()),
            "R2": float(gc_result["client_test_summary_df"]["R2"].mean()),
        }},
        {"component": "gg_model_avg_client", **{
            "MAE": float(gg_result["client_test_summary_df"]["MAE"].mean()),
            "RMSE": float(gg_result["client_test_summary_df"]["RMSE"].mean()),
            "MAPE_percent": float(gg_result["client_test_summary_df"]["MAPE_percent"].mean()),
            "R2": float(gg_result["client_test_summary_df"]["R2"].mean()),
        }},
        {"component": "indirect_net_load_avg_client", **{
            "MAE": float(net_load_summary_df["MAE"].mean()),
            "RMSE": float(net_load_summary_df["RMSE"].mean()),
            "MAPE_percent": float(net_load_summary_df["MAPE_percent"].mean()),
            "R2": float(net_load_summary_df["R2"].mean()),
        }},
    ])
    compare_df.to_csv(
        os.path.join(indirect_root, "indirect_net_load_component_compare.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\nIndirect net load per-client summary:")
    print(net_load_summary_df)
    print(f"\nIndirect net load results directory: {indirect_root}")

    return {
        "gc_result": gc_result,
        "gg_result": gg_result,
        "client_test_summary_df": net_load_summary_df,
        "client_test_pred_map": net_load_client_pred_map,
        "save_dir": indirect_root,
    }


def main():
    cfg = copy.deepcopy(CFG)
    set_default_nine_client_files(cfg)

    cfg.experiment.task_type = "single_target"
    cfg.data.target_col = "gg"
    cfg.feature.use_target_history = True
    cfg.feature.raw_feature_cols = []
    cfg.feature.use_slot_sin_cos = False
    cfg.feature.use_weekday_sin_cos = False
    cfg.feature.use_month_sin_cos = False
    cfg.feature.use_is_weekend = False
    cfg.feature.use_is_holiday = False
    cfg.feature.use_temp_c = True
    cfg.feature.use_rh = False
    cfg.feature.use_wind = True
    cfg.feature.use_ghi = True
    cfg.feature.use_apparent_temp = False
    cfg.decentralized_gcml.training_mode = "all_clients_disjoint_bidirectional"
    cfg.decentralized_gcml.active_client_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    cfg.decentralized_gcml.global_rounds = 20
    cfg.decentralized_gcml.warmup_local_epochs = 1
    cfg.decentralized_gcml.pair_schedule_mode = "round_robin_disjoint"
    cfg.decentralized_gcml.bidirectional_pair_update = True
    cfg.decentralized_gcml.enable_merge_rollback = True
    cfg.decentralized_gcml.save_dir = os.path.join(
        PROJECT_ROOT,
        "runs",
        cfg.decentralized_gcml.training_mode,
    )
    cfg.decentralized_gcml.existing_gc_prediction_dir = cfg.decentralized_gcml.save_dir
    cfg.decentralized_gcml.gg_model_subdir = "gg_model"
    cfg.decentralized_gcml.net_load_from_existing_gc_subdir = "net_load_from_existing_gc"

    set_seed(cfg.train.random_seed)
    ensure_dir(cfg.decentralized_gcml.save_dir)

    training_mode = cfg.decentralized_gcml.training_mode.lower()
    if training_mode == "all_clients_disjoint_bidirectional":
        run_all_clients_gg_and_net_load_from_existing_gc(cfg)
        return

    if training_mode == "fixed_receiver":
        train_decentralized_gcml_receiver_with_random_senders(cfg)
        return

    raise ValueError(
        f"Unsupported decentralized_gcml.training_mode={cfg.decentralized_gcml.training_mode}. "
        "Use 'all_clients_disjoint_bidirectional' or 'fixed_receiver'."
    )


if __name__ == "__main__":
    main()
