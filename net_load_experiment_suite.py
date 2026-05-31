import argparse
import copy
import os

from utils.runtime_env import ensure_conda_dll_paths

ensure_conda_dll_paths()

import pandas as pd
import torch

from client import FederatedClient, get_loss_fn, get_optimizer, run_one_epoch
from config import CFG, RUNS_DIR
from federated_main import (
    build_indirect_gc_cfg,
    build_indirect_gg_cfg,
    calc_prediction_metrics,
    combine_prediction_frames,
    evaluate_regional_from_predictions,
    run_indirect_net_load,
    save_client_outputs,
    save_combined_net_load_outputs,
    save_regional_outputs,
    train_federated_model,
)
from utils.data_utils import ensure_dir, inverse_transform_array, save_config, set_seed
from utils.metrics import plot_round_curve, print_metrics


def float_tag(value):
    return str(value).replace(".", "p").replace("-", "m")


def build_output_dirs(output_root, experiment_name):
    result_dir = os.path.join(output_root, "results", experiment_name)
    checkpoint_dir = os.path.join(output_root, "checkpoints", experiment_name)
    ensure_dir(result_dir)
    ensure_dir(checkpoint_dir)
    return result_dir, checkpoint_dir


def train_one_local_client_component(client, cfg, result_dir, checkpoint_dir, title_prefix):
    ensure_dir(result_dir)
    ensure_dir(checkpoint_dir)

    model = client.build_model()
    criterion = get_loss_fn(cfg.train.loss_name)
    optimizer = get_optimizer(cfg.train.optimizer_name, model, cfg.train.lr)
    device = client.device

    best_val_rmse = float("inf")
    best_model_path = os.path.join(checkpoint_dir, f"{client.client_name}_best_model.pth")
    final_model_path = os.path.join(checkpoint_dir, f"{client.client_name}_final_model.pth")
    no_improve_epochs = 0
    patience = getattr(cfg.train, "early_stop_patience", 3)

    rows = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, _, _ = run_one_epoch(
            model=model,
            loader=client.data["train_loader"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        val_loss, val_pred_scaled, val_true_scaled = run_one_epoch(
            model=model,
            loader=client.data["val_loader"],
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        val_pred_real = inverse_transform_array(client.data["y_scaler"], val_pred_scaled)
        val_true_real = inverse_transform_array(client.data["y_scaler"], val_true_scaled)
        val_metrics = calc_prediction_metrics(
            pd.DataFrame({
                "y_true_step_1": val_true_real[:, 0],
                "y_pred_step_1": val_pred_real[:, 0],
            }),
            horizon=1,
        )

        improved = val_metrics["RMSE"] < best_val_rmse
        if improved:
            best_val_rmse = val_metrics["RMSE"]
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epochs += 1

        rows.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_MAE": val_metrics["MAE"],
            "val_MSE": val_metrics["MSE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAPE_percent": val_metrics["MAPE_percent"],
            "val_R2": val_metrics["R2"],
            "best_val_RMSE": best_val_rmse,
        })

        if patience and no_improve_epochs >= patience:
            break

    torch.save(model.state_dict(), final_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    pred_df, metrics, test_loss = client.evaluate_split(model.state_dict(), split_name="test")

    pd.DataFrame(rows).to_csv(
        os.path.join(result_dir, f"{client.client_name}_training_log.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    if len(rows) > 0:
        plot_round_curve(
            [row["val_RMSE"] for row in rows],
            title=f"{title_prefix} {client.client_name} Val RMSE",
            xlabel="Epoch",
            ylabel="RMSE",
            save_path=os.path.join(result_dir, f"{client.client_name}_val_rmse_curve.png"),
        )

    return pred_df, {
        "client_id": client.client_id,
        "client_name": client.client_name,
        "loss": float(test_loss),
        "MAE": metrics["MAE"],
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAPE_percent": metrics["MAPE_percent"],
        "R2": metrics["R2"],
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }


def run_local_component(base_cfg, component_name, result_dir, checkpoint_dir, title_prefix):
    cfg = build_indirect_gc_cfg(base_cfg) if component_name == "gc" else build_indirect_gg_cfg(base_cfg)
    ensure_dir(result_dir)
    ensure_dir(checkpoint_dir)
    save_config(cfg, result_dir)
    save_config(cfg, checkpoint_dir)

    pred_map = {}
    summary_rows = []
    clients = []
    for idx, path in enumerate(cfg.data.client_files, start=1):
        client = FederatedClient(client_id=idx, data_path=path, cfg=cfg)
        clients.append(client)
        client_result_dir = os.path.join(result_dir, "local_training_logs", client.client_name)
        client_checkpoint_dir = os.path.join(checkpoint_dir, client.client_name)
        pred_df, row = train_one_local_client_component(
            client,
            cfg,
            result_dir=client_result_dir,
            checkpoint_dir=client_checkpoint_dir,
            title_prefix=title_prefix,
        )
        pred_map[client.client_name] = pred_df
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    regional_df, regional_metrics = evaluate_regional_from_predictions(pred_map, clients, cfg)
    save_client_outputs(pred_map, summary_df, result_dir, split_name="test", plot_title_prefix=title_prefix)
    save_regional_outputs(
        regional_df,
        regional_metrics,
        result_dir,
        prefix="regional_test",
        plot_title=f"{title_prefix} Regional Test Prediction",
    )
    print_metrics(regional_metrics, title=f"{title_prefix} Regional Test Metrics")

    return {
        "cfg": cfg,
        "clients": clients,
        "summary_df": summary_df,
        "regional_df": regional_df,
        "regional_metrics": regional_metrics,
        "pred_map": pred_map,
    }


def run_local_indirect_net_load(base_cfg, output_root):
    result_dir, checkpoint_dir = build_output_dirs(output_root, "local_per_client_indirect_net_load")
    gc_result = run_local_component(
        base_cfg,
        component_name="gc",
        result_dir=os.path.join(result_dir, "gc_model"),
        checkpoint_dir=os.path.join(checkpoint_dir, "gc_model"),
        title_prefix="Local Per-Client GC",
    )
    gg_result = run_local_component(
        base_cfg,
        component_name="gg",
        result_dir=os.path.join(result_dir, "gg_model"),
        checkpoint_dir=os.path.join(checkpoint_dir, "gg_model"),
        title_prefix="Local Per-Client GG",
    )

    horizon = base_cfg.data.horizon
    net_load_pred_map = {}
    summary_rows = []
    for client_name, gc_pred_df in gc_result["pred_map"].items():
        gg_pred_df = gg_result["pred_map"][client_name]
        net_df = combine_prediction_frames(gc_pred_df, gg_pred_df, horizon=horizon, op="subtract")
        metrics = calc_prediction_metrics(net_df, horizon)
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
        net_load_pred_map[client_name] = net_df

    net_summary_df = pd.DataFrame(summary_rows).sort_values("client_id").reset_index(drop=True)
    regional_net_df = combine_prediction_frames(
        gc_result["regional_df"],
        gg_result["regional_df"],
        horizon=horizon,
        op="subtract",
    )
    regional_net_metrics = calc_prediction_metrics(regional_net_df, horizon)
    save_combined_net_load_outputs(
        net_load_client_pred_map=net_load_pred_map,
        net_load_summary_df=net_summary_df,
        regional_net_load_df=regional_net_df,
        regional_net_load_metrics=regional_net_metrics,
        save_dir=result_dir,
        title_prefix="Local Per-Client Indirect Net Load",
    )
    compare_df = pd.DataFrame([
        {"component": "gc_model_regional", **gc_result["regional_metrics"]},
        {"component": "gg_model_regional", **gg_result["regional_metrics"]},
        {"component": "indirect_net_load_regional", **regional_net_metrics},
    ])
    compare_df.to_csv(
        os.path.join(result_dir, "indirect_net_load_component_compare.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print_metrics(regional_net_metrics, title="Local Per-Client Indirect Net Load Regional Test Metrics")
    return {
        "regional_test_metrics": regional_net_metrics,
        "client_test_summary_df": net_summary_df,
        "result_dir": result_dir,
        "checkpoint_dir": checkpoint_dir,
    }


def configure_federated_dirs(cfg, output_root, experiment_name):
    result_dir, checkpoint_dir = build_output_dirs(output_root, experiment_name)
    cfg.federated.save_dir = result_dir
    cfg.federated.checkpoint_dir = checkpoint_dir
    return cfg


def run_fedavg_baseline(base_cfg, output_root):
    cfg = copy.deepcopy(base_cfg)
    cfg.experiment.task_type = "net_load"
    cfg.experiment.net_load_method = "indirect"
    cfg.federated.use_rc_regularization = False
    cfg.federated.rc_lambda = 0.0
    cfg.federated.use_head_personalization = False
    configure_federated_dirs(cfg, output_root, "fedavg_baseline_indirect_net_load")
    return run_indirect_net_load(cfg)


def run_fedub_rc(base_cfg, output_root, rc_lambda, tau):
    cfg = copy.deepcopy(base_cfg)
    cfg.experiment.task_type = "net_load"
    cfg.experiment.net_load_method = "indirect"
    cfg.federated.use_rc_regularization = True
    cfg.federated.rc_lambda = rc_lambda
    cfg.federated.use_head_personalization = True
    cfg.federated.head_personalization_tau = tau
    experiment_name = f"fedub_rc_tau{float_tag(tau)}_lambda{float_tag(rc_lambda)}_indirect_net_load"
    configure_federated_dirs(cfg, output_root, experiment_name)
    return run_indirect_net_load(cfg)


def build_fedavg_gc_compare_row(experiment, result, save_dir, checkpoint_dir):
    regional = result["regional_test_metrics"]
    client_summary = result["client_test_summary_df"]
    return {
        "experiment": experiment,
        "save_dir": save_dir,
        "checkpoint_dir": checkpoint_dir,
        "regional_MAE": regional.get("MAE"),
        "regional_MSE": regional.get("MSE"),
        "regional_RMSE": regional.get("RMSE"),
        "regional_MAPE_percent": regional.get("MAPE_percent"),
        "regional_R2": regional.get("R2"),
        "avg_client_MAE": float(client_summary["MAE"].mean()),
        "avg_client_MSE": float(client_summary["MSE"].mean()),
        "avg_client_RMSE": float(client_summary["RMSE"].mean()),
        "avg_client_MAPE_percent": float(client_summary["MAPE_percent"].mean()),
        "avg_client_R2": float(client_summary["R2"].mean()),
    }


def run_fedavg_rc_compare_gc(base_cfg, output_root, rc_lambda):
    cfg_base = copy.deepcopy(base_cfg)
    cfg_base.experiment.task_type = "single_target"
    gc_base_cfg = build_indirect_gc_cfg(cfg_base)
    gc_base_cfg.federated.aggregation_method = "fedavg"
    gc_base_cfg.federated.use_rc_regularization = False
    gc_base_cfg.federated.rc_lambda = 0.0
    gc_base_cfg.federated.use_head_personalization = False
    configure_federated_dirs(gc_base_cfg, output_root, "fedavg_gc_baseline")
    baseline_result = train_federated_model(
        gc_base_cfg,
        save_dir=gc_base_cfg.federated.save_dir,
        run_label="FedAvg GC Baseline",
    )

    cfg_rc = copy.deepcopy(base_cfg)
    cfg_rc.experiment.task_type = "single_target"
    gc_rc_cfg = build_indirect_gc_cfg(cfg_rc)
    gc_rc_cfg.federated.aggregation_method = "fedavg"
    gc_rc_cfg.federated.use_rc_regularization = True
    gc_rc_cfg.federated.rc_lambda = rc_lambda
    gc_rc_cfg.federated.use_head_personalization = False

    experiment_name = f"fedavg_gc_rc_lambda{float_tag(rc_lambda)}"
    configure_federated_dirs(gc_rc_cfg, output_root, experiment_name)
    rc_result = train_federated_model(
        gc_rc_cfg,
        save_dir=gc_rc_cfg.federated.save_dir,
        run_label=f"FedAvg GC RC Lambda {rc_lambda}",
    )

    compare_df = pd.DataFrame([
        build_fedavg_gc_compare_row(
            "fedavg_gc_baseline",
            baseline_result,
            gc_base_cfg.federated.save_dir,
            gc_base_cfg.federated.checkpoint_dir,
        ),
        build_fedavg_gc_compare_row(
            experiment_name,
            rc_result,
            gc_rc_cfg.federated.save_dir,
            gc_rc_cfg.federated.checkpoint_dir,
        ),
    ])
    summary_dir = os.path.join(output_root, "results")
    ensure_dir(summary_dir)
    compare_df.to_csv(
        os.path.join(summary_dir, "fedavg_gc_rc_compare_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "baseline_result": baseline_result,
        "rc_result": rc_result,
        "compare_df": compare_df,
    }


def summarize_federated_result(result, experiment_name, tau):
    summary_df = result["client_test_summary_df"]
    regional = result["regional_test_metrics"]
    save_dir = result.get("save_dir", "")
    if not save_dir and "cfg" in result:
        save_dir = getattr(result["cfg"].federated, "save_dir", "")

    return {
        "experiment": experiment_name,
        "tau": tau,
        "save_dir": save_dir,
        "regional_MAE": regional.get("MAE"),
        "regional_MSE": regional.get("MSE"),
        "regional_RMSE": regional.get("RMSE"),
        "regional_MAPE_percent": regional.get("MAPE_percent"),
        "regional_R2": regional.get("R2"),
        "avg_client_MAE": float(summary_df["MAE"].mean()),
        "avg_client_MSE": float(summary_df["MSE"].mean()),
        "avg_client_RMSE": float(summary_df["RMSE"].mean()),
        "avg_client_MAPE_percent": float(summary_df["MAPE_percent"].mean()),
        "avg_client_R2": float(summary_df["R2"].mean()),
    }


def run_gc_head_personalization_compare(base_cfg, output_root, rc_lambda):
    results = {}
    compare_rows = []

    for tau in [1.0, 0.2]:
        cfg = copy.deepcopy(base_cfg)
        cfg.experiment.task_type = "single_target"
        gc_cfg = build_indirect_gc_cfg(cfg)

        gc_cfg.federated.aggregation_method = "fedavg"
        gc_cfg.federated.use_rc_regularization = True
        gc_cfg.federated.rc_lambda = rc_lambda
        gc_cfg.federated.use_head_personalization = True
        gc_cfg.federated.head_personalization_tau = tau
        gc_cfg.federated.head_param_prefixes = ["fc1", "fc2"]

        experiment_name = f"fedavg_gc_rc_head_fc1_fc2_tau{float_tag(tau)}_lambda{float_tag(rc_lambda)}"
        configure_federated_dirs(gc_cfg, output_root, experiment_name)

        result = train_federated_model(
            gc_cfg,
            save_dir=gc_cfg.federated.save_dir,
            run_label=f"FedAvg GC + RC + Head Personalization tau={tau}",
        )
        result["save_dir"] = gc_cfg.federated.save_dir
        result["checkpoint_dir"] = gc_cfg.federated.checkpoint_dir

        results[f"tau_{tau}"] = result
        compare_rows.append(summarize_federated_result(result, experiment_name, tau))

    compare_df = pd.DataFrame(compare_rows)
    compare_dir = os.path.join(output_root, "results")
    ensure_dir(compare_dir)
    compare_path = os.path.join(
        compare_dir,
        "fedavg_gc_rc_head_fc1_fc2_tau_compare_summary.csv",
    )
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")

    return {
        "results": results,
        "compare_df": compare_df,
        "compare_path": compare_path,
    }


def run_gc_fc1_head_tau01_warmup5(base_cfg, output_root, rc_lambda):
    cfg = copy.deepcopy(base_cfg)
    cfg.experiment.task_type = "single_target"

    gc_cfg = build_indirect_gc_cfg(cfg)

    gc_cfg.federated.aggregation_method = "fedavg"
    gc_cfg.federated.use_rc_regularization = True
    gc_cfg.federated.rc_lambda = rc_lambda

    gc_cfg.federated.use_head_personalization = True
    gc_cfg.federated.head_param_prefixes = ["fc1"]
    gc_cfg.federated.head_personalization_tau = 0.10
    gc_cfg.federated.head_personalization_warmup_rounds = 5
    gc_cfg.federated.head_mask_update_interval = 1
    gc_cfg.federated.use_head_importance_ema = True
    gc_cfg.federated.head_importance_ema_beta = 0.3

    experiment_name = (
        f"fedavg_gc_rc_fc1_head_tau0p10_warmup5_ema0p3_lambda{float_tag(rc_lambda)}"
    )
    configure_federated_dirs(gc_cfg, output_root, experiment_name)

    result = train_federated_model(
        gc_cfg,
        save_dir=gc_cfg.federated.save_dir,
        run_label="FedAvg GC + RC + fc1-only Head Personalization tau=0.10 warmup=5 EMA=0.3",
    )
    result["save_dir"] = gc_cfg.federated.save_dir
    result["checkpoint_dir"] = gc_cfg.federated.checkpoint_dir

    summary_df = result["client_test_summary_df"]
    regional = result["regional_test_metrics"]

    compare_df = pd.DataFrame([{
        "experiment": experiment_name,
        "rc_lambda": rc_lambda,
        "head_prefixes": "fc1",
        "tau": 0.10,
        "warmup_rounds": 5,
        "use_ema": True,
        "ema_beta": 0.3,
        "save_dir": result.get("save_dir", gc_cfg.federated.save_dir),
        "regional_MAE": regional.get("MAE"),
        "regional_MSE": regional.get("MSE"),
        "regional_RMSE": regional.get("RMSE"),
        "regional_MAPE_percent": regional.get("MAPE_percent"),
        "regional_R2": regional.get("R2"),
        "avg_client_MAE": float(summary_df["MAE"].mean()),
        "avg_client_MSE": float(summary_df["MSE"].mean()),
        "avg_client_RMSE": float(summary_df["RMSE"].mean()),
        "avg_client_MAPE_percent": float(summary_df["MAPE_percent"].mean()),
        "avg_client_R2": float(summary_df["R2"].mean()),
    }])

    compare_dir = os.path.join(output_root, "results")
    ensure_dir(compare_dir)
    compare_path = os.path.join(
        compare_dir,
        "fedavg_gc_rc_fc1_head_tau0p10_warmup5_ema0p3_summary.csv",
    )
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")

    return {
        "result": result,
        "compare_df": compare_df,
        "compare_path": compare_path,
    }


def run_gc_fc1_head_tau01_warmup5_maskint2(base_cfg, output_root, rc_lambda):
    cfg = copy.deepcopy(base_cfg)
    cfg.experiment.task_type = "single_target"

    gc_cfg = build_indirect_gc_cfg(cfg)

    gc_cfg.federated.aggregation_method = "fedavg"
    gc_cfg.federated.use_rc_regularization = True
    gc_cfg.federated.rc_lambda = rc_lambda

    gc_cfg.federated.use_head_personalization = True
    gc_cfg.federated.head_param_prefixes = ["fc1"]
    gc_cfg.federated.head_personalization_tau = 0.10
    gc_cfg.federated.head_personalization_warmup_rounds = 5
    gc_cfg.federated.head_mask_update_interval = 2
    gc_cfg.federated.use_head_importance_ema = True
    gc_cfg.federated.head_importance_ema_beta = 0.3

    experiment_name = (
        f"fedavg_gc_rc_fc1_head_tau0p10_warmup5_maskint2_ema0p3_lambda{float_tag(rc_lambda)}"
    )
    configure_federated_dirs(gc_cfg, output_root, experiment_name)

    result = train_federated_model(
        gc_cfg,
        save_dir=gc_cfg.federated.save_dir,
        run_label="FedAvg GC + RC + fc1-only Head Personalization tau=0.10 warmup=5 mask interval=2 EMA=0.3",
    )
    result["save_dir"] = gc_cfg.federated.save_dir
    result["checkpoint_dir"] = gc_cfg.federated.checkpoint_dir

    summary_df = result["client_test_summary_df"]
    regional = result["regional_test_metrics"]

    compare_df = pd.DataFrame([{
        "experiment": experiment_name,
        "rc_lambda": rc_lambda,
        "head_prefixes": "fc1",
        "tau": 0.10,
        "warmup_rounds": 5,
        "head_mask_update_interval": 2,
        "use_ema": True,
        "ema_beta": 0.3,
        "save_dir": result.get("save_dir", gc_cfg.federated.save_dir),
        "regional_MAE": regional.get("MAE"),
        "regional_MSE": regional.get("MSE"),
        "regional_RMSE": regional.get("RMSE"),
        "regional_MAPE_percent": regional.get("MAPE_percent"),
        "regional_R2": regional.get("R2"),
        "avg_client_MAE": float(summary_df["MAE"].mean()),
        "avg_client_MSE": float(summary_df["MSE"].mean()),
        "avg_client_RMSE": float(summary_df["RMSE"].mean()),
        "avg_client_MAPE_percent": float(summary_df["MAPE_percent"].mean()),
        "avg_client_R2": float(summary_df["R2"].mean()),
    }])

    compare_dir = os.path.join(output_root, "results")
    ensure_dir(compare_dir)
    compare_path = os.path.join(
        compare_dir,
        "fedavg_gc_rc_fc1_head_tau0p10_warmup5_maskint2_ema0p3_summary.csv",
    )
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")

    return {
        "result": result,
        "compare_df": compare_df,
        "compare_path": compare_path,
    }


def run_gc_price_fc1_tau010_ema03(base_cfg, output_root, rc_lambda):
    cfg = copy.deepcopy(base_cfg)
    cfg.experiment.task_type = "single_target"

    gc_cfg = build_indirect_gc_cfg(cfg)

    gc_cfg.federated.aggregation_method = "fedavg"
    gc_cfg.federated.use_rc_regularization = True
    gc_cfg.federated.rc_lambda = rc_lambda

    gc_cfg.federated.use_head_personalization = True
    gc_cfg.federated.head_param_prefixes = ["fc1"]
    gc_cfg.federated.head_param_exact_names = []
    gc_cfg.federated.head_personalization_tau = 0.10
    gc_cfg.federated.head_personalization_warmup_rounds = 5
    gc_cfg.federated.head_mask_update_interval = 1
    gc_cfg.federated.use_head_importance_ema = True
    gc_cfg.federated.head_importance_ema_beta = 0.3

    gc_cfg.feature.use_rrp = True
    gc_cfg.feature.rrp_col = "rrp_aud_per_mwh"

    experiment_name = f"gc_price_fc1_tau0p10_ema0p3_lambda{float_tag(rc_lambda)}"
    configure_federated_dirs(gc_cfg, output_root, experiment_name)

    result = train_federated_model(
        gc_cfg,
        save_dir=gc_cfg.federated.save_dir,
        run_label="GC Price Feature + RC + fc1 Head tau=0.10 EMA=0.3",
    )
    result["save_dir"] = gc_cfg.federated.save_dir
    result["checkpoint_dir"] = gc_cfg.federated.checkpoint_dir

    summary_df = result["client_test_summary_df"]
    regional = result["regional_test_metrics"]
    round_summary = summarize_round_logs(gc_cfg.federated.save_dir)
    avg_client_rmse = float(summary_df["RMSE"].mean())
    regional_rmse = regional.get("RMSE")

    compare_df = pd.DataFrame([{
        "experiment_id": "gc_price_fc1_tau0p10_ema0p3",
        "experiment_name": experiment_name,
        "target": "gc",
        "extra_feature": "rrp_aud_per_mwh",
        "rc_lambda": rc_lambda,
        "head_prefixes": "fc1",
        "head_exact_names": "",
        "tau": 0.10,
        "warmup_rounds": 5,
        "mask_update_interval": 1,
        "use_ema": True,
        "ema_beta": 0.3,
        "baseline_avg_client_RMSE_reference": 1.2935,
        "baseline_regional_RMSE_reference": 4.6004,
        "save_dir": result.get("save_dir", gc_cfg.federated.save_dir),
        "checkpoint_dir": result.get("checkpoint_dir", gc_cfg.federated.checkpoint_dir),
        "regional_MAE": regional.get("MAE"),
        "regional_MSE": regional.get("MSE"),
        "regional_RMSE": regional_rmse,
        "regional_MAPE_percent": regional.get("MAPE_percent"),
        "regional_R2": regional.get("R2"),
        "avg_client_MAE": float(summary_df["MAE"].mean()),
        "avg_client_MSE": float(summary_df["MSE"].mean()),
        "avg_client_RMSE": avg_client_rmse,
        "avg_client_MAPE_percent": float(summary_df["MAPE_percent"].mean()),
        "avg_client_R2": float(summary_df["R2"].mean()),
        "avg_client_RMSE_change_vs_reference": avg_client_rmse - 1.2935,
        "regional_RMSE_change_vs_reference": (
            regional_rmse - 4.6004 if regional_rmse is not None else None
        ),
        **round_summary,
    }])

    compare_dir = os.path.join(output_root, "results")
    ensure_dir(compare_dir)
    compare_path = os.path.join(
        compare_dir,
        "gc_price_fc1_tau0p10_ema0p3_summary.csv",
    )
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")

    return {
        "result": result,
        "compare_df": compare_df,
        "compare_path": compare_path,
    }


def safe_log_value(row, key):
    if row is None or key not in row:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    return float(value)


def read_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def summarize_round_logs(save_dir):
    round_df = read_csv_if_exists(os.path.join(save_dir, "federated_round_logs.csv"))
    if round_df is None or len(round_df) == 0:
        return {
            "best_round_by_avg_client_val_RMSE": None,
            "best_avg_client_val_RMSE": None,
            "best_regional_val_RMSE": None,
            "final_round_avg_client_val_RMSE": None,
            "final_round_regional_val_RMSE": None,
        }

    valid = round_df.dropna(subset=["avg_client_val_RMSE"]) if "avg_client_val_RMSE" in round_df.columns else pd.DataFrame()
    if len(valid) > 0:
        best_row = valid.loc[valid["avg_client_val_RMSE"].idxmin()]
    else:
        best_row = None
    final_row = round_df.iloc[-1]

    return {
        "best_round_by_avg_client_val_RMSE": int(best_row["round"]) if best_row is not None else None,
        "best_avg_client_val_RMSE": safe_log_value(best_row, "avg_client_val_RMSE"),
        "best_regional_val_RMSE": safe_log_value(best_row, "regional_val_RMSE"),
        "final_round_avg_client_val_RMSE": safe_log_value(final_row, "avg_client_val_RMSE"),
        "final_round_regional_val_RMSE": safe_log_value(final_row, "regional_val_RMSE"),
    }


def summarize_rc_rg_logs(save_dir):
    rc_df = read_csv_if_exists(os.path.join(save_dir, "fedavg_rc_rg_client_logs.csv"))
    if rc_df is None or len(rc_df) == 0:
        return {"mean_rc_rg_l2": None, "last_rc_rg_l2": None}

    rc_col = "rc_rg_l2_norm" if "rc_rg_l2_norm" in rc_df.columns else "rc_rg_l2"
    if rc_col not in rc_df.columns:
        return {"mean_rc_rg_l2": None, "last_rc_rg_l2": None}

    last_round = rc_df["round"].max() if "round" in rc_df.columns else None
    last_df = rc_df[rc_df["round"] == last_round] if last_round is not None else rc_df.tail(1)
    return {
        "mean_rc_rg_l2": float(rc_df[rc_col].mean()),
        "last_rc_rg_l2": float(last_df[rc_col].mean()),
    }


def summarize_head_mask_logs(save_dir):
    head_df = read_csv_if_exists(os.path.join(save_dir, "head_mask_round_logs.csv"))
    if head_df is None or len(head_df) == 0:
        return {
            "mean_head_jaccard": None,
            "mean_head_hamming": None,
            "mean_importance_gap_ratio": None,
            "mean_importance_selected_share": None,
        }

    def mean_or_none(column):
        if column not in head_df.columns:
            return None
        values = head_df[column].dropna()
        return float(values.mean()) if len(values) > 0 else None

    return {
        "mean_head_jaccard": mean_or_none("mask_jaccard_with_prev_round"),
        "mean_head_hamming": mean_or_none("mask_hamming_change_rate"),
        "mean_importance_gap_ratio": mean_or_none("importance_gap_mean_ratio"),
        "mean_importance_selected_share": mean_or_none("importance_selected_share"),
    }


def build_gc_head_8exp_summary_row(spec, experiment_name, result, rc_lambda):
    save_dir = result.get("save_dir", result["cfg"].federated.save_dir)
    checkpoint_dir = result.get("checkpoint_dir", getattr(result["cfg"].federated, "checkpoint_dir", ""))
    regional = result["regional_test_metrics"]
    client_summary = result["client_test_summary_df"]

    row = {
        "experiment_id": spec["experiment_id"],
        "experiment_name": experiment_name,
        "head_prefixes": ",".join(spec["head_prefixes"]),
        "head_exact_names": ",".join(spec["head_exact_names"]),
        "tau": spec["tau"],
        "use_ema": spec["use_ema"],
        "ema_beta": spec["ema_beta"] if spec["use_ema"] else None,
        "warmup_rounds": 5,
        "mask_update_interval": 1,
        "rc_lambda": rc_lambda,
        "save_dir": save_dir,
        "checkpoint_dir": checkpoint_dir,
        "regional_MAE": regional.get("MAE"),
        "regional_MSE": regional.get("MSE"),
        "regional_RMSE": regional.get("RMSE"),
        "regional_MAPE_percent": regional.get("MAPE_percent"),
        "regional_R2": regional.get("R2"),
        "avg_client_MAE": float(client_summary["MAE"].mean()),
        "avg_client_MSE": float(client_summary["MSE"].mean()),
        "avg_client_RMSE": float(client_summary["RMSE"].mean()),
        "avg_client_MAPE_percent": float(client_summary["MAPE_percent"].mean()),
        "avg_client_R2": float(client_summary["R2"].mean()),
    }
    row.update(summarize_round_logs(save_dir))
    row.update(summarize_rc_rg_logs(save_dir))
    row.update(summarize_head_mask_logs(save_dir))
    return row


def build_gc_head_8exp_per_client_row(spec, result):
    summary_df = result["client_test_summary_df"].sort_values("client_id")
    row = {"experiment_id": spec["experiment_id"]}
    for metric in ["RMSE", "MAE", "R2"]:
        for _, client_row in summary_df.iterrows():
            client_id = int(client_row["client_id"])
            row[f"client_{client_id}_{metric}"] = float(client_row[metric])
    return row


def save_e6_layer_importance_summary(e6_save_dir, output_root):
    layer_path = os.path.join(e6_save_dir, "head_layer_mask_logs.csv")
    layer_df = read_csv_if_exists(layer_path)
    if layer_df is None or len(layer_df) == 0:
        return None

    summary_df = layer_df.groupby("layer_name").agg(
        mean_selected_ratio=("selected_ratio", "mean"),
        mean_selected_importance_mean=("selected_importance_mean", "mean"),
        mean_unselected_importance_mean=("unselected_importance_mean", "mean"),
        mean_importance_gap_ratio=("importance_gap_mean_ratio", "mean"),
        mean_importance_selected_share=("importance_selected_share", "mean"),
        mean_layer_jaccard=("mask_jaccard_with_prev_round_layer", "mean"),
        mean_layer_hamming=("mask_hamming_change_rate_layer", "mean"),
    ).reset_index()
    summary_path = os.path.join(
        output_root,
        "results",
        "gc_head_8exp_E6_layer_importance_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def run_gc_head_8exp_compare(base_cfg, output_root, rc_lambda):
    specs = [
        {
            "experiment_id": "E1_fc1_tau0p05_noema",
            "head_prefixes": ["fc1"],
            "head_exact_names": [],
            "tau": 0.05,
            "use_ema": False,
            "ema_beta": None,
        },
        {
            "experiment_id": "E2_fc1_tau0p05_ema0p8",
            "head_prefixes": ["fc1"],
            "head_exact_names": [],
            "tau": 0.05,
            "use_ema": True,
            "ema_beta": 0.8,
        },
        {
            "experiment_id": "E3_fc1_tau0p30_noema",
            "head_prefixes": ["fc1"],
            "head_exact_names": [],
            "tau": 0.30,
            "use_ema": False,
            "ema_beta": None,
        },
        {
            "experiment_id": "E4_fc1weight_tau0p05_noema",
            "head_prefixes": [],
            "head_exact_names": ["fc1.weight"],
            "tau": 0.05,
            "use_ema": False,
            "ema_beta": None,
        },
        {
            "experiment_id": "E5_fc1weight_tau0p05_ema0p8",
            "head_prefixes": [],
            "head_exact_names": ["fc1.weight"],
            "tau": 0.05,
            "use_ema": True,
            "ema_beta": 0.8,
        },
        {
            "experiment_id": "E6_attention_lstm2_fc1_tau0p10_noema",
            "head_prefixes": ["attention", "lstm2", "fc1"],
            "head_exact_names": [],
            "tau": 0.10,
            "use_ema": False,
            "ema_beta": None,
        },
        {
            "experiment_id": "E7_attention_tau0p10_noema",
            "head_prefixes": ["attention"],
            "head_exact_names": [],
            "tau": 0.10,
            "use_ema": False,
            "ema_beta": None,
        },
        {
            "experiment_id": "E8_lstm2_tau0p10_noema",
            "head_prefixes": ["lstm2"],
            "head_exact_names": [],
            "tau": 0.10,
            "use_ema": False,
            "ema_beta": None,
        },
    ]

    results = {}
    summary_rows = []
    per_client_rows = []
    e6_save_dir = None

    for spec in specs:
        cfg = copy.deepcopy(base_cfg)
        cfg.experiment.task_type = "single_target"
        gc_cfg = build_indirect_gc_cfg(cfg)

        gc_cfg.federated.aggregation_method = "fedavg"
        gc_cfg.federated.use_rc_regularization = True
        gc_cfg.federated.rc_lambda = rc_lambda
        gc_cfg.federated.use_head_personalization = True
        gc_cfg.federated.head_param_prefixes = list(spec["head_prefixes"])
        gc_cfg.federated.head_param_exact_names = list(spec["head_exact_names"])
        gc_cfg.federated.head_personalization_tau = spec["tau"]
        gc_cfg.federated.head_personalization_warmup_rounds = 5
        gc_cfg.federated.head_mask_update_interval = 1
        gc_cfg.federated.use_head_importance_ema = spec["use_ema"]
        gc_cfg.federated.head_importance_ema_beta = (
            0.8 if spec["ema_beta"] is None else spec["ema_beta"]
        )

        experiment_name = f"gc_head8_{spec['experiment_id']}_lambda{float_tag(rc_lambda)}"
        configure_federated_dirs(gc_cfg, output_root, experiment_name)

        result = train_federated_model(
            gc_cfg,
            save_dir=gc_cfg.federated.save_dir,
            run_label=f"GC Head 8Exp {spec['experiment_id']}",
        )
        result["save_dir"] = gc_cfg.federated.save_dir
        result["checkpoint_dir"] = gc_cfg.federated.checkpoint_dir
        results[spec["experiment_id"]] = result

        summary_rows.append(build_gc_head_8exp_summary_row(
            spec,
            experiment_name,
            result,
            rc_lambda,
        ))
        per_client_rows.append(build_gc_head_8exp_per_client_row(spec, result))

        if spec["experiment_id"] == "E6_attention_lstm2_fc1_tau0p10_noema":
            e6_save_dir = gc_cfg.federated.save_dir

        summary_df = pd.DataFrame(summary_rows).sort_values("avg_client_RMSE").reset_index(drop=True)
        summary_df.insert(0, "rank_by_avg_client_RMSE", range(1, len(summary_df) + 1))
        compare_dir = os.path.join(output_root, "results")
        ensure_dir(compare_dir)
        summary_df.to_csv(
            os.path.join(compare_dir, "gc_head_8exp_final_error_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(per_client_rows).to_csv(
            os.path.join(compare_dir, "gc_head_8exp_per_client_error_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    compare_dir = os.path.join(output_root, "results")
    final_summary_path = os.path.join(compare_dir, "gc_head_8exp_final_error_summary.csv")
    per_client_summary_path = os.path.join(compare_dir, "gc_head_8exp_per_client_error_summary.csv")
    e6_layer_summary_path = save_e6_layer_importance_summary(e6_save_dir, output_root) if e6_save_dir else None

    return {
        "results": results,
        "summary_df": pd.read_csv(final_summary_path),
        "final_summary_path": final_summary_path,
        "per_client_summary_path": per_client_summary_path,
        "e6_layer_summary_path": e6_layer_summary_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run indirect net-load experiment suite.")
    parser.add_argument(
        "--experiment",
        choices=[
            "all",
            "local",
            "local_gc",
            "fedavg",
            "fedub",
            "fedavg_rc_compare_gc",
            "gc_head_compare",
            "gc_fc1_head_tau01_warmup5",
            "gc_fc1_head_tau01_warmup5_ema03",
            "gc_fc1_head_tau01_warmup5_maskint2",
            "gc_head_8exp_compare",
            "gc_price_fc1_tau010_ema03",
        ],
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(RUNS_DIR, "net_load_indirect_study"),
        help="Root folder containing results/ and checkpoints/.",
    )
    parser.add_argument(
        "--rc-lambda",
        type=float,
        default=None,
        help="RC regularization coefficient. Defaults to 1.0 for GC-only compare experiments and 10.0 for legacy FedUB net-load.",
    )
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--rounds", type=int, default=None, help="Override federated rounds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override local per-client epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    args = parser.parse_args()

    cfg = copy.deepcopy(CFG)
    cfg.experiment.task_type = "net_load"
    cfg.experiment.net_load_method = "indirect"
    if args.rounds is not None:
        cfg.federated.rounds = args.rounds
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.experiment in ["all", "local", "fedavg", "fedub"]:
        patience = 3 if args.patience is None else args.patience
        cfg.train.early_stop_patience = patience
        cfg.federated.early_stop_patience = patience
    elif args.patience is not None:
        cfg.train.early_stop_patience = args.patience
        cfg.federated.early_stop_patience = args.patience
    set_seed(cfg.train.random_seed)
    ensure_dir(args.output_root)
    ensure_dir(os.path.join(args.output_root, "results"))
    ensure_dir(os.path.join(args.output_root, "checkpoints"))

    result = None
    legacy_rc_lambda = 10.0 if args.rc_lambda is None else args.rc_lambda
    gc_rc_lambda = 1.0 if args.rc_lambda is None else args.rc_lambda
    if args.experiment == "all":
        run_local_indirect_net_load(cfg, args.output_root)
        run_fedavg_baseline(cfg, args.output_root)
        result = run_fedub_rc(cfg, args.output_root, rc_lambda=legacy_rc_lambda, tau=args.tau)
    elif args.experiment == "local":
        result = run_local_indirect_net_load(cfg, args.output_root)
    elif args.experiment == "local_gc":
        result_dir, checkpoint_dir = build_output_dirs(args.output_root, "local_per_client_gc")
        result = run_local_component(
            cfg,
            component_name="gc",
            result_dir=result_dir,
            checkpoint_dir=checkpoint_dir,
            title_prefix="Local Per-Client GC",
        )
    elif args.experiment == "fedavg":
        result = run_fedavg_baseline(cfg, args.output_root)
    elif args.experiment == "fedub":
        result = run_fedub_rc(cfg, args.output_root, rc_lambda=legacy_rc_lambda, tau=args.tau)
    elif args.experiment == "fedavg_rc_compare_gc":
        if args.epochs is not None:
            cfg.federated.local_epochs = args.epochs
        result = run_fedavg_rc_compare_gc(
            base_cfg=cfg,
            output_root=args.output_root,
            rc_lambda=gc_rc_lambda,
        )
    elif args.experiment == "gc_head_compare":
        if args.epochs is not None:
            cfg.federated.local_epochs = args.epochs
        result = run_gc_head_personalization_compare(
            base_cfg=cfg,
            output_root=args.output_root,
            rc_lambda=gc_rc_lambda,
        )
    elif args.experiment in ["gc_fc1_head_tau01_warmup5", "gc_fc1_head_tau01_warmup5_ema03"]:
        if args.epochs is not None:
            cfg.federated.local_epochs = args.epochs
        result = run_gc_fc1_head_tau01_warmup5(
            base_cfg=cfg,
            output_root=args.output_root,
            rc_lambda=gc_rc_lambda,
        )
    elif args.experiment == "gc_fc1_head_tau01_warmup5_maskint2":
        if args.epochs is not None:
            cfg.federated.local_epochs = args.epochs
        result = run_gc_fc1_head_tau01_warmup5_maskint2(
            base_cfg=cfg,
            output_root=args.output_root,
            rc_lambda=gc_rc_lambda,
        )
    elif args.experiment == "gc_head_8exp_compare":
        if args.epochs is not None:
            cfg.federated.local_epochs = args.epochs
        result = run_gc_head_8exp_compare(
            base_cfg=cfg,
            output_root=args.output_root,
            rc_lambda=gc_rc_lambda,
        )
    elif args.experiment == "gc_price_fc1_tau010_ema03":
        if args.epochs is not None:
            cfg.federated.local_epochs = args.epochs
        result = run_gc_price_fc1_tau010_ema03(
            base_cfg=cfg,
            output_root=args.output_root,
            rc_lambda=gc_rc_lambda,
        )


if __name__ == "__main__":
    main()
