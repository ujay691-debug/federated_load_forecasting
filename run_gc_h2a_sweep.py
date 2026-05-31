import copy
import os
import traceback
import pandas as pd

from config import CFG, RUNS_DIR
from federated_main import train_federated_model
from utils.data_utils import ensure_dir, set_seed


def make_base_gc_cfg():
    cfg = copy.deepcopy(CFG)

    cfg.experiment.task_type = "single_target"
    cfg.data.target_col = "gc"

    if hasattr(cfg, "gc_feature"):
        cfg.feature = copy.deepcopy(cfg.gc_feature)

    cfg.federated.rounds = 20
    cfg.federated.local_epochs = 1
    cfg.federated.client_fraction = 1.0
    cfg.federated.eval_every = 1
    cfg.federated.early_stop_patience = 6

    cfg.federated.use_rc_regularization = False
    cfg.federated.rc_lambda = 0.0
    cfg.federated.use_head_personalization = False
    cfg.federated.head_personalization_tau = 0.0

    cfg.federated.checkpoint_dir = None
    return cfg


EXPERIMENTS = [
    {
        "name": "01_fedavg_same_setting",
        "aggregation_method": "fedavg",
    },
    {
        "name": "02_h2a_refs9_attn_feature",
        "aggregation_method": "h2a",
        "h2a_num_refs": 9,
        "h2a_gamma": 0.1,
        "h2a_meta_lr": 1e-3,
        "h2a_warmup_use_all_clients": True,
        "h2a_feature_param_prefixes": ["conv1", "conv2", "lstm1", "lstm2", "attention"],
        "h2a_head_param_prefixes": ["fc1", "fc2"],
    },
    {
        "name": "03_h2a_refs5_attn_feature",
        "aggregation_method": "h2a",
        "h2a_num_refs": 5,
        "h2a_gamma": 0.1,
        "h2a_meta_lr": 1e-3,
        "h2a_warmup_use_all_clients": True,
        "h2a_feature_param_prefixes": ["conv1", "conv2", "lstm1", "lstm2", "attention"],
        "h2a_head_param_prefixes": ["fc1", "fc2"],
    },
    {
        "name": "04_h2a_refs3_attn_feature",
        "aggregation_method": "h2a",
        "h2a_num_refs": 3,
        "h2a_gamma": 0.1,
        "h2a_meta_lr": 1e-3,
        "h2a_warmup_use_all_clients": True,
        "h2a_feature_param_prefixes": ["conv1", "conv2", "lstm1", "lstm2", "attention"],
        "h2a_head_param_prefixes": ["fc1", "fc2"],
    },
    {
        "name": "05_h2a_refs5_attn_head",
        "aggregation_method": "h2a",
        "h2a_num_refs": 5,
        "h2a_gamma": 0.1,
        "h2a_meta_lr": 1e-3,
        "h2a_warmup_use_all_clients": True,
        "h2a_feature_param_prefixes": ["conv1", "conv2", "lstm1", "lstm2"],
        "h2a_head_param_prefixes": ["attention", "fc1", "fc2"],
    },
    {
        "name": "06_h2a_refs5_fc2_only_head",
        "aggregation_method": "h2a",
        "h2a_num_refs": 5,
        "h2a_gamma": 0.1,
        "h2a_meta_lr": 1e-3,
        "h2a_warmup_use_all_clients": True,
        "h2a_feature_param_prefixes": ["conv1", "conv2", "lstm1", "lstm2", "attention", "fc1"],
        "h2a_head_param_prefixes": ["fc2"],
    },
]


def apply_experiment_cfg(cfg, exp):
    cfg.federated.aggregation_method = exp["aggregation_method"]
    cfg.federated.save_dir = os.path.join(RUNS_DIR, "gc_h2a_sweep", exp["name"])
    cfg.federated.checkpoint_dir = None

    if exp["aggregation_method"] == "fedavg":
        cfg.federated.use_rc_regularization = False
        cfg.federated.rc_lambda = 0.0
        cfg.federated.use_head_personalization = False

    if exp["aggregation_method"] == "h2a":
        cfg.federated.use_rc_regularization = False
        cfg.federated.rc_lambda = 0.0
        cfg.federated.use_head_personalization = False

        cfg.federated.h2a_num_refs = exp["h2a_num_refs"]
        cfg.federated.h2a_gamma = exp["h2a_gamma"]
        cfg.federated.h2a_meta_lr = exp["h2a_meta_lr"]
        cfg.federated.h2a_warmup_use_all_clients = exp["h2a_warmup_use_all_clients"]
        cfg.federated.h2a_feature_param_prefixes = list(exp["h2a_feature_param_prefixes"])
        cfg.federated.h2a_head_param_prefixes = list(exp["h2a_head_param_prefixes"])
        cfg.federated.h2a_unmatched_param_policy = "error"

    return cfg


def load_best_round_and_val_rmse(save_dir):
    log_path = os.path.join(save_dir, "federated_round_logs.csv")
    if not os.path.exists(log_path):
        return None, None

    df = pd.read_csv(log_path)
    if "avg_client_val_RMSE" not in df.columns:
        return None, None

    valid = df.dropna(subset=["avg_client_val_RMSE"])
    if len(valid) == 0:
        return None, None

    idx = valid["avg_client_val_RMSE"].idxmin()
    return int(valid.loc[idx, "round"]), float(valid.loc[idx, "avg_client_val_RMSE"])


def load_h2a_alpha_summary(save_dir):
    path = os.path.join(save_dir, "h2a_round_summary_logs.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    if len(df) == 0:
        return {}

    out = {}
    for col in ["h2a_alpha_mean", "h2a_alpha_min", "h2a_alpha_max", "h2a_distance_mean", "h2a_meta_loss_mean"]:
        if col in df.columns:
            out[f"last_{col}"] = float(df[col].dropna().iloc[-1]) if len(df[col].dropna()) > 0 else None
            out[f"mean_{col}"] = float(df[col].dropna().mean()) if len(df[col].dropna()) > 0 else None
    return out


def main():
    set_seed(CFG.train.random_seed)

    root_dir = os.path.join(RUNS_DIR, "gc_h2a_sweep")
    ensure_dir(root_dir)

    summary_path = os.path.join(root_dir, "gc_h2a_sweep_summary.csv")
    summary_rows = []

    for exp in EXPERIMENTS:
        print("\n" + "=" * 120)
        print(f"Running experiment: {exp['name']}")
        print("=" * 120)

        cfg = make_base_gc_cfg()
        cfg = apply_experiment_cfg(cfg, exp)

        try:
            result = train_federated_model(
                cfg,
                save_dir=cfg.federated.save_dir,
                run_label=f"Federated GC {exp['name']}",
            )

            regional = result["regional_test_metrics"]
            client_summary = result["client_test_summary_df"]
            best_round, best_val_rmse = load_best_round_and_val_rmse(cfg.federated.save_dir)

            row = {
                "experiment": exp["name"],
                "aggregation_method": exp["aggregation_method"],
                "save_dir": cfg.federated.save_dir,

                "rounds": cfg.federated.rounds,
                "local_epochs": cfg.federated.local_epochs,
                "early_stop_patience": cfg.federated.early_stop_patience,
                "best_round": best_round,
                "best_avg_client_val_RMSE": best_val_rmse,

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

            if exp["aggregation_method"] == "h2a":
                row.update({
                    "h2a_num_refs": exp["h2a_num_refs"],
                    "h2a_gamma": exp["h2a_gamma"],
                    "h2a_meta_lr": exp["h2a_meta_lr"],
                    "h2a_warmup_use_all_clients": exp["h2a_warmup_use_all_clients"],
                    "h2a_feature_param_prefixes": ",".join(exp["h2a_feature_param_prefixes"]),
                    "h2a_head_param_prefixes": ",".join(exp["h2a_head_param_prefixes"]),
                })
                row.update(load_h2a_alpha_summary(cfg.federated.save_dir))

            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"[OK] Finished experiment: {exp['name']}")
            print(f"[OK] Current summary saved to: {summary_path}")

        except Exception as exc:
            print(f"[ERROR] Experiment failed: {exp['name']}")
            traceback.print_exc()

            summary_rows.append({
                "experiment": exp["name"],
                "aggregation_method": exp["aggregation_method"],
                "save_dir": cfg.federated.save_dir,
                "error": repr(exc),
            })
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 120)
    print("All experiments finished.")
    print(f"Summary saved to: {summary_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()
