from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "runs" / "net_load_indirect_study_v2" / "results"
LOCAL_ROOT = RESULTS_ROOT / "local_per_client_indirect_net_load"
FEDAVG_ROOT = RESULTS_ROOT / "fedavg_baseline_indirect_net_load" / "indirect_net_load"
OUT_DIR = RESULTS_ROOT / "fedavg_vs_local_client_analysis"

METHOD_ROOTS = {
    "FedAvg": FEDAVG_ROOT,
    "Local": LOCAL_ROOT,
}

TARGETS = {
    "gc": {
        "label": "GC",
        "subdir": Path("gc_model"),
        "ylabel": "gc",
    },
    "gg": {
        "label": "GG",
        "subdir": Path("gg_model"),
        "ylabel": "gg",
    },
    "net_load": {
        "label": "Net load",
        "subdir": Path("."),
        "ylabel": "gc - gg",
    },
}

PLOT_COLORS = {
    "true": "#202124",
    "FedAvg": "#1f77b4",
    "Local": "#ff7f0e",
}


def read_metrics(method: str, target: str) -> pd.DataFrame:
    root = METHOD_ROOTS[method]
    subdir = TARGETS[target]["subdir"]
    path = root / subdir / "per_client_results" / "all_clients_test_metrics_summary.csv"
    df = pd.read_csv(path)
    df["method"] = method
    df["target"] = target
    keep = [
        "target",
        "method",
        "client_id",
        "client_name",
        "MAE",
        "MSE",
        "RMSE",
        "MAPE_percent",
        "R2",
    ]
    return df[[col for col in keep if col in df.columns]]


def read_prediction(method: str, target: str, client_id: int) -> pd.DataFrame:
    root = METHOD_ROOTS[method]
    subdir = TARGETS[target]["subdir"]
    path = (
        root
        / subdir
        / "per_client_results"
        / f"client_{client_id}"
        / f"client_{client_id}_test_predictions.csv"
    )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "y_true_step_1" not in df.columns or "y_pred_step_1" not in df.columns:
        raise ValueError(f"Unsupported prediction columns in {path}")
    return df[["timestamp", "y_true_step_1", "y_pred_step_1"]].rename(
        columns={
            "y_true_step_1": "y_true",
            "y_pred_step_1": f"y_pred_{method}",
        }
    )


def build_metrics_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_metrics = pd.concat(
        [
            read_metrics(method, target)
            for target in TARGETS
            for method in METHOD_ROOTS
        ],
        ignore_index=True,
    )

    wide = None
    for method in METHOD_ROOTS:
        method_df = all_metrics[all_metrics["method"] == method].copy()
        rename_cols = {
            metric: f"{method}_{metric}"
            for metric in ["MAE", "MSE", "RMSE", "MAPE_percent", "R2"]
            if metric in method_df.columns
        }
        method_df = method_df.rename(columns=rename_cols)
        method_df = method_df.drop(columns=["method"])
        wide = method_df if wide is None else wide.merge(
            method_df,
            on=["target", "client_id", "client_name"],
            how="inner",
        )

    for metric in ["MAE", "MSE", "RMSE", "MAPE_percent"]:
        wide[f"delta_Local_minus_FedAvg_{metric}"] = (
            wide[f"Local_{metric}"] - wide[f"FedAvg_{metric}"]
        )
        wide[f"delta_pct_Local_minus_FedAvg_{metric}"] = np.where(
            wide[f"FedAvg_{metric}"].abs() > 1e-12,
            wide[f"delta_Local_minus_FedAvg_{metric}"] / wide[f"FedAvg_{metric}"] * 100.0,
            np.nan,
        )
    wide["delta_Local_minus_FedAvg_R2"] = wide["Local_R2"] - wide["FedAvg_R2"]
    wide["rmse_winner"] = np.where(
        wide["delta_Local_minus_FedAvg_RMSE"] < 0,
        "Local",
        np.where(wide["delta_Local_minus_FedAvg_RMSE"] > 0, "FedAvg", "Tie"),
    )
    wide["practical_rmse_winner_1pct"] = np.select(
        [
            wide["delta_pct_Local_minus_FedAvg_RMSE"] < -1.0,
            wide["delta_pct_Local_minus_FedAvg_RMSE"] > 1.0,
        ],
        ["Local", "FedAvg"],
        default="Tie",
    )

    summary_rows = []
    for target, group in wide.groupby("target", sort=False):
        local_better = group[group["delta_Local_minus_FedAvg_RMSE"] < 0]
        fedavg_better = group[group["delta_Local_minus_FedAvg_RMSE"] > 0]
        ties_1pct = group[group["practical_rmse_winner_1pct"] == "Tie"]
        summary_rows.append(
            {
                "target": target,
                "target_label": TARGETS[target]["label"],
                "FedAvg_mean_RMSE": group["FedAvg_RMSE"].mean(),
                "Local_mean_RMSE": group["Local_RMSE"].mean(),
                "mean_delta_Local_minus_FedAvg_RMSE": (
                    group["Local_RMSE"].mean() - group["FedAvg_RMSE"].mean()
                ),
                "mean_delta_pct_Local_minus_FedAvg_RMSE": (
                    (group["Local_RMSE"].mean() - group["FedAvg_RMSE"].mean())
                    / group["FedAvg_RMSE"].mean()
                    * 100.0
                ),
                "FedAvg_mean_R2": group["FedAvg_R2"].mean(),
                "Local_mean_R2": group["Local_R2"].mean(),
                "local_better_clients_exact": ",".join(
                    map(str, local_better["client_id"].astype(int).tolist())
                ),
                "fedavg_better_clients_exact": ",".join(
                    map(str, fedavg_better["client_id"].astype(int).tolist())
                ),
                "local_better_count_exact": int(len(local_better)),
                "fedavg_better_count_exact": int(len(fedavg_better)),
                "tie_within_1pct_clients": ",".join(
                    map(str, ties_1pct["client_id"].astype(int).tolist())
                ),
                "tie_within_1pct_count": int(len(ties_1pct)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    client_signal_rows = []
    for client_id, group in wide.groupby("client_id"):
        group = group.sort_values("target")
        local_gt1 = group[group["delta_pct_Local_minus_FedAvg_RMSE"] < -1.0]
        fedavg_gt1 = group[group["delta_pct_Local_minus_FedAvg_RMSE"] > 1.0]
        max_local_improvement = -group["delta_pct_Local_minus_FedAvg_RMSE"].min()
        mean_change = group["delta_pct_Local_minus_FedAvg_RMSE"].mean()
        if len(local_gt1) >= 2 or max_local_improvement >= 10.0:
            recommendation = "strong_personalization_candidate"
        elif len(fedavg_gt1) >= 2 and len(local_gt1) == 0:
            recommendation = "fedavg_sufficient_or_better"
        else:
            recommendation = "mixed_or_near_tie"
        client_signal_rows.append(
            {
                "client_id": int(client_id),
                "client_name": group["client_name"].iloc[0],
                "local_better_targets_gt1pct": ",".join(local_gt1["target"].tolist()),
                "fedavg_better_targets_gt1pct": ",".join(fedavg_gt1["target"].tolist()),
                "local_better_count_gt1pct": int(len(local_gt1)),
                "fedavg_better_count_gt1pct": int(len(fedavg_gt1)),
                "mean_rmse_change_pct_Local_minus_FedAvg": mean_change,
                "max_local_improvement_pct": max_local_improvement,
                "recommendation": recommendation,
            }
        )
    client_signal = pd.DataFrame(client_signal_rows)

    return all_metrics, wide.sort_values(["target", "client_id"]), summary, client_signal


def build_prediction_stats() -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for client_id in range(1, 10):
            for method in METHOD_ROOTS:
                pred = read_prediction(method, target, client_id)
                err = pred[f"y_pred_{method}"] - pred["y_true"]
                rows.append(
                    {
                        "target": target,
                        "method": method,
                        "client_id": client_id,
                        "n": int(len(pred)),
                        "true_mean": pred["y_true"].mean(),
                        "true_std": pred["y_true"].std(ddof=0),
                        "true_min": pred["y_true"].min(),
                        "true_max": pred["y_true"].max(),
                        "pred_mean": pred[f"y_pred_{method}"].mean(),
                        "bias_pred_minus_true": err.mean(),
                        "abs_bias": err.mean().__abs__(),
                        "error_std": err.std(ddof=0),
                        "corr": pred["y_true"].corr(pred[f"y_pred_{method}"]),
                    }
                )
    return pd.DataFrame(rows)


def build_regional_comparison() -> pd.DataFrame:
    rows = []
    for method, root in METHOD_ROOTS.items():
        component_path = root / "indirect_net_load_component_compare.csv"
        df = pd.read_csv(component_path)
        for _, row in df.iterrows():
            rows.append(
                {
                    "method": method,
                    "component": row["component"],
                    "MAE": row["MAE"],
                    "MSE": row["MSE"],
                    "RMSE": row["RMSE"],
                    "MAPE_percent": row["MAPE_percent"],
                    "R2": row["R2"],
                }
            )
    regional = pd.DataFrame(rows)
    fed = regional[regional["method"] == "FedAvg"].set_index("component")
    loc = regional[regional["method"] == "Local"].set_index("component")
    comp_rows = []
    for component in fed.index.intersection(loc.index):
        comp_rows.append(
            {
                "component": component,
                "FedAvg_RMSE": fed.loc[component, "RMSE"],
                "Local_RMSE": loc.loc[component, "RMSE"],
                "delta_Local_minus_FedAvg_RMSE": loc.loc[component, "RMSE"]
                - fed.loc[component, "RMSE"],
                "delta_pct_Local_minus_FedAvg_RMSE": (
                    (loc.loc[component, "RMSE"] - fed.loc[component, "RMSE"])
                    / fed.loc[component, "RMSE"]
                    * 100.0
                ),
                "FedAvg_R2": fed.loc[component, "R2"],
                "Local_R2": loc.loc[component, "R2"],
                "delta_Local_minus_FedAvg_R2": loc.loc[component, "R2"]
                - fed.loc[component, "R2"],
            }
        )
    return pd.DataFrame(comp_rows)


def plot_rmse_bars(wide: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=False)
    for ax, target in zip(axes, TARGETS):
        group = wide[wide["target"] == target].sort_values("client_id")
        x = np.arange(len(group))
        width = 0.38
        ax.bar(x - width / 2, group["FedAvg_RMSE"], width, label="FedAvg", color=PLOT_COLORS["FedAvg"])
        ax.bar(x + width / 2, group["Local_RMSE"], width, label="Local", color=PLOT_COLORS["Local"])
        ax.set_title(f"{TARGETS[target]['label']} RMSE by client")
        ax.set_xlabel("Client")
        ax.set_xticks(x)
        ax.set_xticklabels(group["client_id"].astype(str).tolist())
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmse_by_client_fedavg_vs_local.png", dpi=180)
    plt.close(fig)


def plot_delta_heatmap(wide: pd.DataFrame) -> None:
    pivot = wide.pivot(
        index="target",
        columns="client_id",
        values="delta_pct_Local_minus_FedAvg_RMSE",
    ).loc[list(TARGETS.keys())]
    data = pivot.to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanmax(np.abs(data))))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([TARGETS[t]["label"] for t in pivot.index])
    ax.set_xlabel("Client")
    ax.set_title("RMSE change: Local minus FedAvg (%)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i, j]:+.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(data[i, j]) > vmax * 0.55 else "#202124",
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("negative = Local better")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmse_delta_pct_heatmap.png", dpi=180)
    plt.close(fig)


def plot_prediction_overlays(show_points: int = 240) -> None:
    for target in TARGETS:
        fig, axes = plt.subplots(3, 3, figsize=(17, 10), sharex=True)
        for client_id, ax in enumerate(axes.ravel(), start=1):
            fed = read_prediction("FedAvg", target, client_id)
            loc = read_prediction("Local", target, client_id)
            merged = fed.merge(
                loc[["timestamp", "y_pred_Local"]],
                on="timestamp",
                how="inner",
            ).head(show_points)
            x = np.arange(len(merged))
            ax.plot(x, merged["y_true"], color=PLOT_COLORS["true"], linewidth=1.5, label="Actual")
            ax.plot(x, merged["y_pred_FedAvg"], color=PLOT_COLORS["FedAvg"], linewidth=1.0, alpha=0.9, label="FedAvg")
            ax.plot(x, merged["y_pred_Local"], color=PLOT_COLORS["Local"], linewidth=1.0, alpha=0.9, label="Local")
            ax.set_title(f"Client {client_id}")
            ax.grid(alpha=0.2)
            if client_id in [1, 4, 7]:
                ax.set_ylabel(TARGETS[target]["ylabel"])
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            frameon=False,
        )
        fig.suptitle(
            f"{TARGETS[target]['label']}: actual vs prediction, first {show_points} test points",
            y=0.99,
        )
        fig.supxlabel("Test sample index")
        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        fig.savefig(OUT_DIR / f"prediction_overlay_{target}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics, wide, summary, client_signal = build_metrics_tables()
    prediction_stats = build_prediction_stats()
    regional = build_regional_comparison()

    all_metrics.to_csv(OUT_DIR / "metrics_long.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT_DIR / "per_client_metrics_comparison.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "summary_by_target.csv", index=False, encoding="utf-8-sig")
    client_signal.to_csv(OUT_DIR / "client_personalization_signal.csv", index=False, encoding="utf-8-sig")
    prediction_stats.to_csv(OUT_DIR / "prediction_error_stats.csv", index=False, encoding="utf-8-sig")
    regional.to_csv(OUT_DIR / "regional_component_comparison.csv", index=False, encoding="utf-8-sig")

    plot_rmse_bars(wide)
    plot_delta_heatmap(wide)
    plot_prediction_overlays()

    print(f"Wrote analysis outputs to: {OUT_DIR}")
    print(summary[[
        "target",
        "FedAvg_mean_RMSE",
        "Local_mean_RMSE",
        "mean_delta_pct_Local_minus_FedAvg_RMSE",
        "local_better_count_exact",
        "fedavg_better_count_exact",
    ]].to_string(index=False))
    print()
    print(client_signal[[
        "client_id",
        "mean_rmse_change_pct_Local_minus_FedAvg",
        "max_local_improvement_pct",
        "recommendation",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
