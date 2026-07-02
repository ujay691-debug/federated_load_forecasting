"""Compare metrics and Shape/Residual shares across experiment run directories."""

import argparse
import os
from typing import Dict, Iterable, Optional

import pandas as pd


METRIC_COLUMNS = ["MAE", "MSE", "RMSE", "MAPE_percent", "R2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare residual predictor experiment results."
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _find_component_row(
    component_df: pd.DataFrame, candidates: Iterable[str]
) -> Optional[pd.Series]:
    if "component" not in component_df.columns:
        return None
    component_names = component_df["component"].astype(str)
    for candidate in candidates:
        matches = component_df.loc[component_names == candidate]
        if not matches.empty:
            return matches.iloc[0]
    return None


def read_run(name: str, run_dir: str) -> Dict:
    run_dir = os.path.abspath(run_dir)
    metrics_path = os.path.join(run_dir, "test_metrics.csv")
    components_path = os.path.join(run_dir, "test_component_summary.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Missing test metrics: {metrics_path}")
    if not os.path.isfile(components_path):
        raise FileNotFoundError(f"Missing component summary: {components_path}")

    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        raise ValueError(f"No rows in {metrics_path}")
    metrics = metrics_df.iloc[0]
    missing_metrics = [col for col in METRIC_COLUMNS if col not in metrics_df.columns]
    if missing_metrics:
        raise ValueError(f"{metrics_path} is missing columns: {missing_metrics}")

    component_df = pd.read_csv(components_path)
    s_row = _find_component_row(component_df, ["S_dev", "S", "S_shift"])
    r_row = _find_component_row(
        component_df, ["R_dev", "Rw_dev", "R", "Rw", "R_shift", "Rw_shift"]
    )
    if s_row is None:
        raise ValueError(f"Cannot find Shape component in {components_path}")
    if r_row is None:
        raise ValueError(f"Cannot find Residual/Rw component in {components_path}")
    for column in ("share_abs", "share_energy"):
        if column not in component_df.columns:
            raise ValueError(f"{components_path} is missing column: {column}")

    return {
        "name": name,
        "run_dir": run_dir,
        **{column: float(metrics[column]) for column in METRIC_COLUMNS},
        "S_share_abs": float(s_row["share_abs"]),
        "R_share_abs": float(r_row["share_abs"]),
        "S_share_energy": float(s_row["share_energy"]),
        "R_share_energy": float(r_row["share_energy"]),
    }


def main():
    args = parse_args()
    if args.names is not None and len(args.names) != len(args.runs):
        raise ValueError(
            f"--names has {len(args.names)} values but --runs has {len(args.runs)}."
        )
    names = args.names or [os.path.basename(os.path.normpath(path)) for path in args.runs]
    rows = [read_run(name, run_dir) for name, run_dir in zip(names, args.runs)]
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()
