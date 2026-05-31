import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
        "R2": float(r2),
    }


def plot_round_curve(values, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_true_pred(y_true, y_pred, save_path, title="真实值与预测值对比", show_n=300):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    show_n = min(show_n, len(y_true))

    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:show_n], label="真实值")
    plt.plot(y_pred[:show_n], label="预测值")
    plt.title(title)
    plt.xlabel("样本点")
    plt.ylabel("负荷")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_metrics_csv(metrics: dict, save_path: str):
    pd.DataFrame([metrics]).to_csv(save_path, index=False, encoding="utf-8-sig")


def print_metrics(metrics: dict, title: str = "Metrics"):
    print(f"\n[{title}]")
    for k, v in metrics.items():
        if "percent" in k.lower():
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.6f}")
