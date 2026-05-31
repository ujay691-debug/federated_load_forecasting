# -*- coding: utf-8 -*-
"""
将 AEMO NSW1 RRP 半小时电价数据合并到 9 个客户端负荷气象数据中。

前提：
1. per_client_merged/client_x_load_weather_30min.csv 中 timestamp 是 UTC 时间
2. timestamp 表示半小时区间结束时刻
3. AEMO 电价文件中 datetime_utc_end 也是 UTC 区间结束时刻
4. 合并后覆盖 per_client_merged 中原文件，文件名不变
"""

from pathlib import Path
import shutil
import pandas as pd


# =========================
# 1. 路径配置
# =========================

PROJECT_ROOT = Path(".")  # 在 federated_load_forecasting 根目录运行即可

CLIENT_DIR = PROJECT_ROOT / "per_client_merged"

PRICE_PATH = (
    PROJECT_ROOT
    / "aemo_nsw1_rrp_2010_2013"
    / "AEMO_NSW1_RRP_2010-07-01_to_2013-06-30_30min_UTC.csv"
)

BACKUP_DIR = PROJECT_ROOT / "per_client_merged_backup_before_rrp"

CLIENT_PATTERN = "client_*_load_weather_30min.csv"

CLIENT_TIME_COL = "timestamp"

# 因为你的负荷 timestamp 是区间结束时刻，所以用 datetime_utc_end 合并
PRICE_TIME_COL = "datetime_utc_end"

PRICE_KEEP_COLS = [
    "rrp_aud_per_mwh",
    "total_demand_mw",
]


# =========================
# 2. 工具函数
# =========================

def read_price_data(price_path: Path) -> pd.DataFrame:
    if not price_path.exists():
        raise FileNotFoundError(f"找不到电价文件: {price_path}")

    price = pd.read_csv(price_path)

    if PRICE_TIME_COL not in price.columns:
        raise ValueError(
            f"电价文件中找不到 {PRICE_TIME_COL} 列。"
            f"当前列名为: {list(price.columns)}"
        )

    missing_cols = [c for c in PRICE_KEEP_COLS if c not in price.columns]
    if missing_cols:
        raise ValueError(
            f"电价文件缺少需要合并的列: {missing_cols}。"
            f"当前列名为: {list(price.columns)}"
        )

    # AEMO 电价文件已经是 UTC，这里统一解析为 UTC
    price[PRICE_TIME_COL] = pd.to_datetime(
        price[PRICE_TIME_COL],
        utc=True,
        errors="coerce"
    )

    price = price.dropna(subset=[PRICE_TIME_COL]).copy()

    # 防止重复时间戳导致 merge 后行数膨胀
    dup_count = price[PRICE_TIME_COL].duplicated().sum()
    if dup_count > 0:
        print(f"警告：电价数据存在重复 {PRICE_TIME_COL}: {dup_count} 条，保留最后一条。")
        price = price.drop_duplicates(subset=[PRICE_TIME_COL], keep="last")

    keep_cols = [PRICE_TIME_COL] + PRICE_KEEP_COLS
    price = price[keep_cols].copy()

    print("电价数据读取完成")
    print(f"电价时间范围: {price[PRICE_TIME_COL].min()} 到 {price[PRICE_TIME_COL].max()}")
    print(f"电价样本数: {len(price)}")

    return price


def backup_client_files(client_files):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in client_files:
        backup_path = BACKUP_DIR / file_path.name
        shutil.copy2(file_path, backup_path)

    print(f"已备份原客户端文件到: {BACKUP_DIR}")


def merge_one_client(client_path: Path, price: pd.DataFrame):
    print(f"\n开始处理: {client_path.name}")

    df = pd.read_csv(client_path)

    if CLIENT_TIME_COL not in df.columns:
        raise ValueError(
            f"{client_path.name} 中找不到时间列 {CLIENT_TIME_COL}。"
            f"当前列名为: {list(df.columns)}"
        )

    original_rows = len(df)
    original_timestamp = df[CLIENT_TIME_COL].copy()

    # 如果文件里已经有电价列，先删除，避免生成 rrp_aud_per_mwh_x / _y
    drop_cols = [c for c in PRICE_KEEP_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"已删除旧电价列: {drop_cols}")

    # 你的 timestamp 已经是 UTC。若是无时区字符串，utc=True 会按 UTC 解析。
    df["_merge_timestamp_utc"] = pd.to_datetime(
        df[CLIENT_TIME_COL],
        utc=True,
        errors="coerce"
    )

    bad_time_count = df["_merge_timestamp_utc"].isna().sum()
    if bad_time_count > 0:
        print(f"警告：{client_path.name} 有 {bad_time_count} 行 timestamp 无法解析。")

    print(
        f"客户端时间范围: "
        f"{df['_merge_timestamp_utc'].min()} 到 {df['_merge_timestamp_utc'].max()}"
    )

    merged = df.merge(
        price,
        left_on="_merge_timestamp_utc",
        right_on=PRICE_TIME_COL,
        how="left"
    )

    # 删除辅助合并列
    merged = merged.drop(columns=["_merge_timestamp_utc", PRICE_TIME_COL])

    # 保持原始 timestamp 文本格式不变
    merged[CLIENT_TIME_COL] = original_timestamp

    if len(merged) != original_rows:
        raise RuntimeError(
            f"{client_path.name} 合并前后行数不一致: "
            f"{original_rows} -> {len(merged)}。请检查电价时间是否重复。"
        )

    missing_rrp = merged["rrp_aud_per_mwh"].isna().sum()
    missing_rate = missing_rrp / len(merged)

    print(f"合并后行数: {len(merged)}")
    print(f"RRP 缺失数量: {missing_rrp}")
    print(f"RRP 缺失比例: {missing_rate:.4%}")

    # 覆盖原文件，名称不变
    merged.to_csv(client_path, index=False, encoding="utf-8-sig")

    print(f"已覆盖保存: {client_path}")


def main():
    if not CLIENT_DIR.exists():
        raise FileNotFoundError(f"找不到客户端文件夹: {CLIENT_DIR}")

    client_files = sorted(CLIENT_DIR.glob(CLIENT_PATTERN))

    if not client_files:
        raise FileNotFoundError(
            f"在 {CLIENT_DIR} 中没有找到匹配文件: {CLIENT_PATTERN}"
        )

    print("找到客户端文件:")
    for p in client_files:
        print(f"  {p}")

    price = read_price_data(PRICE_PATH)

    # 先备份，再覆盖
    backup_client_files(client_files)

    for client_path in client_files:
        merge_one_client(client_path, price)

    print("\n全部客户端合并完成。")
    print("合并方式：client timestamp UTC 区间结束时刻 对齐 AEMO datetime_utc_end。")


if __name__ == "__main__":
    main()