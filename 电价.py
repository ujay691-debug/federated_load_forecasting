# -*- coding: utf-8 -*-
"""
下载 AEMO NSW1 RRP 电价数据，并转换为 UTC 时间。

适用场景：
1. Ausgrid 三年数据窗口：2010-07-01 到 2013-06-30
2. AEMO SETTLEMENTDATE 按 NEM time 处理，即固定 UTC+10
3. 你的 9 个客户端数据已经是 UTC 时间
4. 可输出 30 min 版和 1 h 版
5. 可选批量合并到 client_1.csv ... client_9.csv

依赖：
pip install pandas requests
"""

from pathlib import Path
from datetime import timezone, timedelta
from typing import List, Optional

import time
import pandas as pd
import requests


# ============================================================
# 1. 用户配置区
# ============================================================

REGION = "NSW1"

# Ausgrid 三年时间范围，左闭右开
START_DATE_UTC = "2010-07-01 00:00:00"
END_DATE_UTC = "2013-07-01   00:00:00"

# AEMO NEM time 固定 UTC+10
NEM_TZ = timezone(timedelta(hours=10), name="NEM_TIME_UTC_PLUS_10")

# 输出目录
OUT_DIR = Path("aemo_nsw1_rrp_2010_2013")
RAW_DIR = OUT_DIR / "raw_monthly_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 是否把电价合并到 9 个客户端文件
MERGE_TO_CLIENTS = False

# 你的客户端文件目录
CLIENT_DIR = Path("clients")

# 假设客户端文件名为 client_1.csv ... client_9.csv
CLIENT_FILE_PATTERN = "client_{}.csv"

# 客户端 UTC 时间列名
CLIENT_UTC_COL = "datetime_utc"

# 合并时使用 AEMO 区间开始时刻还是结束时刻
# 如果你的客户端 datetime_utc 表示半小时区间开始，填 "start"
# 如果你的客户端 datetime_utc 表示半小时区间结束，填 "end"
MATCH_MODE = "start"  # "start" 或 "end"

# 合并后输出目录
MERGED_CLIENT_DIR = Path("clients_with_aemo_rrp")
MERGED_CLIENT_DIR.mkdir(parents=True, exist_ok=True)

# AEMO 常用下载路径
BASE_URLS = [
    "https://aemo.com.au/aemo/data/nem/priceanddemand/{filename}",
    "https://www.aemo.com.au/aemo/data/nem/priceanddemand/{filename}",
]


# ============================================================
# 2. 下载 AEMO 月度 CSV
# ============================================================

def make_month_list(start_utc: str, end_utc: str) -> List[pd.Timestamp]:
    """
    生成需要下载的月份列表。
    这里按 UTC 起止日期推导月份。
    对 2010-07 到 2013-06，最终下载 36 个月。
    """
    start = pd.to_datetime(start_utc).replace(day=1)
    end = pd.to_datetime(end_utc).replace(day=1)

    return list(pd.date_range(start=start, end=end, freq="MS", inclusive="left"))


def download_one_month(year: int, month: int, region: str) -> Optional[Path]:
    """
    下载单个月份的 AEMO PRICE_AND_DEMAND 文件。
    文件名格式：PRICE_AND_DEMAND_YYYYMM_NSW1.csv
    """
    yyyymm = f"{year}{month:02d}"
    filename = f"PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
    save_path = RAW_DIR / filename

    if save_path.exists() and save_path.stat().st_size > 200:
        print(f"[EXISTS] {filename}")
        return save_path

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for base_url in BASE_URLS:
        url = base_url.format(filename=filename)

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200 and len(response.content) > 200:
                head = response.content[:500].decode("utf-8", errors="ignore").upper()

                if "REGION" in head and "RRP" in head:
                    save_path.write_bytes(response.content)
                    print(f"[OK] {filename}")
                    return save_path

            print(f"[FAILED] {url}, status={response.status_code}")

        except Exception as e:
            print(f"[ERROR] {url}: {e}")

        time.sleep(0.5)

    print(f"[MISSING] {filename}")
    return None


def download_all_months() -> List[Path]:
    months = make_month_list(START_DATE_UTC, END_DATE_UTC)
    print(f"需要下载月份数: {len(months)}")

    paths = []
    for ts in months:
        path = download_one_month(ts.year, ts.month, REGION)
        if path is not None:
            paths.append(path)

    if len(paths) == 0:
        raise RuntimeError("没有成功下载任何 AEMO 文件，请检查网络或 AEMO 下载路径。")

    print(f"成功下载或读取文件数: {len(paths)}")
    return paths


# ============================================================
# 3. 读取并处理 AEMO 时间
# ============================================================

def read_one_aemo_csv(path: Path) -> pd.DataFrame:
    """
    读取单个月度 CSV，保留 REGION、SETTLEMENTDATE、RRP、TOTALDEMAND。
    """
    df = pd.read_csv(path)

    df.columns = [c.strip().upper() for c in df.columns]

    required = {"REGION", "SETTLEMENTDATE", "RRP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少必要字段: {missing}")

    df = df[df["REGION"].astype(str).str.upper() == REGION].copy()

    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
    df = df.dropna(subset=["SETTLEMENTDATE"])

    df["RRP"] = pd.to_numeric(df["RRP"], errors="coerce")

    if "TOTALDEMAND" in df.columns:
        df["TOTALDEMAND"] = pd.to_numeric(df["TOTALDEMAND"], errors="coerce")
    else:
        df["TOTALDEMAND"] = pd.NA

    keep_cols = ["REGION", "SETTLEMENTDATE", "RRP", "TOTALDEMAND"]

    if "PERIODTYPE" in df.columns:
        keep_cols.append("PERIODTYPE")

    return df[keep_cols]


def add_utc_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    AEMO SETTLEMENTDATE 按固定 UTC+10 的 NEM time 转 UTC。

    重要：
    这里不使用 Australia/Sydney。
    因为 AEMO 历史市场时间按 NEM time，即固定 UTC+10。
    """
    df = df.copy()

    # SETTLEMENTDATE 是 NEM time 下的半小时区间结束时刻
    df["datetime_nem_end"] = df["SETTLEMENTDATE"].dt.tz_localize(NEM_TZ)

    # 转 UTC
    df["datetime_utc_end"] = df["datetime_nem_end"].dt.tz_convert("UTC")

    # 对 30 min 数据，区间开始时刻 = 区间结束时刻 - 30 min
    df["datetime_utc_start"] = df["datetime_utc_end"] - pd.Timedelta(minutes=30)
    df["datetime_nem_start"] = df["datetime_nem_end"] - pd.Timedelta(minutes=30)

    return df


def build_aemo_30min(paths: List[Path]) -> pd.DataFrame:
    """
    合并 36 个月数据，并输出 30 min UTC 电价序列。
    """
    dfs = []
    for path in paths:
        one = read_one_aemo_csv(path)
        dfs.append(one)

    df = pd.concat(dfs, ignore_index=True)
    df = add_utc_time_columns(df)

    df = df.sort_values("datetime_utc_start")
    df = df.drop_duplicates(subset=["REGION", "datetime_utc_start"], keep="last")

    start_utc = pd.Timestamp(START_DATE_UTC, tz="UTC")
    end_utc = pd.Timestamp(END_DATE_UTC, tz="UTC")

    df = df[
        (df["datetime_utc_start"] >= start_utc) &
        (df["datetime_utc_start"] < end_utc)
    ].copy()

    df = df.rename(columns={
        "RRP": "rrp_aud_per_mwh",
        "TOTALDEMAND": "total_demand_mw"
    })

    cols = [
        "datetime_utc_start",
        "datetime_utc_end",
        "datetime_nem_start",
        "datetime_nem_end",
        "REGION",
        "rrp_aud_per_mwh",
        "total_demand_mw",
        "SETTLEMENTDATE"
    ]

    if "PERIODTYPE" in df.columns:
        cols.append("PERIODTYPE")

    df = df[cols].reset_index(drop=True)

    return df


# ============================================================
# 4. 生成 1 小时电价
# ============================================================

def build_aemo_1h(price_30min: pd.DataFrame) -> pd.DataFrame:
    """
    把 30 min RRP 聚合为 1 h。

    普通预测特征建议使用：
    rrp_aud_per_mwh_mean

    如果做市场成本分析，可参考：
    rrp_aud_per_mwh_demand_weighted
    """
    df = price_30min.copy()

    df["datetime_utc_hour_start"] = df["datetime_utc_start"].dt.floor("h")
    df["datetime_utc_hour_end"] = df["datetime_utc_hour_start"] + pd.Timedelta(hours=1)

    hourly = (
        df.groupby("datetime_utc_hour_start", as_index=False)
          .agg(
              rrp_aud_per_mwh_mean=("rrp_aud_per_mwh", "mean"),
              total_demand_mw_mean=("total_demand_mw", "mean"),
              n_halfhours=("rrp_aud_per_mwh", "count")
          )
    )

    hourly["datetime_utc_hour_end"] = hourly["datetime_utc_hour_start"] + pd.Timedelta(hours=1)

    # 需求加权小时电价
    valid = df.dropna(subset=["rrp_aud_per_mwh", "total_demand_mw"]).copy()

    if len(valid) > 0:
        valid["datetime_utc_hour_start"] = valid["datetime_utc_start"].dt.floor("h")
        valid["price_x_demand"] = valid["rrp_aud_per_mwh"] * valid["total_demand_mw"]

        weighted = (
            valid.groupby("datetime_utc_hour_start", as_index=False)
                 .agg(
                     price_x_demand_sum=("price_x_demand", "sum"),
                     demand_sum=("total_demand_mw", "sum")
                 )
        )

        weighted["rrp_aud_per_mwh_demand_weighted"] = (
            weighted["price_x_demand_sum"] / weighted["demand_sum"]
        )

        hourly = hourly.merge(
            weighted[["datetime_utc_hour_start", "rrp_aud_per_mwh_demand_weighted"]],
            on="datetime_utc_hour_start",
            how="left"
        )
    else:
        hourly["rrp_aud_per_mwh_demand_weighted"] = pd.NA

    hourly["REGION"] = REGION

    cols = [
        "datetime_utc_hour_start",
        "datetime_utc_hour_end",
        "REGION",
        "rrp_aud_per_mwh_mean",
        "rrp_aud_per_mwh_demand_weighted",
        "total_demand_mw_mean",
        "n_halfhours"
    ]

    return hourly[cols]


# ============================================================
# 5. 时间连续性检查
# ============================================================

def check_30min_continuity(price_30min: pd.DataFrame) -> None:
    """
    检查 30 min 时间序列是否完整。
    """
    start_utc = pd.Timestamp(START_DATE_UTC, tz="UTC")
    end_utc = pd.Timestamp(END_DATE_UTC, tz="UTC")

    expected_index = pd.date_range(
        start=start_utc,
        end=end_utc - pd.Timedelta(minutes=30),
        freq="30min"
    )

    actual_index = pd.DatetimeIndex(price_30min["datetime_utc_start"])

    missing = expected_index.difference(actual_index)
    duplicated_count = price_30min["datetime_utc_start"].duplicated().sum()

    print("\n========== 时间连续性检查 ==========")
    print(f"理论 30 min 样本数: {len(expected_index)}")
    print(f"实际 30 min 样本数: {len(price_30min)}")
    print(f"缺失时间点数量: {len(missing)}")
    print(f"重复时间点数量: {duplicated_count}")

    if len(missing) > 0:
        missing_path = OUT_DIR / "missing_30min_utc_timestamps.csv"
        pd.DataFrame({"missing_datetime_utc_start": missing}).to_csv(
            missing_path, index=False, encoding="utf-8-sig"
        )
        print(f"缺失时间点已保存: {missing_path}")


# ============================================================
# 6. 可选：合并到 9 个客户端 CSV
# ============================================================

def merge_price_to_one_client(
    client_path: Path,
    price_30min: pd.DataFrame,
    output_path: Path,
    client_utc_col: str = CLIENT_UTC_COL,
    match_mode: str = MATCH_MODE
) -> None:
    """
    把 30 min AEMO RRP 合并到单个客户端文件。

    match_mode:
    - "start": 客户端时间表示半小时区间开始时刻
    - "end": 客户端时间表示半小时区间结束时刻
    """
    if match_mode not in {"start", "end"}:
        raise ValueError("match_mode 只能是 'start' 或 'end'")

    client = pd.read_csv(client_path)

    if client_utc_col not in client.columns:
        raise ValueError(f"{client_path.name} 中没有时间列: {client_utc_col}")

    client[client_utc_col] = pd.to_datetime(client[client_utc_col], utc=True)

    price = price_30min.copy()

    if match_mode == "start":
        price_key = "datetime_utc_start"
    else:
        price_key = "datetime_utc_end"

    price[price_key] = pd.to_datetime(price[price_key], utc=True)

    price_keep = price[
        [
            price_key,
            "rrp_aud_per_mwh",
            "total_demand_mw"
        ]
    ].copy()

    merged = client.merge(
        price_keep,
        left_on=client_utc_col,
        right_on=price_key,
        how="left"
    )

    merged = merged.drop(columns=[price_key])

    missing_rate = merged["rrp_aud_per_mwh"].isna().mean()
    print(f"{client_path.name}: 电价缺失率 = {missing_rate:.4%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")


def merge_price_to_9_clients(price_30min: pd.DataFrame) -> None:
    """
    批量处理 client_1.csv 到 client_9.csv。
    """
    print("\n========== 开始合并 9 个客户端 ==========")

    for i in range(1, 10):
        client_path = CLIENT_DIR / CLIENT_FILE_PATTERN.format(i)
        output_path = MERGED_CLIENT_DIR / f"client_{i}_with_aemo_nsw1_rrp.csv"

        if not client_path.exists():
            print(f"[SKIP] 找不到文件: {client_path}")
            continue

        merge_price_to_one_client(
            client_path=client_path,
            price_30min=price_30min,
            output_path=output_path,
            client_utc_col=CLIENT_UTC_COL,
            match_mode=MATCH_MODE
        )

    print(f"合并后的客户端文件已保存到: {MERGED_CLIENT_DIR}")


# ============================================================
# 7. 主程序
# ============================================================

def main():
    print("========== 下载 AEMO NSW1 RRP ==========")
    paths = download_all_months()

    print("\n========== 构建 30 min UTC 电价数据 ==========")
    price_30min = build_aemo_30min(paths)

    print("\n========== 构建 1 h UTC 电价数据 ==========")
    price_1h = build_aemo_1h(price_30min)

    out_30min = OUT_DIR / "AEMO_NSW1_RRP_2010-07-01_to_2013-06-30_30min_UTC.csv"
    out_1h = OUT_DIR / "AEMO_NSW1_RRP_2010-07-01_to_2013-06-30_1h_UTC.csv"

    price_30min.to_csv(out_30min, index=False, encoding="utf-8-sig")
    price_1h.to_csv(out_1h, index=False, encoding="utf-8-sig")

    print("\n========== 保存完成 ==========")
    print(f"30 min 文件: {out_30min}")
    print(f"1 h 文件:    {out_1h}")

    print("\n========== 30 min 预览 ==========")
    print(price_30min.head())
    print(price_30min.tail())

    print("\n========== 1 h 预览 ==========")
    print(price_1h.head())
    print(price_1h.tail())

    check_30min_continuity(price_30min)

    if MERGE_TO_CLIENTS:
        merge_price_to_9_clients(price_30min)

    print("\n========== 说明 ==========")
    print("AEMO SETTLEMENTDATE 已按固定 UTC+10 的 NEM time 转换为 UTC。")
    print("没有使用 Australia/Sydney，因此不会引入悉尼夏令时。")
    print("如果你的客户端数据已经是 UTC，可以直接用 datetime_utc_start 或 datetime_utc_end 合并。")


if __name__ == "__main__":
    main()