# -*- coding: utf-8 -*-
"""
删除 Ausgrid 客户端数据中的夏令时异常行。

适用场景：
1. 数据位于 per_client_merged 文件夹
2. 文件名类似 client_1_load_weather_30min.csv
3. timestamp 是半小时区间结束时刻
4. 需要删除夏令时导致的重复 timestamp 和关键列缺失行
5. 覆盖原文件，文件名不变
"""

from pathlib import Path
import shutil
import pandas as pd


# =========================
# 1. 路径配置
# =========================

CLIENT_DIR = Path("per_client_merged")
BACKUP_DIR = Path("per_client_merged_backup_before_delete_dst_abnormal")
REPORT_DIR = Path("per_client_merged_cleaning_report")

CLIENT_PATTERN = "client_*_load_weather_30min.csv"
TIME_COL = "timestamp"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 关键列配置
# =========================
# 只检查文件中实际存在的列。
# 你可以按自己的字段名继续添加。

CANDIDATE_REQUIRED_COLS = [
    # 负荷 / 光伏
    "gc",
    "gg",
    "cl",
    "total_load",
    "net_load",

    # 气象
    "temp2m_c",
    "wind10m_ms",
    "ghi_wm2",
    "rh2m_pct",

    # 时间特征，如果你已经生成了也可以检查
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "hour_sin",
    "hour_cos",
    "slot_sin",
    "slot_cos",

    # 电价
    "rrp_aud_per_mwh",
]


# =========================
# 3. 是否删除所有重复 timestamp 行
# =========================
# True：如果某个 timestamp 出现重复，则该 timestamp 对应的所有行都删除，更严格。
# False：只删除重复项，保留第一次出现，更保守。
#
# 对夏令时结束导致的 02:00、02:30 重复，我建议 True。
# 因为这两个重复点已经无法区分真实 UTC 先后顺序。

DROP_ALL_DUPLICATED_TIMESTAMPS = True


def read_client_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if TIME_COL not in df.columns:
        raise ValueError(f"{path.name} 中找不到 {TIME_COL} 列。当前列名为: {list(df.columns)}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    return df


def find_abnormal_timestamps(path: Path, df: pd.DataFrame):
    """
    找出单个客户端中的异常时间点。
    返回：
    abnormal_ts_set: 需要删除的 timestamp 集合
    report: 统计信息
    """
    abnormal_ts = set()

    old_rows = len(df)

    # timestamp 无法解析
    bad_time_mask = df[TIME_COL].isna()
    bad_time_count = int(bad_time_mask.sum())

    # timestamp 重复
    duplicated_mask_all = df[TIME_COL].duplicated(keep=False)
    duplicated_mask_extra = df[TIME_COL].duplicated(keep="first")

    if DROP_ALL_DUPLICATED_TIMESTAMPS:
        dup_ts = set(df.loc[duplicated_mask_all, TIME_COL].dropna())
    else:
        dup_ts = set(df.loc[duplicated_mask_extra, TIME_COL].dropna())

    abnormal_ts.update(dup_ts)

    # 关键列缺失
    required_cols = [c for c in CANDIDATE_REQUIRED_COLS if c in df.columns]

    missing_by_col = {}
    missing_ts = set()

    if required_cols:
        missing_mask = df[required_cols].isna().any(axis=1)
        missing_rows = df.loc[missing_mask].copy()

        missing_ts = set(missing_rows[TIME_COL].dropna())
        abnormal_ts.update(missing_ts)

        miss_counts = df[required_cols].isna().sum()
        missing_by_col = miss_counts[miss_counts > 0].to_dict()
    else:
        missing_mask = pd.Series(False, index=df.index)

    report = {
        "file": path.name,
        "old_rows": old_rows,
        "bad_time_count": bad_time_count,
        "duplicate_timestamp_rows_all": int(duplicated_mask_all.sum()),
        "duplicate_timestamp_rows_extra": int(duplicated_mask_extra.sum()),
        "duplicate_unique_timestamps": len(dup_ts),
        "required_cols_checked": ",".join(required_cols),
        "missing_rows_in_required_cols": int(missing_mask.sum()),
        "missing_unique_timestamps": len(missing_ts),
        "missing_by_col": str(missing_by_col),
        "abnormal_unique_timestamps_this_file": len(abnormal_ts),
    }

    return abnormal_ts, report


def save_abnormal_timestamp_list(abnormal_ts_union):
    abnormal_list = sorted(list(abnormal_ts_union))
    out_path = REPORT_DIR / "abnormal_timestamps_to_delete.csv"

    pd.DataFrame({
        TIME_COL: abnormal_list
    }).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n异常 timestamp 列表已保存: {out_path}")
    print(f"需要统一删除的异常 timestamp 数量: {len(abnormal_list)}")

    if len(abnormal_list) > 0:
        print("\n前几个异常 timestamp:")
        for x in abnormal_list[:20]:
            print(" ", x)

        print("\n后几个异常 timestamp:")
        for x in abnormal_list[-20:]:
            print(" ", x)


def clean_and_overwrite_file(path: Path, abnormal_ts_union):
    """
    删除异常时间点，覆盖原文件。
    """
    df = read_client_file(path)

    old_rows = len(df)

    # 备份
    backup_path = BACKUP_DIR / path.name
    shutil.copy2(path, backup_path)

    # 删除 timestamp 无法解析的行
    df = df.dropna(subset=[TIME_COL]).copy()

    # 删除所有客户端统一判定的异常 timestamp
    df = df[~df[TIME_COL].isin(abnormal_ts_union)].copy()

    # 排序
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # 保存前转回字符串
    df[TIME_COL] = df[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(path, index=False, encoding="utf-8-sig")

    new_rows = len(df)

    return {
        "file": path.name,
        "old_rows": old_rows,
        "new_rows": new_rows,
        "deleted_rows": old_rows - new_rows,
    }


def final_check(path: Path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df[TIME_COL], errors="coerce")

    duplicate_count = int(ts.duplicated().sum())
    bad_time_count = int(ts.isna().sum())

    diffs = ts.sort_values().diff().dropna()
    non_30min_count = int((diffs != pd.Timedelta(minutes=30)).sum())

    required_cols = [c for c in CANDIDATE_REQUIRED_COLS if c in df.columns]
    if required_cols:
        missing_required_rows = int(df[required_cols].isna().any(axis=1).sum())
    else:
        missing_required_rows = 0

    return {
        "file": path.name,
        "rows": len(df),
        "time_start": str(ts.min()),
        "time_end": str(ts.max()),
        "bad_time_count": bad_time_count,
        "duplicate_timestamp_count": duplicate_count,
        "non_30min_interval_count": non_30min_count,
        "missing_required_rows": missing_required_rows,
    }


def main():
    client_files = sorted(CLIENT_DIR.glob(CLIENT_PATTERN))

    if not client_files:
        raise FileNotFoundError(f"在 {CLIENT_DIR} 中没有找到匹配文件: {CLIENT_PATTERN}")

    print("找到客户端文件:")
    for p in client_files:
        print(" ", p)

    # 第一轮：找出所有客户端中的异常 timestamp
    abnormal_ts_union = set()
    detect_reports = []

    print("\n开始扫描异常 timestamp...")

    for path in client_files:
        df = read_client_file(path)
        abnormal_ts, report = find_abnormal_timestamps(path, df)

        abnormal_ts_union.update(abnormal_ts)
        detect_reports.append(report)

        print(f"{path.name}: 本文件异常 timestamp 数量 = {len(abnormal_ts)}")

    detect_report_df = pd.DataFrame(detect_reports)
    detect_report_path = REPORT_DIR / "detect_abnormal_report.csv"
    detect_report_df.to_csv(detect_report_path, index=False, encoding="utf-8-sig")

    print(f"\n异常扫描报告已保存: {detect_report_path}")

    save_abnormal_timestamp_list(abnormal_ts_union)

    # 第二轮：统一删除异常 timestamp，覆盖原文件
    print("\n开始统一删除异常行并覆盖原文件...")
    clean_reports = []

    for path in client_files:
        report = clean_and_overwrite_file(path, abnormal_ts_union)
        clean_reports.append(report)
        print(f"{path.name}: 删除 {report['deleted_rows']} 行，剩余 {report['new_rows']} 行")

    clean_report_df = pd.DataFrame(clean_reports)
    clean_report_path = REPORT_DIR / "clean_overwrite_report.csv"
    clean_report_df.to_csv(clean_report_path, index=False, encoding="utf-8-sig")

    print(f"\n覆盖清洗报告已保存: {clean_report_path}")
    print(f"原始文件已备份到: {BACKUP_DIR}")

    # 第三轮：最终检查
    print("\n开始最终检查...")
    final_reports = []

    for path in client_files:
        report = final_check(path)
        final_reports.append(report)

        print("\n" + path.name)
        print("  行数:", report["rows"])
        print("  时间范围:", report["time_start"], "到", report["time_end"])
        print("  无法解析 timestamp:", report["bad_time_count"])
        print("  重复 timestamp:", report["duplicate_timestamp_count"])
        print("  非 30min 间隔数量:", report["non_30min_interval_count"])
        print("  关键列仍有缺失的行数:", report["missing_required_rows"])

    final_report_df = pd.DataFrame(final_reports)
    final_report_path = REPORT_DIR / "final_check_report.csv"
    final_report_df.to_csv(final_report_path, index=False, encoding="utf-8-sig")

    print(f"\n最终检查报告已保存: {final_report_path}")
    print("\n全部完成。")


if __name__ == "__main__":
    main()