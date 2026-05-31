import os
import json
import random
import warnings
from dataclasses import asdict
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore", message=".*padding='same'.*")


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config(cfg, save_dir: str):
    ensure_dir(save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def infer_freq_minutes(dt_series: pd.Series) -> int:
    dt_sorted = dt_series.sort_values().drop_duplicates()
    diffs = dt_sorted.diff().dropna()
    if len(diffs) == 0:
        raise ValueError("无法从 datetime 推断分辨率，时间列样本不足。")
    minutes = int(diffs.mode().iloc[0].total_seconds() // 60)
    return minutes


def get_slots_per_day(freq_minutes: int) -> int:
    if freq_minutes <= 0 or 1440 % freq_minutes != 0:
        raise ValueError(f"freq_minutes={freq_minutes} 不合法，无法整除一天。")
    return 1440 // freq_minutes


def get_scaler(name: str):
    name = name.lower()
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    if name == "none":
        return None
    raise ValueError(f"不支持的 scaler 类型: {name}")


def inverse_transform_array(scaler, arr: np.ndarray) -> np.ndarray:
    if scaler is None:
        return arr
    original_shape = arr.shape
    arr_2d = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    restored = scaler.inverse_transform(arr_2d)
    return restored.reshape(original_shape)


def add_derived_columns(df: pd.DataFrame, net_load_col: str = "net_load") -> pd.DataFrame:
    out = df.copy()
    if net_load_col not in out.columns and "gc" in out.columns and "gg" in out.columns:
        out[net_load_col] = out["gc"].astype(float) - out["gg"].astype(float)
    return out


def add_timestamp_occurrence_key(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    # Preserve repeated wall-clock timestamps such as DST fallback hours.
    out["_timestamp_occurrence"] = out.groupby(timestamp_col).cumcount()
    return out


def drop_duplicate_timestamps(df: pd.DataFrame, timestamp_col: str, keep: str = "first") -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.drop_duplicates(subset=[timestamp_col], keep=keep)
    return out.sort_values(timestamp_col).reset_index(drop=True)


def build_features(df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, List[str]]:
    df = add_derived_columns(df.copy(), cfg.data.net_load_col)
    dc = cfg.data
    fc = cfg.feature

    if dc.datetime_col not in df.columns:
        raise ValueError(f"数据中缺少时间列: {dc.datetime_col}")

    df[dc.datetime_col] = pd.to_datetime(df[dc.datetime_col])
    if dc.sort_by_time:
        df = df.sort_values(dc.datetime_col).reset_index(drop=True)

    if dc.use_time_range:
        if dc.start_time is not None:
            df = df[df[dc.datetime_col] >= pd.to_datetime(dc.start_time)]
        if dc.end_time is not None:
            df = df[df[dc.datetime_col] <= pd.to_datetime(dc.end_time)]
        df = df.reset_index(drop=True)

    freq_minutes = infer_freq_minutes(df[dc.datetime_col]) if dc.freq_minutes == "auto" else int(dc.freq_minutes)
    slots_per_day = get_slots_per_day(freq_minutes)

    dt = df[dc.datetime_col]
    slot_idx = (dt.dt.hour * 60 + dt.dt.minute) // freq_minutes
    weekday_idx = dt.dt.weekday
    month_idx = dt.dt.month - 1

    if fc.use_slot_sin_cos:
        df["slot_sin"] = np.sin(2 * np.pi * slot_idx / slots_per_day)
        df["slot_cos"] = np.cos(2 * np.pi * slot_idx / slots_per_day)

    if fc.use_weekday_sin_cos:
        df["weekday_sin"] = np.sin(2 * np.pi * weekday_idx / 7.0)
        df["weekday_cos"] = np.cos(2 * np.pi * weekday_idx / 7.0)

    if fc.use_month_sin_cos:
        df["month_sin"] = np.sin(2 * np.pi * month_idx / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * month_idx / 12.0)

    if fc.use_is_weekend:
        df["is_weekend"] = (weekday_idx >= 5).astype(int)

    if fc.use_is_holiday and "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    if fc.use_temp_c:
        temp_mode = fc.temp_source_mode.lower()
        if temp_mode == "auto":
            if fc.temp_c_col in df.columns:
                df["temp_c"] = df[fc.temp_c_col].astype(float)
            elif fc.temp_k_col in df.columns:
                df["temp_c"] = df[fc.temp_k_col].astype(float) - 273.15
            else:
                raise ValueError(
                    f"已启用 temp_c，但数据中既没有 {fc.temp_c_col}，也没有 {fc.temp_k_col}"
                )
        elif temp_mode == "c":
            if fc.temp_c_col not in df.columns:
                raise ValueError(f"缺少列 {fc.temp_c_col}")
            df["temp_c"] = df[fc.temp_c_col].astype(float)
        elif temp_mode == "k":
            if fc.temp_k_col not in df.columns:
                raise ValueError(f"缺少列 {fc.temp_k_col}")
            df["temp_c"] = df[fc.temp_k_col].astype(float) - 273.15
        else:
            raise ValueError('temp_source_mode 只支持 "auto" / "c" / "k"')

    if fc.use_rh:
        if fc.rh_col not in df.columns:
            raise ValueError(f"已启用 rh2m_pct，但数据中缺少列 {fc.rh_col}")
        df["rh2m_pct"] = df[fc.rh_col].clip(lower=0, upper=100)

    if fc.use_wind:
        if fc.wind_col not in df.columns:
            raise ValueError(f"已启用 wind10m_ms，但数据中缺少列 {fc.wind_col}")
        df["wind10m_ms"] = df[fc.wind_col].clip(lower=0)

    if fc.use_ghi:
        if fc.ghi_col not in df.columns:
            raise ValueError(f"已启用 ghi_wm2，但数据中缺少列 {fc.ghi_col}")
        df["ghi_wm2"] = df[fc.ghi_col].astype(float)

    if fc.use_apparent_temp:
        needed = ["temp_c", "rh2m_pct", "wind10m_ms"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"已启用 apparent_temp_c，但缺少列: {missing}")
        e = (df["rh2m_pct"] / 100.0) * 6.105 * np.exp(17.27 * df["temp_c"] / (237.7 + df["temp_c"]))
        df["apparent_temp_c"] = df["temp_c"] + 0.33 * e - 0.70 * df["wind10m_ms"] - 4.0

    feature_cols: List[str] = []

    if fc.use_target_history:
        feature_cols.append(dc.target_col)

    if getattr(fc, "use_rrp", False):
        rrp_col = getattr(fc, "rrp_col", "rrp_aud_per_mwh")
        if rrp_col not in df.columns:
            raise ValueError(
                f"Missing RRP feature column '{rrp_col}' in data. "
                "Set cfg.feature.use_rrp=False or check cfg.feature.rrp_col."
            )
        if rrp_col not in feature_cols:
            feature_cols.append(rrp_col)

    for col in getattr(fc, "raw_feature_cols", []):
        if col not in df.columns:
            raise ValueError(
                f"Missing raw feature column '{col}' in data. "
                f"Please check cfg.feature.raw_feature_cols or the input CSV columns."
            )
        if col not in feature_cols:
            feature_cols.append(col)

    optional_cols = [
        "slot_sin", "slot_cos",
        "weekday_sin", "weekday_cos",
        "month_sin", "month_cos",
        "is_weekend", "is_holiday",
        "temp_c", "rh2m_pct", "wind10m_ms", "ghi_wm2", "apparent_temp_c",
    ]
    for col in optional_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)

    missing_required = [c for c in [dc.target_col] + feature_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"数据中缺少这些必要列: {missing_required}")

    if dc.dropna:
        df = df.dropna(subset=feature_cols + [dc.target_col]).reset_index(drop=True)

    return df, feature_cols


def split_df_by_time(df: pd.DataFrame, cfg):
    n = len(df)
    train_end = int(n * cfg.data.train_ratio)
    val_end = int(n * (cfg.data.train_ratio + cfg.data.val_ratio))

    if train_end <= cfg.data.seq_len or val_end <= train_end:
        raise ValueError("训练/验证划分太小，无法形成有效序列。")

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def fit_and_transform_x(train_df, val_df, test_df, feature_cols, cfg):
    no_scale_cols = set(cfg.feature.no_scale_cols)
    scale_cols = [c for c in feature_cols if c not in no_scale_cols]
    keep_cols = [c for c in feature_cols if c in no_scale_cols]

    x_scaler = get_scaler(cfg.train.scaler_x)
    if x_scaler is not None and len(scale_cols) > 0:
        x_scaler.fit(train_df[scale_cols].values)

        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()

        train_scaled.loc[:, scale_cols] = x_scaler.transform(train_df[scale_cols].values)
        val_scaled.loc[:, scale_cols] = x_scaler.transform(val_df[scale_cols].values)
        test_scaled.loc[:, scale_cols] = x_scaler.transform(test_df[scale_cols].values)
    else:
        x_scaler = None
        train_scaled, val_scaled, test_scaled = train_df.copy(), val_df.copy(), test_df.copy()

    return train_scaled, val_scaled, test_scaled, x_scaler, scale_cols, keep_cols


def fit_and_transform_y(train_df, val_df, test_df, cfg):
    target_col = cfg.data.target_col
    y_scaler = get_scaler(cfg.train.scaler_y)

    train_y = train_df[[target_col]].values
    val_y = val_df[[target_col]].values
    test_y = test_df[[target_col]].values

    if y_scaler is not None:
        train_y = y_scaler.fit_transform(train_y)
        val_y = y_scaler.transform(val_y)
        test_y = y_scaler.transform(test_y)

    return train_y, val_y, test_y, y_scaler


def create_sequences(feature_array: np.ndarray, target_array: np.ndarray, timestamp_array, seq_len: int, horizon: int):
    xs, ys, ts = [], [], []
    total_len = len(feature_array)

    for end_idx in range(seq_len, total_len - horizon + 1):
        start_idx = end_idx - seq_len
        x = feature_array[start_idx:end_idx, :]
        y = target_array[end_idx:end_idx + horizon, 0]
        label_ts = timestamp_array[end_idx:end_idx + horizon]

        xs.append(x)
        ys.append(y)
        ts.append(label_ts)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    ts = np.asarray(ts)

    if horizon == 1:
        ys = ys.reshape(-1, 1)
        ts = ts.reshape(-1, 1)

    return xs, ys, ts


def make_dataloader(x_seq, y_seq, cfg, shuffle=False):
    return DataLoader(
        SeqDataset(x_seq, y_seq),
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory and str(cfg.train.device).startswith("cuda"),
    )


def prepare_client_data(data_path: str, cfg) -> Dict:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到客户端数据文件: {data_path}")

    df = add_derived_columns(pd.read_csv(data_path), cfg.data.net_load_col)
    df, feature_cols = build_features(df, cfg)
    train_df, val_df, test_df = split_df_by_time(df, cfg)

    train_scaled_df, val_scaled_df, test_scaled_df, x_scaler, scale_cols, keep_cols = fit_and_transform_x(
        train_df, val_df, test_df, feature_cols, cfg
    )
    y_train_scaled, y_val_scaled, y_test_scaled, y_scaler = fit_and_transform_y(train_df, val_df, test_df, cfg)

    x_train = train_scaled_df[feature_cols].values
    x_val = val_scaled_df[feature_cols].values
    x_test = test_scaled_df[feature_cols].values

    ts_train = pd.to_datetime(train_df[cfg.data.datetime_col]).values
    ts_val = pd.to_datetime(val_df[cfg.data.datetime_col]).values
    ts_test = pd.to_datetime(test_df[cfg.data.datetime_col]).values

    x_train_seq, y_train_seq, train_ts_seq = create_sequences(
        x_train, y_train_scaled, ts_train, cfg.data.seq_len, cfg.data.horizon
    )
    x_val_seq, y_val_seq, val_ts_seq = create_sequences(
        x_val, y_val_scaled, ts_val, cfg.data.seq_len, cfg.data.horizon
    )
    x_test_seq, y_test_seq, test_ts_seq = create_sequences(
        x_test, y_test_scaled, ts_test, cfg.data.seq_len, cfg.data.horizon
    )

    return {
        "feature_cols": feature_cols,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "scale_cols": scale_cols,
        "keep_cols": keep_cols,
        "train_loader": make_dataloader(x_train_seq, y_train_seq, cfg, shuffle=True),
        "val_loader": make_dataloader(x_val_seq, y_val_seq, cfg, shuffle=False),
        "test_loader": make_dataloader(x_test_seq, y_test_seq, cfg, shuffle=False),
        "train_samples": len(x_train_seq),
        "val_samples": len(x_val_seq),
        "test_samples": len(x_test_seq),
        "val_timestamps": val_ts_seq,
        "test_timestamps": test_ts_seq,
        "train_timestamps": train_ts_seq,
        "raw_df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def save_scalers(x_scaler, y_scaler, save_dir: str, prefix: str):
    ensure_dir(save_dir)
    if x_scaler is not None:
        joblib.dump(x_scaler, os.path.join(save_dir, f"{prefix}_x_scaler.save"))
    if y_scaler is not None:
        joblib.dump(y_scaler, os.path.join(save_dir, f"{prefix}_y_scaler.save"))


def build_centralized_aggregate_dataframe(client_files: List[str], cfg) -> pd.DataFrame:
    dt_col = cfg.data.datetime_col
    target_col = cfg.data.target_col

    weather_source_cols = []
    if cfg.feature.use_temp_c:
        weather_source_cols.append(cfg.feature.temp_c_col if cfg.feature.temp_source_mode == "c" else cfg.feature.temp_k_col)
        if cfg.feature.temp_source_mode == "auto":
            weather_source_cols.extend([cfg.feature.temp_c_col, cfg.feature.temp_k_col])
    if cfg.feature.use_rh:
        weather_source_cols.append(cfg.feature.rh_col)
    if cfg.feature.use_wind:
        weather_source_cols.append(cfg.feature.wind_col)
    if cfg.feature.use_ghi:
        weather_source_cols.append(cfg.feature.ghi_col)
    if getattr(cfg.feature, "use_rrp", False):
        weather_source_cols.append(getattr(cfg.feature, "rrp_col", "rrp_aud_per_mwh"))
    for col in getattr(cfg.feature, "raw_feature_cols", []):
        if col not in (dt_col, target_col):
            weather_source_cols.append(col)

    weather_source_cols = list(dict.fromkeys(weather_source_cols))

    merged = None
    used_weather_cols = set()

    for idx, path in enumerate(client_files, start=1):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到客户端数据文件: {path}")

        df = pd.read_csv(path)
        if dt_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"{path} 缺少 {dt_col} 或 {target_col}")

        keep_cols = [dt_col, target_col]
        for col in weather_source_cols:
            if col in df.columns:
                keep_cols.append(col)
                used_weather_cols.add(col)

        tmp = df[keep_cols].copy()
        rename_map = {target_col: f"{target_col}_client_{idx}"}
        for col in keep_cols:
            if col != dt_col and col != target_col:
                rename_map[col] = f"{col}_client_{idx}"
        tmp = tmp.rename(columns=rename_map)
        tmp = add_timestamp_occurrence_key(tmp, dt_col)

        if merged is None:
            merged = tmp
        else:
            merged = pd.merge(merged, tmp, on=[dt_col, "_timestamp_occurrence"], how="inner")

    if merged is None or len(merged) == 0:
        raise ValueError("聚合后的中心化数据为空，请检查时间戳是否可对齐。")

    out = pd.DataFrame({dt_col: merged[dt_col].copy()})
    target_cols = [f"{target_col}_client_{i}" for i in range(1, len(client_files) + 1)]
    out[target_col] = merged[target_cols].sum(axis=1)

    for col in used_weather_cols:
        candidate_cols = [f"{col}_client_{i}" for i in range(1, len(client_files) + 1) if f"{col}_client_{i}" in merged.columns]
        if len(candidate_cols) > 0:
            out[col] = merged[candidate_cols].mean(axis=1)

    out = out.sort_values(dt_col).reset_index(drop=True)
    return out
