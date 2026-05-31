import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def build_default_client_files(num_clients: int = 9) -> List[str]:
    return [os.path.join(DATA_DIR, f"client_{i}.csv") for i in range(1, num_clients + 1)]


@dataclass
class DataConfig:
    client_files: List[str] = field(default_factory=lambda: [
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_1_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_2_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_3_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_4_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_5_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_6_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_7_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_8_load_weather_30min.csv"),
        os.path.join(PROJECT_ROOT, "per_client_merged", "client_9_load_weather_30min.csv"),
    ])
    datetime_col: str = "timestamp"
    target_col: str = "gc"
    net_load_col: str = "net_load"
    seq_len: int = 48
    horizon: int = 1

    train_ratio: float = 0.8
    val_ratio: float = 0.1

    dropna: bool = True
    sort_by_time: bool = True
    freq_minutes: str = "auto"

    use_time_range: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class FeatureConfig:
    use_target_history: bool = True
    use_rrp: bool = True
    rrp_col: str = "rrp_aud_per_mwh"
    raw_feature_cols: List[str] = field(default_factory=list)

    use_slot_sin_cos: bool = True
    use_weekday_sin_cos: bool = True
    use_month_sin_cos: bool = True
    use_is_weekend: bool = True
    use_is_holiday: bool = False

    use_temp_c: bool = True
    temp_source_mode: str = "auto"
    temp_c_col: str = "temp2m_c"
    temp_k_col: str = "temp2m_k"

    use_rh: bool = False
    rh_col: str = "rh2m_pct"

    use_wind: bool = True
    wind_col: str = "wind10m_ms"

    use_ghi: bool = False
    ghi_col: str = "ghi_wm2"

    use_apparent_temp: bool = False

    no_scale_cols: List[str] = field(default_factory=lambda: ["is_weekend", "is_holiday"])


@dataclass
class GCFeatureConfig:
    use_target_history: bool = True
    use_rrp: bool = False

    rrp_col: str = "rrp_aud_per_mwh"
    raw_feature_cols: List[str] = field(default_factory=list)

    use_slot_sin_cos: bool = True
    use_weekday_sin_cos: bool = True
    use_month_sin_cos: bool = True
    use_is_weekend: bool = True
    use_is_holiday: bool = False

    use_temp_c: bool = True
    temp_source_mode: str = "auto"
    temp_c_col: str = "temp2m_c"
    temp_k_col: str = "temp2m_k"

    use_rh: bool = False
    rh_col: str = "rh2m_pct"

    use_wind: bool = True
    wind_col: str = "wind10m_ms"

    use_ghi: bool =True
    ghi_col: str = "ghi_wm2"

    use_apparent_temp: bool = False

    no_scale_cols: List[str] = field(default_factory=lambda: ["is_weekend", "is_holiday"])


@dataclass
class GGFeatureConfig:
    use_target_history: bool = True
    use_rrp: bool = False
    rrp_col: str = "rrp_aud_per_mwh"
    raw_feature_cols: List[str] = field(default_factory=list)

    use_slot_sin_cos: bool = False
    use_weekday_sin_cos: bool = False
    use_month_sin_cos: bool = False
    use_is_weekend: bool = True
    use_is_holiday: bool = False

    use_temp_c: bool = True
    temp_source_mode: str = "auto"
    temp_c_col: str = "temp2m_c"
    temp_k_col: str = "temp2m_k"

    use_rh: bool = False
    rh_col: str = "rh2m_pct"

    use_wind: bool = True
    wind_col: str = "wind10m_ms"

    use_ghi: bool = True
    ghi_col: str = "ghi_wm2"

    use_apparent_temp: bool = False

    no_scale_cols: List[str] = field(default_factory=lambda: ["is_weekend", "is_holiday"])


@dataclass
class ModelConfig:
    use_attention: bool = True

    conv1_channels: int = 32
    conv2_channels: int = 64
    conv1_kernel: int = 3
    conv2_kernel: int = 3

    pool1_kernel: int = 2
    pool2_kernel: int = 3

    lstm_hidden1: int = 32
    lstm_hidden2: int = 16

    attn_units: int = 20
    fc_hidden: int = 16
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 20
    early_stop_patience: int = 6
    lr: float = 1e-3
    random_seed: int = 2024
    num_workers: int = 0
    pin_memory: bool = True

    loss_name: str = "mse"
    optimizer_name: str = "adam"

    scaler_x: str = "minmax"
    scaler_y: str = "minmax"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    task_type: str = "single_target"  # "single_target" or "net_load"
    net_load_method: str = "indirect"  # "direct" or "indirect"


@dataclass
class FederatedConfig:
    aggregation_method: str = "fedavg"
    rounds: int = 20
    local_epochs: int = 1
    client_fraction: float = 1.0
    eval_every: int = 1
    save_dir: str = os.path.join(RUNS_DIR, "fedavg_gc_电价_rc_fc1_head_tau0p10_warmup5_ema0")
    checkpoint_dir: Optional[str] = None
    best_model_name: str = "best_global_model.pth"
    best_checkpoint_name: str = "best_global_checkpoint.pth"
    final_model_name: str = "final_global_model.pth"
    early_stop_patience: int = 6
    use_rc_regularization: bool = True
    rc_lambda: float = 1.0
    use_head_personalization: bool = True
    head_personalization_tau: float = 0.10
    head_param_prefixes: List[str] = field(default_factory=lambda: ["fc1"])
    head_personalization_warmup_rounds: int = 5
    head_mask_update_interval: int = 1
    head_param_exact_names: List[str] = field(default_factory=list)
    use_head_importance_ema: bool = True
    head_importance_ema_beta: float = 0.0
    h2a_num_refs: int = 5
    h2a_embed_dim: int = 16
    h2a_shared_hidden_dim: int = 32
    h2a_branch_hidden_dim: int = 64
    h2a_meta_lr: float = 1e-3
    h2a_gamma: float = 0.25
    h2a_warmup_use_all_clients: bool = True
    h2a_reference_mode: str = "fixed"  # "adaptive" or "fixed"
    h2a_fixed_missing_policy: str = "adaptive"  # "adaptive" or "error"
    h2a_fixed_ref_client_ids: Dict[int, List[int]] = field(default_factory=lambda: {
        1: [8, 6, 7],
        2: [5, 3, 8],
        3: [5, 2, 8],
        4: [2, 5, 3],
        5: [2, 3, 8],
        6: [7, 1, 8],
        7: [6, 1, 8],
        8: [1, 5, 3],
    })
    h2a_feature_param_prefixes: List[str] = field(default_factory=lambda: [
        "conv1", "conv2","lstm1", "lstm2",
    ])
    h2a_head_param_prefixes: List[str] = field(default_factory=lambda: [
 "attention","fc1", "fc2"
    ])
    h2a_unmatched_param_policy: str = "error"


@dataclass
class CentralizedConfig:
    save_dir: str = os.path.join(RUNS_DIR, "centralized")
    best_model_name: str = "best_centralized_model.pth"


@dataclass
class DecentralizedGCMLConfig:
    training_mode: str = "all_clients_disjoint_bidirectional"
    active_client_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    global_rounds: int = 20
    rounds: int = 20
    local_epochs: int = 1
    warmup_local_epochs: int = 1
    pair_schedule_mode: str = "round_robin_disjoint"
    bidirectional_pair_update: bool = True
    transfer_epochs: int = 1
    lambda_transfer: float = 0.03
    repulsion_lambda: float = 0.005
    repulsion_margin: float = 0.05
    alpha_grid_step: float = 0.05
    rho: float = 0.02
    r_max: float = 0.5
    a_max: float = 1.0
    eps: float = 1e-8
    eval_every: int = 1
    save_dir: str = os.path.join(RUNS_DIR, "all_clients_disjoint_bidirectional")
    existing_gc_prediction_dir: Optional[str] = None
    gg_model_subdir: str = "gg_model"
    net_load_from_existing_gc_subdir: str = "net_load_from_existing_gc"
    pair_mode: str = "alternate"  # "alternate" means two clients take turns as receiver.
    receiver_client_id: int = 5
    sender_candidate_client_ids: List[int] = field(default_factory=lambda: [6, 7, 2])
    sender_selection_mode: str = "round_robin"  # "random" or "round_robin"
    enable_merge_rollback: bool = True
    best_model_name_template: str = "best_client_{client_id}_model.pth"
    best_model_name_client1: str = "best_client_1_model.pth"
    best_model_name_client2: str = "best_client_2_model.pth"
    best_receiver_model_name: str = "best_receiver_client_5_model.pth"


@dataclass
class DecomposedNetLoadConfig:
    save_dir: str = os.path.join(RUNS_DIR, "pv_only_decomposition")
    best_model_name: str = "best_pv_only_model.pth"
    final_model_name: str = "final_pv_only_model.pth"
    client_files: List[str] = field(default_factory=lambda: DataConfig().client_files)

    decompose_mode: str = "predict_gg_only"
    target_col: str = "gg"
    load_col: str = "gc"
    pv_col: str = "gg"
    net_load_col: str = "net_load"
    capacity_col: str = "total_pv_capacity_kw"
    ghi_col: str = "ghi_wm2"

    # 输入端只允许使用 net_load 历史，不允许使用 gc/gg 历史作为输入
    input_history_col: str = "net_load"

    # Input feature switches
    use_slot_sin_cos: bool = True

    # 输入特征开关
    use_slot_sin_cos: bool = True
    use_weekday_sin_cos: bool = True
    use_month_sin_cos: bool = True
    use_is_weekend: bool = True
    use_temp_c: bool = True
    use_wind: bool = True
    use_ghi_feature: bool = True
    use_rrp: bool = True
    raw_feature_cols: List[str] = field(default_factory=list)

    # 光伏物理约束
    use_pv_gate: bool = True
    ghi_gate_threshold: float = 5.0
    use_effective_pv_capacity: bool = True
    pv_capacity_quantile: float = 0.995
    pv_capacity_alpha: float = 1.10
    min_pv_capacity_eps: float = 1e-6

    # 损失权重
    lambda_gg: float = 1.0
    lambda_gc_reconstruction: float = 0.0
    early_stop_metric: str = "gg_RMSE"

    # 运行设置
    run_all_clients: bool = True
    save_predictions: bool = True
    save_plots: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    gc_feature: GCFeatureConfig = field(default_factory=GCFeatureConfig)
    gg_feature: GGFeatureConfig = field(default_factory=GGFeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    centralized: CentralizedConfig = field(default_factory=CentralizedConfig)
    decentralized_gcml: DecentralizedGCMLConfig = field(default_factory=DecentralizedGCMLConfig)
    decomposed_net_load: DecomposedNetLoadConfig = field(default_factory=DecomposedNetLoadConfig)


CFG = Config()
