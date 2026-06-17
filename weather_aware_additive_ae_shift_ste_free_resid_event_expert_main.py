"""AND-style weather-aware additive AE-shift net-load forecasting with free residual and event expert.

Examples:
    python weather_aware_additive_ae_shift_ste_free_resid_event_expert_main.py --client-id 2 --epochs 60 --lambda-ste 1.0 --lambda-next 1.0 --alpha-ste 0.1 --lambda-event-corr 0.1 --lambda-event-gate 0.02 --lambda-event-normal 0.01 --event-weight-beta 2.0
    python weather_aware_additive_ae_shift_ste_free_resid_event_expert_main.py --client-id 2 --epochs 60 --no-use-event-expert
"""

import argparse
import copy
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

try:
    from utils.runtime_env import ensure_conda_dll_paths

    ensure_conda_dll_paths()
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import CFG
from models.cnn_lstm import Attention, SamePadMaxPool1d
from utils.data_utils import ensure_dir, get_scaler, set_seed
from utils.metrics import calc_metrics, plot_round_curve, plot_true_pred, print_metrics, save_metrics_csv


@dataclass
class WeatherAwareAEShiftExperimentConfig:
    seq_len: int = 48
    horizon: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    ghi_threshold: float = 10.0
    smooth_k: int = 3
    lambda_next: float = 1.0
    lambda_ste: float = 1.0
    alpha_ste: float = 0.1
    lambda_trend_smooth: float = 0.0
    lambda_ae_y: float = 0.0
    lambda_ae_resid: float = 0.0
    lambda_ru_mean: float = 0.0
    lambda_rw_ae: float = 0.0
    user_resid_scale: float = 0.3
    user_day_scale: float = 0.5
    detach_user_resid_input: bool = True
    num_basis: int = 48
    ae_y_hidden: int = 24
    ae_y_bottleneck: int = 12
    ae_resid_hidden: int = 16
    ae_resid_bottleneck: int = 8
    weather_resid_ae_hidden: int = 16
    weather_resid_ae_bottleneck: int = 8
    weather_conv1_channels: int = 32
    weather_conv2_channels: int = 64
    weather_kernel: int = 3
    weather_lstm_hidden1: int = 32
    weather_lstm_hidden2: int = 16
    weather_attn_units: int = 20
    user_weibull_lstm_hidden: int = 16
    user_weibull_fc_hidden: int = 8
    fc_hidden: int = 16
    dropout: float = 0.0
    use_event_expert: bool = True
    event_expert_hidden: int = 16
    event_gate_hidden: int = 16
    lambda_event_corr: float = 0.1
    lambda_event_gate: float = 0.02
    lambda_event_normal: float = 0.01
    event_weight_beta: float = 2.0
    event_ramp_quantile: float = 0.8
    event_ghi_quantile: float = 0.8
    event_peak_quantile: float = 0.8
    event_tau_ramp: float = 0.05
    event_tau_ghi: float = 0.05
    event_tau_zero: float = 0.05
    event_tau_peak: float = 0.05
    event_zero_eps: float = 0.05
    detach_event_base_error: bool = True


class WeatherAwareAEShiftDataset(Dataset):
    """Sequence-to-sequence shifted samples for H=48.

    y_hist is rows[start_idx:end_idx]; y_shift is rows[start_idx + 1:end_idx + 1].
    end_idx is the t+1 target index, so the last y_shift point is the single-step
    forecast target.
    """

    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.arrays = arrays
        self.length = int(arrays["y_shift"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "y_hist": torch.tensor(self.arrays["y_hist"][idx], dtype=torch.float32),
            "y_shift": torch.tensor(self.arrays["y_shift"][idx], dtype=torch.float32),
            "time_hist_scalar": torch.tensor(self.arrays["time_hist_scalar"][idx], dtype=torch.float32),
            "time_shift_scalar": torch.tensor(self.arrays["time_shift_scalar"][idx], dtype=torch.float32),
            "weather_hist": torch.tensor(self.arrays["weather_hist"][idx], dtype=torch.float32),
            "weather_shift": torch.tensor(self.arrays["weather_shift"][idx], dtype=torch.float32),
            "time_enc_hist": torch.tensor(self.arrays["time_enc_hist"][idx], dtype=torch.float32),
            "time_enc_shift": torch.tensor(self.arrays["time_enc_shift"][idx], dtype=torch.float32),
            "timestamp_future": self.arrays["timestamp_future"][idx],
            "timestamp_shift": self.arrays["timestamp_shift"][idx],
        }


class PeriodicSineSequenceBranch(nn.Module):
    """Learnable sine-basis periodic branch for the whole shift sequence."""

    def __init__(self, num_basis: int = 48, period: int = 48):
        super().__init__()
        self.num_basis = int(num_basis)
        self.period = float(period)
        self.omega = nn.Parameter(torch.empty(self.num_basis))
        self.phi = nn.Parameter(torch.empty(self.num_basis))
        self.amplitude = nn.Parameter(torch.empty(self.num_basis))
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            k = torch.arange(self.num_basis, dtype=torch.float32)
            omega = 2.0 * math.pi * k / self.period
            phi = math.pi / 2.0 + torch.remainder(k, 2.0) * math.pi / 2.0
            self.omega.copy_(omega)
            self.phi.copy_(phi)
            self.amplitude.fill_(1.0)
            self.bias.zero_()

    def forward(self, time_shift_scalar: torch.Tensor) -> torch.Tensor:
        time_index = time_shift_scalar * self.period
        basis = torch.sin(
            time_index * self.omega.view(1, 1, -1) + self.phi.view(1, 1, -1)
        )
        return torch.matmul(basis, self.amplitude.view(-1, 1)) + self.bias


class LinearTrendBranch(nn.Module):
    """Simple linear MLP trend branch: 48 -> 48."""

    def __init__(self, seq_len: int = 48):
        super().__init__()
        self.seq_len = int(seq_len)
        self.linear = nn.Linear(self.seq_len, self.seq_len)

    def forward(self, y_hist: torch.Tensor) -> torch.Tensor:
        trend = self.linear(y_hist.squeeze(-1))
        return trend.unsqueeze(-1)


class SequenceAutoEncoder(nn.Module):
    """Flat sequence autoencoder with Tanh bottleneck."""

    def __init__(self, seq_len: int, hidden: int, bottleneck: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.encoder = nn.Sequential(
            nn.Linear(self.seq_len, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(bottleneck)),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(bottleneck), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), self.seq_len),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(self.encoder(seq.squeeze(-1)))
        return recon.unsqueeze(-1)


class WeatherResidualSeqBranch(nn.Module):
    """CNN-LSTM-Attention weather residual branch returning raw Rw_shift only.

    This branch intentionally does not receive calendar/time encodings. The
    daylight gate is applied by the parent model after the raw residual is made.
    """

    def __init__(
        self,
        hist_input_dim: int = 6,
        shift_exog_dim: int = 5,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        kernel_size: int = 3,
        lstm_hidden1: int = 32,
        lstm_hidden2: int = 16,
        attn_units: int = 20,
        fc_hidden: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            hist_input_dim,
            conv1_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.pool1 = SamePadMaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(
            conv1_channels,
            conv2_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.pool2 = SamePadMaxPool1d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(conv2_channels, lstm_hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hidden1, lstm_hidden2, batch_first=True)
        self.attention = Attention(input_dim=lstm_hidden2, attn_units=attn_units)
        self.weather_seq_head = nn.Sequential(
            nn.Linear(lstm_hidden2 + attn_units + shift_exog_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(
        self,
        base_resid_hist: torch.Tensor,
        weather_hist: torch.Tensor,
        weather_shift: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([base_resid_hist, weather_hist], dim=-1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        seq_hidden, _ = self.lstm2(x)
        attn_vec = self.attention(seq_hidden)
        attn_seq = attn_vec.unsqueeze(1).repeat(1, weather_shift.size(1), 1)
        decoder_input = torch.cat([seq_hidden, attn_seq, weather_shift], dim=-1)
        return self.weather_seq_head(decoder_input)


class WeibullAttention(nn.Module):
    """Global sequence-level Weibull attention over hidden states."""

    def __init__(self):
        super().__init__()
        self.raw_kappa = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.raw_lambda = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, seq_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        horizon = int(seq_hidden.size(1))
        positions = torch.arange(
            1,
            horizon + 1,
            dtype=seq_hidden.dtype,
            device=seq_hidden.device,
        )
        kappa = F.softplus(self.raw_kappa) + 1e-4
        lambda_ = F.softplus(self.raw_lambda) + 1e-4
        scaled = positions / lambda_
        alpha = (kappa / lambda_) * torch.pow(scaled, kappa - 1.0) * torch.exp(
            -torch.pow(scaled, kappa)
        )
        alpha = alpha / (alpha.sum() + 1e-6)
        context = torch.sum(seq_hidden * alpha.view(1, horizon, 1), dim=1)
        context_seq = context.unsqueeze(1).repeat(1, horizon, 1)
        return context_seq, alpha.view(1, horizon, 1)


class UserWeibullResidualSeqBranch(nn.Module):
    """LSTM user residual branch with sequence-level Weibull attention."""

    def __init__(
        self,
        time_dim: int = 7,
        lstm_hidden: int = 16,
        fc_hidden: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1 + 1 + time_dim + 1,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.weibull_attention = WeibullAttention()
        self.user_seq_head = nn.Sequential(
            nn.Linear(2 * lstm_hidden + 1 + time_dim + 1, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(
        self,
        user_resid_hist: torch.Tensor,
        temp_hist: torch.Tensor,
        time_enc_hist: torch.Tensor,
        daylight_hist: torch.Tensor,
        temp_shift: torch.Tensor,
        time_enc_shift: torch.Tensor,
        daylight_shift: torch.Tensor,
    ) -> torch.Tensor:
        encoder_input = torch.cat([user_resid_hist, temp_hist, time_enc_hist, daylight_hist], dim=-1)
        seq_hidden, _ = self.lstm(encoder_input)
        context_seq, _ = self.weibull_attention(seq_hidden)
        decoder_input = torch.cat(
            [seq_hidden, context_seq, temp_shift, time_enc_shift, daylight_shift],
            dim=-1,
        )
        return self.user_seq_head(decoder_input)


class EventExpertCorrection(nn.Module):
    """Lightweight event gate plus correction head for sudden operating conditions."""

    def __init__(self, event_expert_hidden: int = 16, event_gate_hidden: int = 16):
        super().__init__()
        input_dim = 16
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, int(event_gate_hidden)),
            nn.ReLU(),
            nn.Linear(int(event_gate_hidden), 1),
        )
        self.expert_net = nn.Sequential(
            nn.Linear(input_dim, int(event_expert_hidden)),
            nn.ReLU(),
            nn.Linear(int(event_expert_hidden), 1),
        )

    def forward(
        self,
        y_base_shift_pred: torch.Tensor,
        s_shift: torch.Tensor,
        r_shift: torch.Tensor,
        y_hist: torch.Tensor,
        weather_shift: torch.Tensor,
        time_enc_shift: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del y_hist
        delta_base_shift = torch.zeros_like(y_base_shift_pred)
        delta_base_shift[:, 1:, :] = y_base_shift_pred[:, 1:, :] - y_base_shift_pred[:, :-1, :]
        z_shift = torch.cat(
            [
                y_base_shift_pred,
                s_shift,
                r_shift,
                delta_base_shift,
                weather_shift,
                time_enc_shift,
            ],
            dim=-1,
        )
        q_shift = torch.sigmoid(self.gate_net(z_shift))
        c_shift = self.expert_net(z_shift)
        event_correction_shift = q_shift * c_shift
        return {
            "event_gate_shift": q_shift,
            "event_correction_raw_shift": c_shift,
            "event_correction_shift": event_correction_shift,
        }


class WeatherAwareAEShiftNetLoadModel(nn.Module):
    """y_shift_pred = S_shift + free Rw_shift + optional event expert correction."""

    def __init__(self, cfg: WeatherAwareAEShiftExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = int(cfg.seq_len)
        self.periodic_branch = PeriodicSineSequenceBranch(
            num_basis=cfg.num_basis,
            period=cfg.seq_len,
        )
        self.trend_branch = LinearTrendBranch(seq_len=cfg.seq_len)
        self.shape_fusion = nn.Linear(2, 1)
        self.y_autoencoder = SequenceAutoEncoder(
            seq_len=cfg.seq_len,
            hidden=cfg.ae_y_hidden,
            bottleneck=cfg.ae_y_bottleneck,
        )
        self.weather_branch = WeatherResidualSeqBranch(
            hist_input_dim=1 + 5,
            shift_exog_dim=5,
            conv1_channels=cfg.weather_conv1_channels,
            conv2_channels=cfg.weather_conv2_channels,
            kernel_size=cfg.weather_kernel,
            lstm_hidden1=cfg.weather_lstm_hidden1,
            lstm_hidden2=cfg.weather_lstm_hidden2,
            attn_units=cfg.weather_attn_units,
            fc_hidden=cfg.fc_hidden,
            dropout=cfg.dropout,
        )
        if cfg.use_event_expert:
            self.event_expert = EventExpertCorrection(
                event_expert_hidden=cfg.event_expert_hidden,
                event_gate_hidden=cfg.event_gate_hidden,
            )
        else:
            self.event_expert = None
        self._init_shape_fusion()

    def _init_shape_fusion(self):
        with torch.no_grad():
            self.shape_fusion.weight.fill_(0.5)
            self.shape_fusion.bias.zero_()

    def forward(
        self,
        y_hist: torch.Tensor,
        time_hist_scalar: torch.Tensor,
        time_shift_scalar: torch.Tensor,
        weather_hist: torch.Tensor,
        weather_shift: torch.Tensor,
        time_enc_hist: torch.Tensor,
        time_enc_shift: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del time_hist_scalar, time_enc_hist
        p_shift = self.periodic_branch(time_shift_scalar)
        t_shift = self.trend_branch(y_hist)
        s_shift = self.shape_fusion(torch.cat([p_shift, t_shift], dim=-1))

        y_ae_recon_hist = self.y_autoencoder(y_hist)
        base_resid_hist = y_hist - y_ae_recon_hist
        rw_raw_shift = self.weather_branch(
            base_resid_hist,
            weather_hist,
            weather_shift,
        )

        daylight_hist = weather_hist[..., 3:4]
        daylight_shift = weather_shift[..., 3:4]
        rw_shift = rw_raw_shift
        rw_hist_proxy = torch.cat(
            [torch.zeros_like(rw_shift[:, :1, :]), rw_shift[:, :-1, :]],
            dim=1,
        )
        zero_resid = torch.zeros_like(rw_shift)
        rw_ae_recon_hist = torch.zeros_like(rw_hist_proxy)
        weather_unexplained_hist = torch.zeros_like(rw_hist_proxy)
        user_resid_hist = torch.zeros_like(base_resid_hist)
        ru_raw_shift = zero_resid
        ru_shift = zero_resid
        user_day_gate = torch.ones_like(daylight_shift)
        resid_ae_recon_hist = torch.zeros_like(base_resid_hist)
        resid_ae_error_hist = torch.zeros_like(base_resid_hist)

        y_base_shift_pred = s_shift + rw_shift
        if self.event_expert is not None:
            event_outputs = self.event_expert(
                y_base_shift_pred=y_base_shift_pred,
                s_shift=s_shift,
                r_shift=rw_shift,
                y_hist=y_hist,
                weather_shift=weather_shift,
                time_enc_shift=time_enc_shift,
            )
            y_shift_pred = y_base_shift_pred + event_outputs["event_correction_shift"]
        else:
            event_outputs = {
                "event_gate_shift": torch.zeros_like(rw_shift),
                "event_correction_raw_shift": torch.zeros_like(rw_shift),
                "event_correction_shift": torch.zeros_like(rw_shift),
            }
            y_shift_pred = y_base_shift_pred
        y_pred_future = y_shift_pred[:, -1, :]
        y_base_pred_future = y_base_shift_pred[:, -1, :]
        event_correction_future = event_outputs["event_correction_shift"][:, -1, :]
        return {
            "y_shift_pred": y_shift_pred,
            "y_pred_future": y_pred_future,
            "y_base_shift_pred": y_base_shift_pred,
            "y_base_pred_future": y_base_pred_future,
            "S_shift": s_shift,
            "P_shift": p_shift,
            "T_shift": t_shift,
            "Rw_raw_shift": rw_raw_shift,
            "Rw_shift": rw_shift,
            "Rw_hist_proxy": rw_hist_proxy,
            "rw_ae_recon_hist": rw_ae_recon_hist,
            "weather_unexplained_hist": weather_unexplained_hist,
            "user_resid_hist": user_resid_hist,
            "Ru_raw_shift": ru_raw_shift,
            "Ru_shift": ru_shift,
            "daylight_hist": daylight_hist,
            "daylight_shift": daylight_shift,
            "user_day_gate": user_day_gate,
            "y_ae_recon_hist": y_ae_recon_hist,
            "base_resid_hist": base_resid_hist,
            "resid_ae_recon_hist": resid_ae_recon_hist,
            "resid_ae_error_hist": resid_ae_error_hist,
            "event_gate_shift": event_outputs["event_gate_shift"],
            "event_correction_raw_shift": event_outputs["event_correction_raw_shift"],
            "event_correction_shift": event_outputs["event_correction_shift"],
            "event_correction_future": event_correction_future,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weather-aware additive AE-shift decomposition with free residual and event expert correction."
    )
    parser.add_argument("--client-id", type=int, default=2, help="1-based client id. Omit to train all clients.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("runs", "weather_aware_additive_ae_shift_ste_free_resid_event_expert"),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.train.lr)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=CFG.train.random_seed)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--ghi-threshold", type=float, default=10.0)
    parser.add_argument("--smooth-k", type=int, default=3)
    parser.add_argument("--lambda-next", type=float, default=1.0)
    parser.add_argument("--lambda-ste", type=float, default=1.0)
    parser.add_argument("--alpha-ste", type=float, default=0.1)
    parser.add_argument("--lambda-trend-smooth", type=float, default=0.0)
    parser.add_argument("--lambda-ae-y", type=float, default=0.0)
    parser.add_argument("--lambda-ae-resid", type=float, default=0.0)
    parser.add_argument("--lambda-ru-mean", type=float, default=0.0)
    parser.add_argument("--lambda-rw-ae", type=float, default=0.0)
    parser.add_argument("--user-resid-scale", type=float, default=0.3)
    parser.add_argument("--user-day-scale", type=float, default=0.5)
    parser.add_argument("--detach-user-resid-input", action="store_true", default=True)
    parser.add_argument("--no-detach-user-resid-input", action="store_false", dest="detach_user_resid_input")
    parser.add_argument("--num-basis", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--component-eval-every", type=int, default=1)
    parser.add_argument("--component-sample-n", type=int, default=300)
    parser.add_argument("--plot-n", type=int, default=500)
    parser.add_argument("--device", default=CFG.train.device)
    parser.add_argument("--use-event-expert", action="store_true", default=True)
    parser.add_argument("--no-use-event-expert", action="store_false", dest="use_event_expert")
    parser.add_argument("--event-expert-hidden", type=int, default=16)
    parser.add_argument("--event-gate-hidden", type=int, default=16)
    parser.add_argument("--lambda-event-corr", type=float, default=0.1)
    parser.add_argument("--lambda-event-gate", type=float, default=0.02)
    parser.add_argument("--lambda-event-normal", type=float, default=0.01)
    parser.add_argument("--event-weight-beta", type=float, default=2.0)
    parser.add_argument("--event-ramp-quantile", type=float, default=0.8)
    parser.add_argument("--event-ghi-quantile", type=float, default=0.8)
    parser.add_argument("--event-peak-quantile", type=float, default=0.8)
    parser.add_argument("--event-tau-ramp", type=float, default=0.05)
    parser.add_argument("--event-tau-ghi", type=float, default=0.05)
    parser.add_argument("--event-tau-zero", type=float, default=0.05)
    parser.add_argument("--event-tau-peak", type=float, default=0.05)
    parser.add_argument("--event-zero-eps", type=float, default=0.05)
    parser.add_argument("--detach-event-base-error", action="store_true", default=True)
    parser.add_argument("--no-detach-event-base-error", action="store_false", dest="detach_event_base_error")
    return parser.parse_args()


def normalize_device(device_name: str) -> torch.device:
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is not available. Falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(device_name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_client_id_from_path(path: str, fallback: int) -> int:
    stem = os.path.basename(path)
    for token in stem.replace("-", "_").split("_"):
        if token.isdigit():
            return int(token)
    return int(fallback)


def scaler_to_dict(scaler) -> Dict:
    if scaler is None:
        return {"type": "none"}
    result = {"type": scaler.__class__.__name__}
    for name in ["data_min_", "data_max_", "data_range_", "scale_", "min_", "mean_", "var_"]:
        if hasattr(scaler, name):
            result[name.rstrip("_")] = np.asarray(getattr(scaler, name), dtype=float).reshape(-1).tolist()
    return result


def target_affine_params(y_scaler) -> Tuple[float, float]:
    if y_scaler is None:
        return 1.0, 0.0
    zero = y_scaler.inverse_transform(np.array([[0.0]], dtype=np.float64))[0, 0]
    one = y_scaler.inverse_transform(np.array([[1.0]], dtype=np.float64))[0, 0]
    return float(one - zero), float(zero)


def inverse_target(y_scaler, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if y_scaler is None:
        return values.reshape(-1)
    return y_scaler.inverse_transform(values).reshape(-1)


def component_to_real(values: np.ndarray, scale: float, offset: float = 0.0) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1) * scale + offset


def read_weather_aware_dataframe(csv_path: str, cfg: WeatherAwareAEShiftExperimentConfig) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find client CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    timestamp_col = CFG.data.datetime_col
    required = [timestamp_col, "ghi_wm2", "wind10m_ms"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    if "temp2m_c" not in df.columns and "temp2m_k" not in df.columns:
        raise ValueError(f"{csv_path} must contain temp2m_c or temp2m_k.")
    if "net_load" not in df.columns:
        if "gc" not in df.columns or "gg" not in df.columns:
            raise ValueError(f"{csv_path} must contain net_load or both gc and gg.")
        df["net_load"] = pd.to_numeric(df["gc"], errors="coerce") - pd.to_numeric(df["gg"], errors="coerce")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["net_load"] = pd.to_numeric(df["net_load"], errors="coerce")
    df["ghi_wm2"] = pd.to_numeric(df["ghi_wm2"], errors="coerce")
    df["wind10m_ms"] = pd.to_numeric(df["wind10m_ms"], errors="coerce")
    if "temp2m_c" in df.columns:
        df["temp_c"] = pd.to_numeric(df["temp2m_c"], errors="coerce")
        temp_source = "temp2m_c"
    else:
        df["temp_c"] = pd.to_numeric(df["temp2m_k"], errors="coerce") - 273.15
        temp_source = "temp2m_k_minus_273.15"

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df["time_scalar_norm"] = np.arange(len(df), dtype=np.float64) / float(cfg.seq_len)
    df["daylight"] = (df["ghi_wm2"] > cfg.ghi_threshold).astype(float)
    df["ramp_ghi"] = df["ghi_wm2"].diff().fillna(0.0)

    dt = df[timestamp_col]
    hour_float = dt.dt.hour.astype(float) + dt.dt.minute.astype(float) / 60.0
    hour_angle = 2.0 * np.pi * hour_float / 24.0
    weekday_angle = 2.0 * np.pi * dt.dt.weekday.astype(float) / 7.0
    month_angle = 2.0 * np.pi * (dt.dt.month.astype(float) - 1.0) / 12.0
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["weekday_sin"] = np.sin(weekday_angle)
    df["weekday_cos"] = np.cos(weekday_angle)
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(float)

    needed = [
        timestamp_col,
        "net_load",
        "ghi_wm2",
        "temp_c",
        "wind10m_ms",
        "daylight",
        "ramp_ghi",
        "time_scalar_norm",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
    ]
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    if len(df) <= cfg.seq_len:
        raise ValueError(f"Not enough clean rows ({len(df)}) for seq_len={cfg.seq_len}.")
    df.attrs["raw_rows"] = int(before)
    df.attrs["clean_rows"] = int(len(df))
    df.attrs["temp_source"] = temp_source
    return df


def split_df_by_ratio(df: pd.DataFrame, train_ratio: float, val_ratio: float, seq_len: int):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= seq_len or val_end <= train_end or n <= val_end + seq_len:
        raise ValueError(
            f"Invalid split sizes for n={n}, seq_len={seq_len}: "
            f"train_end={train_end}, val_end={val_end}."
        )
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def fit_transform_feature_groups(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    y_scaler = get_scaler(CFG.train.scaler_y)
    raw_weather_scaler = get_scaler(CFG.train.scaler_x)
    ramp_scaler = get_scaler(CFG.train.scaler_x)

    out = {
        "train": train_df.copy(),
        "val": val_df.copy(),
        "test": test_df.copy(),
    }

    y_scaler.fit(train_df[["net_load"]].values)
    raw_weather_cols = ["ghi_wm2", "temp_c", "wind10m_ms"]
    raw_weather_scaler.fit(train_df[raw_weather_cols].values)
    ramp_scaler.fit(train_df[["ramp_ghi"]].values)

    for split in out.values():
        split.loc[:, "y_scaled"] = y_scaler.transform(split[["net_load"]].values).astype(np.float32)
        raw_scaled = raw_weather_scaler.transform(split[raw_weather_cols].values).astype(np.float32)
        split.loc[:, ["ghi_scaled", "temp_scaled", "wind_scaled"]] = raw_scaled
        split.loc[:, "ramp_ghi_scaled"] = ramp_scaler.transform(split[["ramp_ghi"]].values).astype(np.float32)

    scalers = {
        "y_scaler": y_scaler,
        "raw_weather_scaler": raw_weather_scaler,
        "ramp_scaler": ramp_scaler,
    }
    return out["train"], out["val"], out["test"], scalers


def compute_event_thresholds(train_scaled: pd.DataFrame, cfg: WeatherAwareAEShiftExperimentConfig) -> Dict[str, float]:
    train_abs_dy = np.abs(np.diff(train_scaled["y_scaled"].values.astype(np.float64)))
    train_abs_dghi = np.abs(np.diff(train_scaled["ghi_scaled"].values.astype(np.float64)))
    y_scaled = train_scaled["y_scaled"].values.astype(np.float64)
    thresholds = {
        "event_ramp_threshold": float(np.quantile(train_abs_dy, cfg.event_ramp_quantile)) if len(train_abs_dy) else 0.0,
        "event_ghi_threshold": float(np.quantile(train_abs_dghi, cfg.event_ghi_quantile)) if len(train_abs_dghi) else 0.0,
        "event_peak_threshold": float(np.quantile(y_scaled, cfg.event_peak_quantile)) if len(y_scaled) else 0.0,
    }
    return thresholds


def create_sequences_for_split(split_df: pd.DataFrame, seq_len: int) -> Dict[str, np.ndarray]:
    if len(split_df) <= seq_len:
        raise ValueError(f"Split has only {len(split_df)} rows; seq_len={seq_len}.")

    y_scaled = split_df["y_scaled"].values.astype(np.float32)
    time_scalar = split_df["time_scalar_norm"].values.astype(np.float32)
    weather = split_df[
        ["ghi_scaled", "temp_scaled", "wind_scaled", "daylight", "ramp_ghi_scaled"]
    ].values.astype(np.float32)
    time_enc = split_df[
        ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos", "is_weekend"]
    ].values.astype(np.float32)
    timestamps = split_df[CFG.data.datetime_col].astype(str).values

    arrays = {
        "y_hist": [],
        "y_shift": [],
        "time_hist_scalar": [],
        "time_shift_scalar": [],
        "weather_hist": [],
        "weather_shift": [],
        "time_enc_hist": [],
        "time_enc_shift": [],
        "timestamp_future": [],
        "timestamp_shift": [],
    }

    for end_idx in range(seq_len, len(split_df)):
        start_idx = end_idx - seq_len
        hist_slice = slice(start_idx, end_idx)
        shift_slice = slice(start_idx + 1, end_idx + 1)

        arrays["y_hist"].append(y_scaled[hist_slice].reshape(seq_len, 1))
        arrays["y_shift"].append(y_scaled[shift_slice].reshape(seq_len, 1))
        arrays["time_hist_scalar"].append(time_scalar[hist_slice].reshape(seq_len, 1))
        arrays["time_shift_scalar"].append(time_scalar[shift_slice].reshape(seq_len, 1))
        arrays["weather_hist"].append(weather[hist_slice])
        arrays["weather_shift"].append(weather[shift_slice])
        arrays["time_enc_hist"].append(time_enc[hist_slice])
        arrays["time_enc_shift"].append(time_enc[shift_slice])
        arrays["timestamp_future"].append(str(timestamps[end_idx]))
        arrays["timestamp_shift"].append("|".join(str(ts) for ts in timestamps[shift_slice]))

    for key in [
        "y_hist",
        "y_shift",
        "time_hist_scalar",
        "time_shift_scalar",
        "weather_hist",
        "weather_shift",
        "time_enc_hist",
        "time_enc_shift",
    ]:
        arrays[key] = np.asarray(arrays[key], dtype=np.float32)
    arrays["timestamp_future"] = np.asarray(arrays["timestamp_future"], dtype=object)
    arrays["timestamp_shift"] = np.asarray(arrays["timestamp_shift"], dtype=object)
    return arrays


def make_loader(dataset: WeatherAwareAEShiftDataset, batch_size: int, device: torch.device, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.train.num_workers,
        pin_memory=CFG.train.pin_memory and str(device).startswith("cuda"),
    )


def prepare_client_data(csv_path: str, cfg: WeatherAwareAEShiftExperimentConfig, batch_size: int, device: torch.device):
    df = read_weather_aware_dataframe(csv_path, cfg)
    train_df, val_df, test_df = split_df_by_ratio(df, cfg.train_ratio, cfg.val_ratio, cfg.seq_len)
    train_scaled, val_scaled, test_scaled, scalers = fit_transform_feature_groups(train_df, val_df, test_df)
    event_thresholds = compute_event_thresholds(train_scaled, cfg)
    for name, value in event_thresholds.items():
        setattr(cfg, name, value)

    train_arrays = create_sequences_for_split(train_scaled, cfg.seq_len)
    val_arrays = create_sequences_for_split(val_scaled, cfg.seq_len)
    test_arrays = create_sequences_for_split(test_scaled, cfg.seq_len)

    data = {
        "raw_df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_samples": int(train_arrays["y_shift"].shape[0]),
        "val_samples": int(val_arrays["y_shift"].shape[0]),
        "test_samples": int(test_arrays["y_shift"].shape[0]),
        "train_loader": make_loader(WeatherAwareAEShiftDataset(train_arrays), batch_size, device, shuffle=True),
        "val_loader": make_loader(WeatherAwareAEShiftDataset(val_arrays), batch_size, device, shuffle=False),
        "test_loader": make_loader(WeatherAwareAEShiftDataset(test_arrays), batch_size, device, shuffle=False),
        "scalers": scalers,
        "event_thresholds": event_thresholds,
        "split_info": {
            "raw_rows": int(df.attrs.get("raw_rows", len(df))),
            "clean_rows": int(df.attrs.get("clean_rows", len(df))),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_samples": int(train_arrays["y_shift"].shape[0]),
            "val_samples": int(val_arrays["y_shift"].shape[0]),
            "test_samples": int(test_arrays["y_shift"].shape[0]),
            "temp_source": df.attrs.get("temp_source", "unknown"),
        },
    }
    return data


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if key in {"timestamp_future", "timestamp_shift"}:
            out[key] = value
        else:
            out[key] = value.to(device=device, dtype=torch.float32)
    return out


def compute_fourier_loss(pred_seq: torch.Tensor, true_seq: torch.Tensor) -> torch.Tensor:
    """MSE between rFFT amplitude spectra for [B, H] or [B, H, 1] sequences."""
    if pred_seq.dim() == 3 and pred_seq.size(-1) == 1:
        pred_seq = pred_seq.squeeze(-1)
    if true_seq.dim() == 3 and true_seq.size(-1) == 1:
        true_seq = true_seq.squeeze(-1)

    pred_fft = torch.fft.rfft(pred_seq, dim=1)
    true_fft = torch.fft.rfft(true_seq, dim=1)
    return F.mse_loss(torch.abs(pred_fft), torch.abs(true_fft))


def compute_weighted_sequence_loss(
    pred_seq: torch.Tensor,
    true_seq: torch.Tensor,
    daylight_shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sequence MSE with extra emphasis on daylight and high-ramp points."""
    abs_ramp = torch.zeros_like(true_seq)
    abs_ramp[:, 1:, :] = torch.abs(true_seq[:, 1:, :] - true_seq[:, :-1, :])
    ramp_norm = abs_ramp / (torch.mean(abs_ramp.detach()) + eps)
    weights = 1.0 + daylight_shift + ramp_norm
    sq_error = (pred_seq - true_seq) ** 2
    return torch.sum(weights * sq_error) / (torch.sum(weights) + eps)


def compute_trend_smooth_loss(trend_seq: torch.Tensor) -> torch.Tensor:
    """Second-difference smoothness penalty for the trend branch."""
    if trend_seq.size(1) < 3:
        return torch.zeros((), dtype=trend_seq.dtype, device=trend_seq.device)
    second_diff = trend_seq[:, 2:, :] - 2.0 * trend_seq[:, 1:-1, :] + trend_seq[:, :-2, :]
    return torch.mean(second_diff ** 2)


def _threshold_tensor(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(float(value), dtype=like.dtype, device=like.device)


def compute_event_soft_mask(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    cfg: WeatherAwareAEShiftExperimentConfig,
) -> torch.Tensor:
    y_shift = batch["y_shift"]
    weather_shift = batch["weather_shift"]
    dy = torch.zeros_like(y_shift)
    dy[:, 1:, :] = torch.abs(y_shift[:, 1:, :] - y_shift[:, :-1, :])

    ramp_thr = _threshold_tensor(getattr(cfg, "event_ramp_threshold", 0.0), y_shift)
    ghi_thr = _threshold_tensor(getattr(cfg, "event_ghi_threshold", 0.0), y_shift)
    peak_thr = _threshold_tensor(getattr(cfg, "event_peak_threshold", 0.0), y_shift)

    tau_ramp = max(float(cfg.event_tau_ramp), 1e-6)
    tau_ghi = max(float(cfg.event_tau_ghi), 1e-6)
    tau_zero = max(float(cfg.event_tau_zero), 1e-6)
    tau_peak = max(float(cfg.event_tau_peak), 1e-6)

    m_ramp = torch.sigmoid((dy - ramp_thr) / tau_ramp)
    dghi = torch.abs(weather_shift[..., 4:5])
    m_ghi = torch.sigmoid((dghi - ghi_thr) / tau_ghi)
    m_zero = torch.sigmoid((float(cfg.event_zero_eps) - torch.abs(outputs["y_base_shift_pred"])) / tau_zero)
    m_peak = torch.sigmoid((y_shift - peak_thr) / tau_peak)
    event_mask_shift = torch.max(torch.max(m_ramp, m_ghi), torch.max(m_zero, m_peak))
    return event_mask_shift.detach()


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: WeatherAwareAEShiftExperimentConfig,
):
    y_shift = batch["y_shift"]
    y_shift_pred = outputs["y_shift_pred"]
    loss_seq = F.mse_loss(y_shift_pred, y_shift)
    loss_fourier = compute_fourier_loss(y_shift_pred, y_shift)
    loss_ste = cfg.alpha_ste * loss_fourier + (1.0 - cfg.alpha_ste) * loss_seq
    y_true_future = y_shift[:, -1, :]
    event_mask_shift = compute_event_soft_mask(batch, outputs, cfg)
    event_mask_future = event_mask_shift[:, -1, :]
    next_sq = (outputs["y_pred_future"] - y_true_future) ** 2
    loss_next = torch.mean((1.0 + float(cfg.event_weight_beta) * event_mask_future) * next_sq)

    zero = torch.zeros_like(loss_ste)
    loss_weighted_seq = loss_seq
    loss_trend_smooth = zero
    loss_ae_y = zero
    loss_ae_resid = zero
    loss_ru_mean = zero
    loss_rw_ae = zero

    weighted_loss_next = cfg.lambda_next * loss_next
    weighted_loss_ste = cfg.lambda_ste * loss_ste
    weighted_loss_trend_smooth = zero
    weighted_loss_ae_y = zero
    weighted_loss_ae_resid = zero
    weighted_loss_ru_mean = zero
    weighted_loss_rw_ae = zero
    loss = weighted_loss_ste + weighted_loss_next
    if cfg.use_event_expert:
        base_error = y_shift - outputs["y_base_shift_pred"]
        if cfg.detach_event_base_error:
            base_error = base_error.detach()
        event_correction_shift = outputs["event_correction_shift"]
        loss_event_corr = torch.mean(event_mask_shift * (event_correction_shift - base_error) ** 2)
        q = outputs["event_gate_shift"].clamp(1e-6, 1.0 - 1e-6)
        loss_event_gate = F.binary_cross_entropy(q, event_mask_shift)
        loss_event_normal = torch.mean((1.0 - event_mask_shift) * event_correction_shift ** 2)
        weighted_loss_event_corr = cfg.lambda_event_corr * loss_event_corr
        weighted_loss_event_gate = cfg.lambda_event_gate * loss_event_gate
        weighted_loss_event_normal = cfg.lambda_event_normal * loss_event_normal
        loss = loss + weighted_loss_event_corr + weighted_loss_event_gate + weighted_loss_event_normal
    else:
        loss_event_corr = zero
        loss_event_gate = zero
        loss_event_normal = zero
        weighted_loss_event_corr = zero
        weighted_loss_event_gate = zero
        weighted_loss_event_normal = zero

    event_correction_abs = torch.abs(outputs["event_correction_shift"])
    event_den = torch.sum(event_mask_shift) + 1e-6
    normal_den = torch.sum(1.0 - event_mask_shift) + 1e-6
    return {
        "loss": loss,
        "loss_next": loss_next,
        "weighted_loss_next": weighted_loss_next,
        "loss_ste": loss_ste,
        "loss_seq": loss_seq,
        "loss_weighted_seq": loss_weighted_seq,
        "loss_fourier": loss_fourier,
        "weighted_loss_ste": weighted_loss_ste,
        "loss_trend_smooth": loss_trend_smooth,
        "weighted_loss_trend_smooth": weighted_loss_trend_smooth,
        "loss_ae_y": loss_ae_y,
        "weighted_loss_ae_y": weighted_loss_ae_y,
        "loss_ae_resid": loss_ae_resid,
        "weighted_loss_ae_resid": weighted_loss_ae_resid,
        "loss_ru_mean": loss_ru_mean,
        "weighted_loss_ru_mean": weighted_loss_ru_mean,
        "loss_rw_ae": loss_rw_ae,
        "weighted_loss_rw_ae": weighted_loss_rw_ae,
        "loss_event_corr": loss_event_corr,
        "weighted_loss_event_corr": weighted_loss_event_corr,
        "loss_event_gate": loss_event_gate,
        "weighted_loss_event_gate": weighted_loss_event_gate,
        "loss_event_normal": loss_event_normal,
        "weighted_loss_event_normal": weighted_loss_event_normal,
        "event_mask_mean": torch.mean(event_mask_shift),
        "event_gate_mean": torch.mean(outputs["event_gate_shift"]),
        "event_corr_abs_mean": torch.mean(event_correction_abs),
        "event_corr_abs_event_mean": torch.sum(event_mask_shift * event_correction_abs) / event_den,
        "event_corr_abs_normal_mean": torch.sum((1.0 - event_mask_shift) * event_correction_abs) / normal_den,
    }


def _as_list(values) -> List[str]:
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    if hasattr(values, "tolist"):
        out = values.tolist()
        if isinstance(out, list):
            return [str(v) for v in out]
    return [str(values)]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: WeatherAwareAEShiftExperimentConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
    collect_outputs: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "loss_next": 0.0,
        "weighted_loss_next": 0.0,
        "loss_ste": 0.0,
        "loss_seq": 0.0,
        "loss_weighted_seq": 0.0,
        "loss_fourier": 0.0,
        "weighted_loss_ste": 0.0,
        "loss_trend_smooth": 0.0,
        "weighted_loss_trend_smooth": 0.0,
        "loss_ae_y": 0.0,
        "weighted_loss_ae_y": 0.0,
        "loss_ae_resid": 0.0,
        "weighted_loss_ae_resid": 0.0,
        "loss_ru_mean": 0.0,
        "weighted_loss_ru_mean": 0.0,
        "loss_rw_ae": 0.0,
        "weighted_loss_rw_ae": 0.0,
        "loss_event_corr": 0.0,
        "weighted_loss_event_corr": 0.0,
        "loss_event_gate": 0.0,
        "weighted_loss_event_gate": 0.0,
        "loss_event_normal": 0.0,
        "weighted_loss_event_normal": 0.0,
        "event_mask_mean": 0.0,
        "event_gate_mean": 0.0,
        "event_corr_abs_mean": 0.0,
        "event_corr_abs_event_mean": 0.0,
        "event_corr_abs_normal_mean": 0.0,
    }
    sample_count = 0
    collected = {
        "timestamp": [],
        "timestamp_shift": [],
        "y_true": [],
        "y_pred": [],
        "y_base_pred": [],
        "S_shift": [],
        "P_shift": [],
        "T_shift": [],
        "Rw_raw_shift": [],
        "Rw_shift": [],
        "Ru_raw_shift": [],
        "Ru_shift": [],
        "daylight_shift": [],
        "user_day_gate": [],
        "event_gate_shift": [],
        "event_correction_raw_shift": [],
        "event_correction_shift": [],
        "y_shift_true_seq": [],
        "y_shift_pred_seq": [],
        "S_shift_seq": [],
        "Rw_shift_seq": [],
        "Ru_shift_seq": [],
        "event_correction_shift_seq": [],
    }

    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(
                y_hist=batch["y_hist"],
                time_hist_scalar=batch["time_hist_scalar"],
                time_shift_scalar=batch["time_shift_scalar"],
                weather_hist=batch["weather_hist"],
                weather_shift=batch["weather_shift"],
                time_enc_hist=batch["time_enc_hist"],
                time_enc_shift=batch["time_enc_shift"],
            )
            losses = compute_losses(outputs, batch, cfg)
            if is_train:
                losses["loss"].backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()

        batch_size = int(batch["y_shift"].shape[0])
        sample_count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().cpu().item()) * batch_size

        if collect_outputs:
            collected["timestamp"].extend(_as_list(raw_batch["timestamp_future"]))
            collected["timestamp_shift"].extend(_as_list(raw_batch["timestamp_shift"]))
            collected["y_true"].append(batch["y_shift"][:, -1, :].detach().cpu().numpy())
            collected["y_pred"].append(outputs["y_pred_future"].detach().cpu().numpy())
            collected["y_base_pred"].append(outputs["y_base_pred_future"].detach().cpu().numpy())
            for key in [
                "S_shift",
                "P_shift",
                "T_shift",
                "Rw_raw_shift",
                "Rw_shift",
                "Ru_raw_shift",
                "Ru_shift",
                "daylight_shift",
                "user_day_gate",
                "event_gate_shift",
                "event_correction_raw_shift",
                "event_correction_shift",
            ]:
                collected[key].append(outputs[key][:, -1, :].detach().cpu().numpy())
            collected["y_shift_true_seq"].append(batch["y_shift"].detach().cpu().numpy())
            collected["y_shift_pred_seq"].append(outputs["y_shift_pred"].detach().cpu().numpy())
            for key in ["S_shift", "Rw_shift", "Ru_shift"]:
                collected[f"{key}_seq"].append(outputs[key].detach().cpu().numpy())
            collected["event_correction_shift_seq"].append(outputs["event_correction_shift"].detach().cpu().numpy())

    if sample_count <= 0:
        raise ValueError("DataLoader has no samples.")
    stats = {key: value / sample_count for key, value in totals.items()}

    if not collect_outputs:
        return stats, None

    flattened = {
        "timestamp": np.asarray(collected["timestamp"], dtype=object),
        "timestamp_shift": np.asarray(collected["timestamp_shift"], dtype=object),
    }
    for key, chunks in collected.items():
        if key in {"timestamp", "timestamp_shift"}:
            continue
        if key.endswith("_seq"):
            flattened[key] = np.concatenate(chunks, axis=0)
        else:
            flattened[key] = np.concatenate(chunks, axis=0).reshape(-1)
    return stats, flattened


def build_prediction_dataframe(collected: Dict[str, np.ndarray], y_scaler) -> pd.DataFrame:
    target_scale, target_offset = target_affine_params(y_scaler)
    y_true = inverse_target(y_scaler, collected["y_true"])
    y_pred = inverse_target(y_scaler, collected["y_pred"])
    y_base_pred = inverse_target(y_scaler, collected["y_base_pred"])

    s_real = component_to_real(collected["S_shift"], target_scale, target_offset)
    rw_raw_real = component_to_real(collected["Rw_raw_shift"], target_scale, 0.0)
    rw_real = component_to_real(collected["Rw_shift"], target_scale, 0.0)
    ru_raw_real = component_to_real(collected["Ru_raw_shift"], target_scale, 0.0)
    ru_real = component_to_real(collected["Ru_shift"], target_scale, 0.0)
    event_correction_raw_real = component_to_real(collected["event_correction_raw_shift"], target_scale, 0.0)
    event_correction_real = component_to_real(collected["event_correction_shift"], target_scale, 0.0)
    s_dev = component_to_real(collected["S_shift"], target_scale, 0.0)
    rw_dev = component_to_real(collected["Rw_shift"], target_scale, 0.0)
    ru_dev = component_to_real(collected["Ru_shift"], target_scale, 0.0)
    event_correction_raw_dev = component_to_real(collected["event_correction_raw_shift"], target_scale, 0.0)
    event_correction_dev = component_to_real(collected["event_correction_shift"], target_scale, 0.0)
    p_internal = component_to_real(collected["P_shift"], target_scale, 0.0)
    t_internal = component_to_real(collected["T_shift"], target_scale, 0.0)
    additive_sum = s_real + rw_real + ru_real + event_correction_real

    return pd.DataFrame(
        {
            "timestamp": collected["timestamp"],
            "y_true": y_true,
            "y_pred": y_pred,
            "y_base_pred": y_base_pred,
            "component_additive_sum": additive_sum,
            "component_additive_error": additive_sum - y_pred,
            "S_real": s_real,
            "Rw_raw_real": rw_raw_real,
            "Rw_real": rw_real,
            "Ru_raw_real": ru_raw_real,
            "Ru_real": ru_real,
            "event_gate": collected["event_gate_shift"],
            "event_correction_raw_real": event_correction_raw_real,
            "event_correction_real": event_correction_real,
            "S_dev": s_dev,
            "Rw_dev": rw_dev,
            "Ru_dev": ru_dev,
            "event_correction_raw_dev": event_correction_raw_dev,
            "event_correction_dev": event_correction_dev,
            "P_future": p_internal,
            "T_future": t_internal,
            "S_scaled": collected["S_shift"],
            "Rw_raw_scaled": collected["Rw_raw_shift"],
            "Rw_scaled": collected["Rw_shift"],
            "Ru_raw_scaled": collected["Ru_raw_shift"],
            "Ru_scaled": collected["Ru_shift"],
            "event_correction_raw_scaled": collected["event_correction_raw_shift"],
            "event_correction_scaled": collected["event_correction_shift"],
            "daylight_shift": collected["daylight_shift"],
            "user_day_gate": collected["user_day_gate"],
            "P_scaled": collected["P_shift"],
            "T_scaled": collected["T_shift"],
            "y_true_scaled": collected["y_true"],
            "y_pred_scaled": collected["y_pred"],
            "y_base_pred_scaled": collected["y_base_pred"],
        }
    )


def compute_component_stats(collected: Dict[str, np.ndarray], y_scaler, eps: float = 1e-12) -> List[Dict[str, float]]:
    target_scale, _ = target_affine_params(y_scaler)
    components = {
        "S_dev": component_to_real(collected["S_shift"], target_scale, 0.0),
        "Rw_dev": component_to_real(collected["Rw_shift"], target_scale, 0.0),
        "Ru_dev": component_to_real(collected["Ru_shift"], target_scale, 0.0),
        "Event_dev": component_to_real(collected["event_correction_shift"], target_scale, 0.0),
    }
    dyn_components = {name: values - np.mean(values) for name, values in components.items()}

    mean_abs = {name: float(np.mean(np.abs(values))) for name, values in components.items()}
    mean_energy = {name: float(np.mean(values ** 2)) for name, values in components.items()}
    dyn_mean_abs = {name: float(np.mean(np.abs(values))) for name, values in dyn_components.items()}
    den_abs = sum(mean_abs.values()) + eps
    den_energy = sum(mean_energy.values()) + eps
    den_dyn_abs = sum(dyn_mean_abs.values()) + eps

    rows = []
    for name, values in components.items():
        dyn_values = dyn_components[name]
        rows.append(
            {
                "component": name,
                "mean": float(np.mean(values)),
                "mean_abs": mean_abs[name],
                "rms": float(np.sqrt(np.mean(values ** 2))),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "share_abs": float(mean_abs[name] / den_abs),
                "share_energy": float(mean_energy[name] / den_energy),
                "dyn_mean": float(np.mean(dyn_values)),
                "dyn_mean_abs": dyn_mean_abs[name],
                "dyn_rms": float(np.sqrt(np.mean(dyn_values ** 2))),
                "dyn_std": float(np.std(dyn_values)),
                "share_dyn_abs": float(dyn_mean_abs[name] / den_dyn_abs),
            }
        )
    return rows


def component_share_dict(stats_rows: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    name_map = {"S_dev": "S", "Rw_dev": "Rw", "Ru_dev": "Ru", "Event_dev": "Event"}
    out = {}
    for row in stats_rows:
        name = name_map[row["component"]]
        out[f"{prefix}_share_{name}_abs"] = row["share_abs"]
        out[f"{prefix}_share_{name}_dyn_abs"] = row["share_dyn_abs"]
        out[f"{prefix}_share_{name}_energy"] = row["share_energy"]
    return out


def _masked_mae(true_values: np.ndarray, pred_values: np.ndarray, mask: np.ndarray) -> float:
    if int(np.sum(mask)) <= 0:
        return float("nan")
    return float(np.mean(np.abs(true_values[mask] - pred_values[mask])))


def compute_event_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    y_true = pred_df["y_true"].values.astype(np.float64)
    y_pred = pred_df["y_pred"].values.astype(np.float64)
    y_base_pred = pred_df["y_base_pred"].values.astype(np.float64)
    event_gate = pred_df["event_gate"].values.astype(np.float64)
    event_corr = pred_df["event_correction_real"].values.astype(np.float64)
    daylight = pred_df["daylight_shift"].values.astype(np.float64)

    if len(y_true) < 2:
        return {
            "MAE_all_base": float(np.mean(np.abs(y_true - y_base_pred))) if len(y_true) else float("nan"),
            "MAE_all_final": float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan"),
        }

    dy_true = np.abs(np.diff(y_true))
    ramp_thr = float(np.quantile(dy_true, 0.8))
    ramp_mask = dy_true > ramp_thr
    normal_mask = ~ramp_mask
    night_mask = daylight[1:] < 0.5
    day_mask = daylight[1:] >= 0.5

    true_next = y_true[1:]
    pred_next = y_pred[1:]
    base_next = y_base_pred[1:]
    gate_next = event_gate[1:]
    corr_next = event_corr[1:]

    return {
        "MAE_all_base": float(np.mean(np.abs(y_true - y_base_pred))),
        "MAE_all_final": float(np.mean(np.abs(y_true - y_pred))),
        "MAE_ramp_base": _masked_mae(true_next, base_next, ramp_mask),
        "MAE_ramp_final": _masked_mae(true_next, pred_next, ramp_mask),
        "MAE_day_ramp_base": _masked_mae(true_next, base_next, ramp_mask & day_mask),
        "MAE_day_ramp_final": _masked_mae(true_next, pred_next, ramp_mask & day_mask),
        "MAE_night_ramp_base": _masked_mae(true_next, base_next, ramp_mask & night_mask),
        "MAE_night_ramp_final": _masked_mae(true_next, pred_next, ramp_mask & night_mask),
        "mean_gate_all": float(np.mean(event_gate)),
        "mean_gate_ramp": float(np.mean(gate_next[ramp_mask])) if np.any(ramp_mask) else float("nan"),
        "mean_gate_normal": float(np.mean(gate_next[normal_mask])) if np.any(normal_mask) else float("nan"),
        "mean_abs_event_corr_all": float(np.mean(np.abs(event_corr))),
        "mean_abs_event_corr_ramp": float(np.mean(np.abs(corr_next[ramp_mask]))) if np.any(ramp_mask) else float("nan"),
        "mean_abs_event_corr_normal": float(np.mean(np.abs(corr_next[normal_mask]))) if np.any(normal_mask) else float("nan"),
    }


def build_component_sample_dataframe(
    collected: Dict[str, np.ndarray],
    y_scaler,
    epoch: int,
    split: str,
    sample_n: int,
) -> pd.DataFrame:
    df = build_prediction_dataframe(collected, y_scaler)
    if sample_n is not None and sample_n > 0:
        df = df.head(int(sample_n)).copy()
    df.insert(0, "split", split)
    df.insert(0, "epoch", int(epoch))
    keep_cols = [
        "epoch",
        "split",
        "timestamp",
        "y_true",
        "y_pred",
        "y_base_pred",
        "S_real",
        "Rw_real",
        "Ru_real",
        "event_gate",
        "event_correction_real",
        "S_dev",
        "Rw_dev",
        "Ru_dev",
        "event_correction_dev",
        "P_future",
        "T_future",
    ]
    return df[keep_cols]


def build_shift_sequence_sample_dataframe(collected: Dict[str, np.ndarray], y_scaler, sample_n: int = 5) -> pd.DataFrame:
    target_scale, target_offset = target_affine_params(y_scaler)
    n = min(int(sample_n), int(collected["y_shift_true_seq"].shape[0]))
    rows = []
    for sample_id in range(n):
        timestamps = str(collected["timestamp_shift"][sample_id]).split("|")
        true_scaled = collected["y_shift_true_seq"][sample_id].reshape(-1)
        pred_scaled = collected["y_shift_pred_seq"][sample_id].reshape(-1)
        s_scaled = collected["S_shift_seq"][sample_id].reshape(-1)
        rw_scaled = collected["Rw_shift_seq"][sample_id].reshape(-1)
        ru_scaled = collected["Ru_shift_seq"][sample_id].reshape(-1)
        event_scaled = collected["event_correction_shift_seq"][sample_id].reshape(-1)
        for step in range(len(true_scaled)):
            rows.append(
                {
                    "sample_id": sample_id,
                    "step": step + 1,
                    "timestamp": timestamps[step] if step < len(timestamps) else "",
                    "y_shift_true": inverse_target(y_scaler, true_scaled[step])[0],
                    "y_shift_pred": inverse_target(y_scaler, pred_scaled[step])[0],
                    "S_shift": component_to_real(s_scaled[step], target_scale, target_offset)[0],
                    "Rw_shift": component_to_real(rw_scaled[step], target_scale, 0.0)[0],
                    "Ru_shift": component_to_real(ru_scaled[step], target_scale, 0.0)[0],
                    "event_correction_shift": component_to_real(event_scaled[step], target_scale, 0.0)[0],
                }
            )
    return pd.DataFrame(rows)


def _plot_lines(df: pd.DataFrame, columns: List[str], title: str, ylabel: str, save_path: str, plot_n: int):
    n = min(len(df), int(plot_n))
    x_axis = np.arange(n)
    plt.figure(figsize=(12, 5))
    for col in columns:
        plt.plot(x_axis, df[col].values[:n], label=col)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_test_diagnostic_plots(pred_df: pd.DataFrame, test_component_stats: List[Dict[str, float]], save_dir: str, plot_n: int):
    _plot_lines(
        pred_df,
        ["y_true", "y_pred"],
        "Test True vs Predicted Net Load",
        "Net load",
        os.path.join(save_dir, "test_true_pred.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["S_real", "Rw_real", "Ru_real", "y_pred"],
        "Test Additive Components: S_real + Rw_real + Ru_real",
        "Net load",
        os.path.join(save_dir, "test_components_additive.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["y_true", "y_base_pred", "y_pred"],
        "Test Base Prediction vs Event-Expert Corrected Prediction",
        "Net load",
        os.path.join(save_dir, "test_base_vs_expert_prediction.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["event_gate"],
        "Test Event Expert Gate",
        "Gate",
        os.path.join(save_dir, "test_event_gate.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["event_correction_real"],
        "Test Event Expert Correction",
        "Net load correction",
        os.path.join(save_dir, "test_event_correction.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["event_gate", "event_correction_real"],
        "Test Event Expert Gate and Correction",
        "Gate / net load correction",
        os.path.join(save_dir, "test_event_expert_signals.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["S_real", "Rw_real", "Ru_real", "event_correction_real", "y_pred"],
        "Test Additive Components with Event Correction",
        "Net load",
        os.path.join(save_dir, "test_components_additive_with_event.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["S_dev", "Rw_dev", "Ru_dev"],
        "Test Dynamic Component Contributions",
        "Net load contribution without offset",
        os.path.join(save_dir, "test_components_dev.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["P_future", "T_future", "S_dev"],
        "Test Internal P/T Before Shape Fusion and S_dev",
        "Internal target-scale value without offset",
        os.path.join(save_dir, "test_shape_internal_PT.png"),
        plot_n,
    )
    _plot_lines(
        pred_df,
        ["Rw_raw_real", "Rw_real", "Ru_raw_real", "Ru_real"],
        "Test Raw vs Gated Residuals",
        "Net load contribution",
        os.path.join(save_dir, "test_raw_vs_gated_residuals.png"),
        plot_n,
    )

    share_df = pd.DataFrame(test_component_stats)
    share_df["component_label"] = share_df["component"].str.replace("_dev", "", regex=False)
    plt.figure(figsize=(7, 4))
    plt.bar(share_df["component_label"], share_df["share_abs"])
    plt.title("Test Mean Absolute Component Share")
    plt.xlabel("Component")
    plt.ylabel("share_abs")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_component_share_bar.png"), dpi=200)
    plt.close()

    y_true = pred_df["y_true"].values.astype(np.float64)
    y_pred = pred_df["y_pred"].values.astype(np.float64)
    true_amp = np.abs(np.fft.rfft(y_true))
    pred_amp = np.abs(np.fft.rfft(y_pred))
    freq_idx = np.arange(len(true_amp))
    plt.figure(figsize=(12, 5))
    plt.plot(freq_idx, true_amp, label="y_true FFT amplitude")
    plt.plot(freq_idx, pred_amp, label="y_pred FFT amplitude")
    plt.title("Test FFT Amplitude Spectrum: True vs Predicted")
    plt.xlabel("Frequency bin")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_fft_true_pred.png"), dpi=200)
    plt.close()


def save_checkpoint(path: str, model: nn.Module, cfg: WeatherAwareAEShiftExperimentConfig, metadata: Dict):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "experiment_config": asdict(cfg),
            "metadata": metadata,
        },
        path,
    )


def load_model_state(path: str, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    return checkpoint, {"model_state_dict": checkpoint}


def train_one_client(client_id: int, csv_path: str, args, base_cfg: WeatherAwareAEShiftExperimentConfig):
    cfg = copy.deepcopy(base_cfg)
    client_name = f"client_{client_id}"
    client_dir = os.path.abspath(os.path.join(args.output_root, client_name))
    ensure_dir(client_dir)

    device = normalize_device(args.device)
    data = prepare_client_data(csv_path, cfg, args.batch_size, device)
    y_scaler = data["scalers"]["y_scaler"]

    model = WeatherAwareAEShiftNetLoadModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config_payload = {
        "client_id": int(client_id),
        "client_name": client_name,
        "csv_path": csv_path,
        "csv_path_abs": os.path.abspath(csv_path),
        "output_dir": client_dir,
        "target": "net_load if present else gc - gg",
        "input_policy": {
            "history": "past net_load, time scalar, weather, and time encodings",
            "shift_exogenous": "weather_shift and time_enc_shift may include t+1 weather/time; no future net_load/gc/gg inputs",
            "weather_branch_policy": "Rw branch uses base_resid_hist and weather only; time encodings are not used by the weather residual branch",
            "future_weather_assumption": "perfect known future weather or available weather forecast",
            "model_output": "Shape + FreeResidual + EventExpertCorrection",
            "no_user_residual": True,
            "daylight_gate_on_residual": False,
            "event_expert": bool(cfg.use_event_expert),
            "weighted_ste": False,
            "event_weighted_next_loss": True,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "patience": int(args.patience),
            "seq_len": int(cfg.seq_len),
            "horizon": int(cfg.horizon),
            "seed": int(args.seed),
            "device": str(device),
            "optimizer": "Adam",
            "early_stop": "validation RMSE of y_shift[:, -1, :] in original net_load scale",
            "grad_clip": float(args.grad_clip),
            "loss_formula": (
                "lambda_ste * (alpha_ste * Fourier + (1 - alpha_ste) * MSE_seq) "
                "+ lambda_next * EventWeightedMSE_last_step "
                "+ lambda_event_corr * EventResidualCorrection "
                "+ lambda_event_gate * BCE(event_gate, event_mask) "
                "+ lambda_event_normal * NormalSuppression"
            ),
            "lambda_next": float(cfg.lambda_next),
            "lambda_ste": float(cfg.lambda_ste),
            "alpha_ste": float(cfg.alpha_ste),
            "lambda_trend_smooth": float(cfg.lambda_trend_smooth),
            "lambda_ae_y": float(cfg.lambda_ae_y),
            "lambda_ae_resid": float(cfg.lambda_ae_resid),
            "lambda_ru_mean": float(cfg.lambda_ru_mean),
            "lambda_rw_ae": float(cfg.lambda_rw_ae),
            "lambda_event_corr": float(cfg.lambda_event_corr),
            "lambda_event_gate": float(cfg.lambda_event_gate),
            "lambda_event_normal": float(cfg.lambda_event_normal),
            "event_weight_beta": float(cfg.event_weight_beta),
            "user_resid_scale": float(cfg.user_resid_scale),
            "user_day_scale": float(cfg.user_day_scale),
            "detach_user_resid_input": bool(cfg.detach_user_resid_input),
            "detach_event_base_error": bool(cfg.detach_event_base_error),
            "component_eval_every": int(args.component_eval_every),
            "component_sample_n": int(args.component_sample_n),
            "plot_n": int(args.plot_n),
        },
        "experiment_config": asdict(cfg),
        "event_thresholds": data["event_thresholds"],
        "split_info": data["split_info"],
        "scalers": {name: scaler_to_dict(scaler) for name, scaler in data["scalers"].items()},
        "model_parameter_count": int(count_parameters(model)),
    }
    with open(os.path.join(client_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"Weather-aware AND-style AE-shift training | Free Residual + Event Expert Correction | {client_name}")
    print(f"CSV: {csv_path}")
    print(
        "Train/Val/Test rows: "
        f"{data['split_info']['train_rows']}/{data['split_info']['val_rows']}/{data['split_info']['test_rows']}"
    )
    print(
        "Train/Val/Test samples: "
        f"{data['train_samples']}/{data['val_samples']}/{data['test_samples']}"
    )
    print(f"Temperature source: {data['split_info']['temp_source']}")
    print(
        "Loss weights: "
        f"lambda_next={cfg.lambda_next}, lambda_ste={cfg.lambda_ste}, alpha_ste={cfg.alpha_ste}, "
        f"lambda_trend_smooth={cfg.lambda_trend_smooth}, "
        f"lambda_ae_y={cfg.lambda_ae_y}, lambda_ae_resid={cfg.lambda_ae_resid}, "
        f"lambda_ru_mean={cfg.lambda_ru_mean}, lambda_rw_ae={cfg.lambda_rw_ae}, "
        f"lambda_event_corr={cfg.lambda_event_corr}, lambda_event_gate={cfg.lambda_event_gate}, "
        f"lambda_event_normal={cfg.lambda_event_normal}, event_weight_beta={cfg.event_weight_beta}, "
        f"use_event_expert={cfg.use_event_expert}, num_basis={cfg.num_basis}"
    )
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Device: {device}")
    print("=" * 100)

    best_val_rmse = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    rows = []
    component_summary_rows = []
    component_sample_rows = []
    best_model_path = os.path.join(client_dir, "best_model.pth")
    final_model_path = os.path.join(client_dir, "final_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_stats, _ = run_epoch(
            model=model,
            loader=data["train_loader"],
            device=device,
            cfg=cfg,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            collect_outputs=False,
        )
        val_stats, val_collected = run_epoch(
            model=model,
            loader=data["val_loader"],
            device=device,
            cfg=cfg,
            optimizer=None,
            grad_clip=0.0,
            collect_outputs=True,
        )
        val_true = inverse_target(y_scaler, val_collected["y_true"])
        val_pred = inverse_target(y_scaler, val_collected["y_pred"])
        val_metrics = calc_metrics(val_true, val_pred)
        val_component_stats = compute_component_stats(val_collected, y_scaler)
        val_share_metrics = component_share_dict(val_component_stats, prefix="val")

        improved = val_metrics["RMSE"] < best_val_rmse
        if improved:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(
                best_model_path,
                model,
                cfg,
                {"epoch": int(epoch), "val_RMSE": float(best_val_rmse), "config": config_payload},
            )
        else:
            no_improve_epochs += 1

        row = {
            "epoch": int(epoch),
            "train_loss": train_stats["loss"],
            "train_loss_next": train_stats["loss_next"],
            "train_weighted_loss_next": train_stats["weighted_loss_next"],
            "train_loss_ste": train_stats["loss_ste"],
            "train_loss_seq": train_stats["loss_seq"],
            "train_loss_weighted_seq": train_stats["loss_weighted_seq"],
            "train_loss_fourier": train_stats["loss_fourier"],
            "train_weighted_loss_ste": train_stats["weighted_loss_ste"],
            "train_loss_trend_smooth": train_stats["loss_trend_smooth"],
            "train_weighted_loss_trend_smooth": train_stats["weighted_loss_trend_smooth"],
            "train_loss_ae_y": train_stats["loss_ae_y"],
            "train_weighted_loss_ae_y": train_stats["weighted_loss_ae_y"],
            "train_loss_ae_resid": train_stats["loss_ae_resid"],
            "train_weighted_loss_ae_resid": train_stats["weighted_loss_ae_resid"],
            "train_loss_ru_mean": train_stats["loss_ru_mean"],
            "train_weighted_loss_ru_mean": train_stats["weighted_loss_ru_mean"],
            "train_loss_rw_ae": train_stats["loss_rw_ae"],
            "train_weighted_loss_rw_ae": train_stats["weighted_loss_rw_ae"],
            "train_loss_event_corr": train_stats["loss_event_corr"],
            "train_weighted_loss_event_corr": train_stats["weighted_loss_event_corr"],
            "train_loss_event_gate": train_stats["loss_event_gate"],
            "train_weighted_loss_event_gate": train_stats["weighted_loss_event_gate"],
            "train_loss_event_normal": train_stats["loss_event_normal"],
            "train_weighted_loss_event_normal": train_stats["weighted_loss_event_normal"],
            "train_event_mask_mean": train_stats["event_mask_mean"],
            "train_event_gate_mean": train_stats["event_gate_mean"],
            "train_event_corr_abs_mean": train_stats["event_corr_abs_mean"],
            "train_event_corr_abs_event_mean": train_stats["event_corr_abs_event_mean"],
            "train_event_corr_abs_normal_mean": train_stats["event_corr_abs_normal_mean"],
            "val_loss": val_stats["loss"],
            "val_loss_next": val_stats["loss_next"],
            "val_weighted_loss_next": val_stats["weighted_loss_next"],
            "val_loss_ste": val_stats["loss_ste"],
            "val_loss_seq": val_stats["loss_seq"],
            "val_loss_weighted_seq": val_stats["loss_weighted_seq"],
            "val_loss_fourier": val_stats["loss_fourier"],
            "val_weighted_loss_ste": val_stats["weighted_loss_ste"],
            "val_loss_trend_smooth": val_stats["loss_trend_smooth"],
            "val_weighted_loss_trend_smooth": val_stats["weighted_loss_trend_smooth"],
            "val_loss_ae_y": val_stats["loss_ae_y"],
            "val_weighted_loss_ae_y": val_stats["weighted_loss_ae_y"],
            "val_loss_ae_resid": val_stats["loss_ae_resid"],
            "val_weighted_loss_ae_resid": val_stats["weighted_loss_ae_resid"],
            "val_loss_ru_mean": val_stats["loss_ru_mean"],
            "val_weighted_loss_ru_mean": val_stats["weighted_loss_ru_mean"],
            "val_loss_rw_ae": val_stats["loss_rw_ae"],
            "val_weighted_loss_rw_ae": val_stats["weighted_loss_rw_ae"],
            "val_loss_event_corr": val_stats["loss_event_corr"],
            "val_weighted_loss_event_corr": val_stats["weighted_loss_event_corr"],
            "val_loss_event_gate": val_stats["loss_event_gate"],
            "val_weighted_loss_event_gate": val_stats["weighted_loss_event_gate"],
            "val_loss_event_normal": val_stats["loss_event_normal"],
            "val_weighted_loss_event_normal": val_stats["weighted_loss_event_normal"],
            "val_event_mask_mean": val_stats["event_mask_mean"],
            "val_event_gate_mean": val_stats["event_gate_mean"],
            "val_event_corr_abs_mean": val_stats["event_corr_abs_mean"],
            "val_event_corr_abs_event_mean": val_stats["event_corr_abs_event_mean"],
            "val_event_corr_abs_normal_mean": val_stats["event_corr_abs_normal_mean"],
            "val_MAE": val_metrics["MAE"],
            "val_MSE": val_metrics["MSE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAPE_percent": val_metrics["MAPE_percent"],
            "val_R2": val_metrics["R2"],
            "best_val_RMSE": best_val_rmse,
        }
        row.update(val_share_metrics)
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(client_dir, "training_log.csv"), index=False, encoding="utf-8-sig")

        if args.component_eval_every > 0 and epoch % args.component_eval_every == 0:
            split_collected = {"val": val_collected}
            _, split_collected["train"] = run_epoch(
                model=model,
                loader=data["train_loader"],
                device=device,
                cfg=cfg,
                optimizer=None,
                grad_clip=0.0,
                collect_outputs=True,
            )
            _, split_collected["test"] = run_epoch(
                model=model,
                loader=data["test_loader"],
                device=device,
                cfg=cfg,
                optimizer=None,
                grad_clip=0.0,
                collect_outputs=True,
            )
            for split_name, collected in split_collected.items():
                for stats_row in compute_component_stats(collected, y_scaler):
                    component_summary_rows.append({"epoch": int(epoch), "split": split_name, **stats_row})
            if args.component_sample_n > 0:
                component_sample_rows.extend(
                    build_component_sample_dataframe(
                        val_collected,
                        y_scaler,
                        epoch=epoch,
                        split="val",
                        sample_n=args.component_sample_n,
                    ).to_dict("records")
                )
            pd.DataFrame(component_summary_rows).to_csv(
                os.path.join(client_dir, "epoch_component_summary.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(component_sample_rows).to_csv(
                os.path.join(client_dir, "epoch_component_samples.csv"),
                index=False,
                encoding="utf-8-sig",
            )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"TrainLoss={train_stats['loss']:.6f} | "
            f"Next={train_stats['loss_next']:.6f} | "
            f"STEW={train_stats['weighted_loss_ste']:.6f} | "
            f"TrendSmoothW={train_stats['weighted_loss_trend_smooth']:.6f} | "
            f"RwAEW={train_stats['weighted_loss_rw_ae']:.6f} | "
            f"EventGate={train_stats['event_gate_mean']:.4f} | "
            f"EventMask={train_stats['event_mask_mean']:.4f} | "
            f"EventCorrAbs={train_stats['event_corr_abs_mean']:.4f} | "
            f"ValLoss={val_stats['loss']:.6f} | "
            f"ValRMSE={val_metrics['RMSE']:.6f} | "
            f"BestRMSE={best_val_rmse:.6f}"
        )

        if args.patience > 0 and no_improve_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if not os.path.exists(best_model_path):
        save_checkpoint(
            best_model_path,
            model,
            cfg,
            {"epoch": int(rows[-1]["epoch"]), "val_RMSE": float(best_val_rmse), "config": config_payload},
        )
    save_checkpoint(
        final_model_path,
        model,
        cfg,
        {"epoch": int(rows[-1]["epoch"]), "best_epoch": int(best_epoch), "config": config_payload},
    )

    log_df = pd.DataFrame(rows)
    plot_round_curve(
        log_df["train_loss"].values,
        title=f"{client_name} Train Loss",
        xlabel="Epoch",
        ylabel="Scaled Loss",
        save_path=os.path.join(client_dir, "train_loss_curve.png"),
    )
    plot_round_curve(
        log_df["val_RMSE"].values,
        title=f"{client_name} Validation RMSE",
        xlabel="Epoch",
        ylabel="RMSE",
        save_path=os.path.join(client_dir, "val_rmse_curve.png"),
    )

    best_state, _ = load_model_state(best_model_path, device)
    model.load_state_dict(best_state)
    test_stats, test_collected = run_epoch(
        model=model,
        loader=data["test_loader"],
        device=device,
        cfg=cfg,
        optimizer=None,
        grad_clip=0.0,
        collect_outputs=True,
    )
    pred_df = build_prediction_dataframe(test_collected, y_scaler)
    metrics = calc_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)
    base_metrics = calc_metrics(pred_df["y_true"].values, pred_df["y_base_pred"].values)
    event_metrics = compute_event_metrics(pred_df)
    metrics_row = {
        "client_id": int(client_id),
        "N": int(len(pred_df)),
        "seq_len": int(cfg.seq_len),
        "horizon": int(cfg.horizon),
        "test_loss": float(test_stats["loss"]),
        "MAE_base": float(base_metrics["MAE"]),
        "RMSE_base": float(base_metrics["RMSE"]),
        "R2_base": float(base_metrics["R2"]),
        "MAE_ramp_base": float(event_metrics.get("MAE_ramp_base", float("nan"))),
        "MAE_ramp_final": float(event_metrics.get("MAE_ramp_final", float("nan"))),
        "MAE_night_ramp_base": float(event_metrics.get("MAE_night_ramp_base", float("nan"))),
        "MAE_night_ramp_final": float(event_metrics.get("MAE_night_ramp_final", float("nan"))),
        "mean_gate_ramp": float(event_metrics.get("mean_gate_ramp", float("nan"))),
        "mean_gate_normal": float(event_metrics.get("mean_gate_normal", float("nan"))),
    }
    metrics_row.update(metrics)
    test_component_stats = compute_component_stats(test_collected, y_scaler)
    metrics_row.update(component_share_dict(test_component_stats, prefix="test"))

    pred_df[
        [
            "timestamp",
            "y_true",
            "y_base_pred",
            "y_pred",
            "event_gate",
            "event_correction_real",
            "y_true_scaled",
            "y_base_pred_scaled",
            "y_pred_scaled",
        ]
    ].to_csv(
        os.path.join(client_dir, "test_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pred_df.to_csv(os.path.join(client_dir, "component_predictions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(test_component_stats).to_csv(
        os.path.join(client_dir, "test_component_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    build_shift_sequence_sample_dataframe(test_collected, y_scaler, sample_n=5).to_csv(
        os.path.join(client_dir, "test_shift_sequence_samples.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([event_metrics]).to_csv(
        os.path.join(client_dir, "test_event_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_metrics_csv(metrics_row, os.path.join(client_dir, "test_metrics.csv"))
    plot_true_pred(
        pred_df["y_true"].values,
        pred_df["y_pred"].values,
        save_path=os.path.join(client_dir, "test_prediction.png"),
        title=f"{client_name} Weather-Aware AE-Shift Net Load Prediction",
        show_n=args.plot_n,
    )
    save_test_diagnostic_plots(pred_df, test_component_stats, save_dir=client_dir, plot_n=args.plot_n)
    print_metrics(metrics, title=f"{client_name} Test Metrics")
    print(f"Output directory: {client_dir}")

    summary = {
        "client_id": int(client_id),
        "client_name": client_name,
        "client_dir": client_dir,
        "best_epoch": int(best_epoch),
        "best_val_RMSE": float(best_val_rmse),
        "test_loss": float(test_stats["loss"]),
    }
    summary.update(metrics)
    return summary


def select_clients(args) -> List[Tuple[int, str]]:
    client_files = list(CFG.data.client_files)
    if args.client_id is not None:
        if args.client_id < 1 or args.client_id > len(client_files):
            raise ValueError(f"--client-id must be in [1, {len(client_files)}], got {args.client_id}.")
        return [(int(args.client_id), client_files[args.client_id - 1])]
    return [(infer_client_id_from_path(path, idx), path) for idx, path in enumerate(client_files, start=1)]


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_root)

    exp_cfg = WeatherAwareAEShiftExperimentConfig(
        seq_len=int(args.seq_len),
        horizon=1,
        train_ratio=float(CFG.data.train_ratio),
        val_ratio=float(CFG.data.val_ratio),
        ghi_threshold=float(args.ghi_threshold),
        smooth_k=int(args.smooth_k),
        lambda_next=float(args.lambda_next),
        lambda_ste=float(args.lambda_ste),
        alpha_ste=float(args.alpha_ste),
        lambda_trend_smooth=float(args.lambda_trend_smooth),
        lambda_ae_y=float(args.lambda_ae_y),
        lambda_ae_resid=float(args.lambda_ae_resid),
        lambda_ru_mean=float(args.lambda_ru_mean),
        lambda_rw_ae=float(args.lambda_rw_ae),
        user_resid_scale=float(args.user_resid_scale),
        user_day_scale=float(args.user_day_scale),
        detach_user_resid_input=bool(args.detach_user_resid_input),
        num_basis=int(args.num_basis),
        dropout=float(args.dropout),
        use_event_expert=bool(args.use_event_expert),
        event_expert_hidden=int(args.event_expert_hidden),
        event_gate_hidden=int(args.event_gate_hidden),
        lambda_event_corr=float(args.lambda_event_corr),
        lambda_event_gate=float(args.lambda_event_gate),
        lambda_event_normal=float(args.lambda_event_normal),
        event_weight_beta=float(args.event_weight_beta),
        event_ramp_quantile=float(args.event_ramp_quantile),
        event_ghi_quantile=float(args.event_ghi_quantile),
        event_peak_quantile=float(args.event_peak_quantile),
        event_tau_ramp=float(args.event_tau_ramp),
        event_tau_ghi=float(args.event_tau_ghi),
        event_tau_zero=float(args.event_tau_zero),
        event_tau_peak=float(args.event_tau_peak),
        event_zero_eps=float(args.event_zero_eps),
        detach_event_base_error=bool(args.detach_event_base_error),
    )

    print(
        "Weather-aware AND-style AE-shift baseline training | Shape + Weather Residual\n"
        "Loss/config: "
        f"seq_len={exp_cfg.seq_len}, horizon={exp_cfg.horizon}, "
        f"lambda_next={exp_cfg.lambda_next}, "
        f"lambda_ste={exp_cfg.lambda_ste}, alpha_ste={exp_cfg.alpha_ste}, "
        f"lambda_trend_smooth={exp_cfg.lambda_trend_smooth}, "
        f"lambda_ae_y={exp_cfg.lambda_ae_y}, lambda_ae_resid={exp_cfg.lambda_ae_resid}, "
        f"lambda_ru_mean={exp_cfg.lambda_ru_mean}, lambda_rw_ae={exp_cfg.lambda_rw_ae}, "
        f"num_basis={exp_cfg.num_basis}, weighted_ste=False, no_user_residual=True"
    )

    summaries = []
    for client_id, csv_path in select_clients(args):
        summaries.append(train_one_client(client_id, csv_path, args, exp_cfg))

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.output_root, "all_clients_test_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("=" * 100)
    print(f"Finished {len(summaries)} client(s). Summary: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
