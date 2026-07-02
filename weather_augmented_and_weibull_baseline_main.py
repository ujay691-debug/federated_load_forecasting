"""Weather-augmented AND-Weibull paper baseline: Shape + Residual.

The data pipeline, chronological split, training loop, and artifact writers are
reused from ``weather_aware_additive_ae_shift_wste_weibull_user_main.py``.
Only the baseline-specific model, STE-only objective, diagnostics, and metadata
are defined here.  No user-residual branch or daylight residual gate is used.

Example:
    python weather_augmented_and_weibull_baseline_main.py --client-id 2 \
        --epochs 60 --lambda-ste 1.0 --alpha-ste 0.1 --num-basis 48
"""

import argparse
import builtins
import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from utils.data_utils import ensure_dir, set_seed

import weather_aware_additive_ae_shift_wste_weibull_user_main as base


EXPERIMENT_TITLE = "Weather-augmented AND-Weibull baseline | Shape + Residual"
BASE_BUILD_PREDICTION_DATAFRAME = base.build_prediction_dataframe


@dataclass
class WeatherAwareAEShiftExperimentConfig:
    """Configuration for the strict two-component paper baseline."""

    seq_len: int = 48
    horizon: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    ghi_threshold: float = 10.0
    smooth_k: int = 3
    lambda_ste: float = 1.0
    alpha_ste: float = 0.1
    num_basis: int = 48
    ae_y_hidden: int = 24
    ae_y_bottleneck: int = 12
    resid_weibull_lstm_hidden: int = 32
    resid_weibull_fc_hidden: int = 16
    dropout: float = 0.0

    # Compatibility-only fields expected by the shared logger.  They are
    # inactive and never participate in this model's forward pass or loss.
    lambda_next: float = 0.0
    lambda_trend_smooth: float = 0.0
    lambda_ae_y: float = 0.0
    lambda_ae_resid: float = 0.0
    lambda_ru_mean: float = 0.0
    lambda_rw_ae: float = 0.0
    user_resid_scale: float = 0.0
    user_day_scale: float = 1.0
    detach_user_resid_input: bool = True


class PeriodicSineSequenceBranch(nn.Module):
    """Learnable paper-style sine basis ``sin(W0 o^T + Phi) a``."""

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
            self.omega.copy_(2.0 * math.pi * k / self.period)
            self.phi.copy_(math.pi / 2.0 + torch.remainder(k, 2.0) * math.pi / 2.0)
            self.amplitude.fill_(1.0)
            self.bias.zero_()

    def forward(self, time_shift_scalar: torch.Tensor) -> torch.Tensor:
        time_index = time_shift_scalar * self.period
        basis = torch.sin(
            time_index * self.omega.view(1, 1, -1)
            + self.phi.view(1, 1, -1)
        )
        return torch.matmul(basis, self.amplitude.view(-1, 1)) + self.bias


class LinearTrendBranch(nn.Module):
    """Paper-style linear sequence trend: H -> H."""

    def __init__(self, seq_len: int = 48):
        super().__init__()
        self.linear = nn.Linear(int(seq_len), int(seq_len))

    def forward(self, y_hist: torch.Tensor) -> torch.Tensor:
        return self.linear(y_hist.squeeze(-1)).unsqueeze(-1)


class SequenceAutoEncoder(nn.Module):
    """Flat history autoencoder used to construct the residual input."""

    def __init__(self, seq_len: int, hidden: int, bottleneck: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(seq_len), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(bottleneck)),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(bottleneck), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(seq_len)),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(seq.squeeze(-1))).unsqueeze(-1)


class WeibullAttention(nn.Module):
    """Learnable normalized Weibull attention over sequence positions."""

    def __init__(self):
        super().__init__()
        self.raw_kappa = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.raw_lambda = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, seq_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        horizon = int(seq_hidden.size(1))
        positions = torch.arange(
            1, horizon + 1, dtype=seq_hidden.dtype, device=seq_hidden.device
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


class WeatherAugmentedWeibullResidualSeqBranch(nn.Module):
    """Single residual branch: AE residual + GHI/temperature -> Weibull Attention LSTM."""

    def __init__(
        self,
        hist_input_dim: int = 3,
        shift_exog_dim: int = 2,
        lstm_hidden: int = 32,
        fc_hidden: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=int(hist_input_dim),
            hidden_size=int(lstm_hidden),
            batch_first=True,
        )
        self.weibull_attention = WeibullAttention()
        self.residual_seq_head = nn.Sequential(
            nn.Linear(2 * int(lstm_hidden) + int(shift_exog_dim), int(fc_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(fc_hidden), 1),
        )

    def forward(
        self,
        base_resid_hist: torch.Tensor,
        weather_hist: torch.Tensor,
        weather_shift: torch.Tensor,
    ) -> torch.Tensor:
        encoder_input = torch.cat([base_resid_hist, weather_hist], dim=-1)
        seq_hidden, _ = self.lstm(encoder_input)
        context_seq, _ = self.weibull_attention(seq_hidden)
        decoder_input = torch.cat([seq_hidden, context_seq, weather_shift], dim=-1)
        return self.residual_seq_head(decoder_input)


class WeatherAwareAEShiftNetLoadModel(nn.Module):
    """Strict baseline model: ``y_shift_pred = S_shift + R_shift``."""

    def __init__(self, cfg: WeatherAwareAEShiftExperimentConfig):
        super().__init__()
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
        self.residual_branch = WeatherAugmentedWeibullResidualSeqBranch(
            hist_input_dim=1 + 2,
            shift_exog_dim=2,
            lstm_hidden=cfg.resid_weibull_lstm_hidden,
            fc_hidden=cfg.resid_weibull_fc_hidden,
            dropout=cfg.dropout,
        )
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
        del time_hist_scalar, time_enc_hist, time_enc_shift

        p_shift = self.periodic_branch(time_shift_scalar)
        t_shift = self.trend_branch(y_hist)
        s_shift = self.shape_fusion(torch.cat([p_shift, t_shift], dim=-1))

        y_ae_recon_hist = self.y_autoencoder(y_hist)
        base_resid_hist = y_hist - y_ae_recon_hist
        # Weather feature order is [GHI, temperature, wind, daylight, GHI ramp].
        # This baseline intentionally exposes only GHI and temperature to the
        # residual predictor, for both historical and shifted exogenous inputs.
        residual_weather_hist = weather_hist[..., 0:2]
        residual_weather_shift = weather_shift[..., 0:2]
        r_raw_shift = self.residual_branch(
            base_resid_hist, residual_weather_hist, residual_weather_shift
        )
        r_shift = r_raw_shift  # Deliberately no daylight gate.

        y_shift_pred = s_shift + r_shift
        y_pred_future = y_shift_pred[:, -1, :]

        daylight_hist = weather_hist[..., 3:4]
        daylight_shift = weather_shift[..., 3:4]
        zeros_r = torch.zeros_like(r_shift)
        zeros_hist = torch.zeros_like(base_resid_hist)
        ones_gate = torch.ones_like(daylight_shift)
        rw_hist_proxy = torch.cat(
            [torch.zeros_like(r_shift[:, :1, :]), r_shift[:, :-1, :]], dim=1
        )

        return {
            "y_shift_pred": y_shift_pred,
            "y_pred_future": y_pred_future,
            "S_shift": s_shift,
            "P_shift": p_shift,
            "T_shift": t_shift,
            "R_raw_shift": r_raw_shift,
            "R_shift": r_shift,
            # Rw is a compatibility alias for the sole residual component.
            "Rw_raw_shift": r_raw_shift,
            "Rw_shift": r_shift,
            "Ru_raw_shift": zeros_r,
            "Ru_shift": zeros_r,
            "daylight_hist": daylight_hist,
            "daylight_shift": daylight_shift,
            "user_day_gate": ones_gate,
            "y_ae_recon_hist": y_ae_recon_hist,
            "base_resid_hist": base_resid_hist,
            "Rw_hist_proxy": rw_hist_proxy,
            "rw_ae_recon_hist": torch.zeros_like(rw_hist_proxy),
            "weather_unexplained_hist": torch.zeros_like(rw_hist_proxy),
            "user_resid_hist": zeros_hist,
            "resid_ae_recon_hist": zeros_hist,
            "resid_ae_error_hist": zeros_hist,
        }


def compute_fourier_loss(
    pred_seq: torch.Tensor, true_seq: torch.Tensor
) -> torch.Tensor:
    if pred_seq.dim() == 3 and pred_seq.size(-1) == 1:
        pred_seq = pred_seq.squeeze(-1)
    if true_seq.dim() == 3 and true_seq.size(-1) == 1:
        true_seq = true_seq.squeeze(-1)
    return F.mse_loss(
        torch.abs(torch.fft.rfft(pred_seq, dim=1)),
        torch.abs(torch.fft.rfft(true_seq, dim=1)),
    )


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: WeatherAwareAEShiftExperimentConfig,
) -> Dict[str, torch.Tensor]:
    """STE only: no next-step, weighting, smoothness, AE, or residual losses."""

    y_shift_pred = outputs["y_shift_pred"]
    y_shift = batch["y_shift"]
    loss_seq = F.mse_loss(y_shift_pred, y_shift)
    loss_fourier = compute_fourier_loss(y_shift_pred, y_shift)
    loss_ste = cfg.alpha_ste * loss_fourier + (1.0 - cfg.alpha_ste) * loss_seq
    weighted_loss_ste = cfg.lambda_ste * loss_ste
    zero = y_shift_pred.new_zeros(())
    return {
        "loss": weighted_loss_ste,
        "loss_next": zero,
        "weighted_loss_next": zero,
        "loss_ste": loss_ste,
        "loss_seq": loss_seq,
        "loss_weighted_seq": loss_seq,
        "loss_fourier": loss_fourier,
        "weighted_loss_ste": weighted_loss_ste,
        "loss_trend_smooth": zero,
        "weighted_loss_trend_smooth": zero,
        "loss_ae_y": zero,
        "weighted_loss_ae_y": zero,
        "loss_ae_resid": zero,
        "weighted_loss_ae_resid": zero,
        "loss_ru_mean": zero,
        "weighted_loss_ru_mean": zero,
        "loss_rw_ae": zero,
        "weighted_loss_rw_ae": zero,
    }


def build_prediction_dataframe(collected: Dict[str, np.ndarray], y_scaler) -> pd.DataFrame:
    """Keep legacy columns and add explicit R columns for this baseline."""

    df = BASE_BUILD_PREDICTION_DATAFRAME(collected, y_scaler)
    df["R_real"] = df["Rw_real"]
    df["R_dev"] = df["Rw_dev"]
    df["R_raw_real"] = df["Rw_raw_real"]
    df["component_additive_sum"] = df["S_real"] + df["R_real"]
    df["component_additive_error"] = df["component_additive_sum"] - df["y_pred"]
    return df


def save_test_diagnostic_plots(
    pred_df: pd.DataFrame,
    test_component_stats: List[Dict[str, float]],
    save_dir: str,
    plot_n: int,
):
    """Write the paper-baseline component and spectral diagnostics."""

    del test_component_stats
    base._plot_lines(
        pred_df,
        ["y_true", "y_pred"],
        "Test True vs Predicted Net Load",
        "Net load",
        os.path.join(save_dir, "test_true_pred.png"),
        plot_n,
    )
    base._plot_lines(
        pred_df,
        ["S_real", "R_real", "y_pred"],
        "Test Additive Components: S_real + R_real",
        "Net load",
        os.path.join(save_dir, "test_components_additive.png"),
        plot_n,
    )
    base._plot_lines(
        pred_df,
        ["S_dev", "R_dev"],
        "Test Dynamic Component Contributions: S_dev and R_dev",
        "Net load contribution without offset",
        os.path.join(save_dir, "test_components_dev.png"),
        plot_n,
    )
    base._plot_lines(
        pred_df,
        ["P_future", "T_future", "S_dev"],
        "Test Internal P/T Before Shape Fusion and S_dev",
        "Internal target-scale value without offset",
        os.path.join(save_dir, "test_shape_internal_PT.png"),
        plot_n,
    )

    y_true = pred_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.float64)
    true_amp = np.abs(np.fft.rfft(y_true))
    pred_amp = np.abs(np.fft.rfft(y_pred))
    freq_idx = np.arange(len(true_amp))
    base.plt.figure(figsize=(12, 5))
    base.plt.plot(freq_idx, true_amp, label="y_true FFT amplitude")
    base.plt.plot(freq_idx, pred_amp, label="y_pred FFT amplitude")
    base.plt.title("Test FFT Amplitude Spectrum: True vs Predicted")
    base.plt.xlabel("Frequency bin")
    base.plt.ylabel("Amplitude")
    base.plt.legend()
    base.plt.tight_layout()
    base.plt.savefig(os.path.join(save_dir, "test_fft_true_pred.png"), dpi=200)
    base.plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description=EXPERIMENT_TITLE)
    parser.add_argument("--client-id", type=int, default=2)
    parser.add_argument(
        "--output-root",
        default=os.path.join("runs", "weather_augmented_and_weibull_baseline"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.train.lr)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=CFG.train.random_seed)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--ghi-threshold", type=float, default=10.0)
    parser.add_argument("--smooth-k", type=int, default=3)
    parser.add_argument("--lambda-ste", type=float, default=1.0)
    parser.add_argument("--alpha-ste", type=float, default=0.1)
    parser.add_argument("--num-basis", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--component-eval-every", type=int, default=1)
    parser.add_argument("--component-sample-n", type=int, default=300)
    parser.add_argument("--plot-n", type=int, default=500)
    parser.add_argument("--device", default=CFG.train.device)

    # Accepted only so older launch scripts remain callable.
    parser.add_argument("--lambda-next", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-trend-smooth", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-ae-y", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-ae-resid", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-ru-mean", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-rw-ae", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--user-resid-scale", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--user-day-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--detach-user-resid-input", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-detach-user-resid-input", action="store_false", dest="detach_user_resid_input", help=argparse.SUPPRESS)
    return parser.parse_args()


def _baseline_print(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], str):
        text = args[0]
        prefix = "Weather-aware additive AE-shift training | "
        if text.startswith(prefix):
            return builtins.print(
                f"{EXPERIMENT_TITLE} | {text[len(prefix):]}", **kwargs
            )
    return builtins.print(*args, **kwargs)


def install_shared_pipeline_hooks():
    """Bind the shared trainer to this baseline's strict implementations."""

    base.WeatherAwareAEShiftExperimentConfig = WeatherAwareAEShiftExperimentConfig
    base.WeatherAwareAEShiftNetLoadModel = WeatherAwareAEShiftNetLoadModel
    base.compute_losses = compute_losses
    base.build_prediction_dataframe = build_prediction_dataframe
    base.save_test_diagnostic_plots = save_test_diagnostic_plots
    base.print = _baseline_print


def _baseline_config_payload(payload: Dict, cfg: WeatherAwareAEShiftExperimentConfig) -> Dict:
    payload = copy.deepcopy(payload)
    payload["experiment_title"] = EXPERIMENT_TITLE
    payload["input_policy"] = {
        "model_output": "Shape + Residual",
        "history_inputs": "y_hist, time_hist_scalar, weather_hist, time_enc_hist",
        "shift_exogenous_inputs": "time_shift_scalar, weather_shift, time_enc_shift",
        "residual_construction": "y_hist - AE_y(y_hist)",
        "residual_predictor": "Weibull Attention LSTM with GHI and temperature exogenous inputs only",
        "residual_branch": "Weibull Attention LSTM",
        "residual_weather_features": ["ghi_scaled", "temp_scaled"],
        "rw_compatibility_alias": "Rw_shift denotes the single Residual component, not a separate weather residual",
        "weather_augmented_residual": True,
        "future_weather_assumption": "perfect known future weather or available weather forecast",
        "no_future_net_load_gc_gg": True,
        "no_user_residual": True,
        "no_daylight_gate_on_residual": True,
        "weighted_ste": False,
        "no_weighted_ste": True,
        "no_auxiliary_losses": True,
        "loss": "STE only",
    }
    payload["training"].update(
        {
            "loss_formula": "lambda_ste * [alpha_ste * Fourier + (1 - alpha_ste) * MSE_seq]",
            "lambda_next": 0.0,
            "lambda_trend_smooth": 0.0,
            "lambda_ae_y": 0.0,
            "lambda_ae_resid": 0.0,
            "lambda_ru_mean": 0.0,
            "lambda_rw_ae": 0.0,
            "inactive_for_this_baseline": [
                "lambda_next",
                "lambda_trend_smooth",
                "lambda_ae_y",
                "lambda_ae_resid",
                "lambda_ru_mean",
                "lambda_rw_ae",
                "user_resid_scale",
                "user_day_scale",
                "detach_user_resid_input",
            ],
        }
    )
    payload["experiment_config"] = asdict(cfg)
    return payload


def _rewrite_saved_metadata(client_dir: str, cfg: WeatherAwareAEShiftExperimentConfig):
    """Make config.json and checkpoint metadata describe the actual baseline."""

    config_path = os.path.join(client_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload = _baseline_config_payload(payload, cfg)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for filename in ("best_model.pth", "final_model.pth"):
        path = os.path.join(client_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            checkpoint["experiment_config"] = asdict(cfg)
            metadata = checkpoint.setdefault("metadata", {})
            metadata["config"] = payload
            torch.save(checkpoint, path)


def train_one_client(
    client_id: int,
    csv_path: str,
    args,
    base_cfg: WeatherAwareAEShiftExperimentConfig,
):
    summary = base.train_one_client(client_id, csv_path, args, base_cfg)
    _rewrite_saved_metadata(summary["client_dir"], base_cfg)
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_root)
    install_shared_pipeline_hooks()

    exp_cfg = WeatherAwareAEShiftExperimentConfig(
        seq_len=int(args.seq_len),
        horizon=1,
        train_ratio=float(CFG.data.train_ratio),
        val_ratio=float(CFG.data.val_ratio),
        ghi_threshold=float(args.ghi_threshold),
        smooth_k=int(args.smooth_k),
        lambda_ste=float(args.lambda_ste),
        alpha_ste=float(args.alpha_ste),
        num_basis=int(args.num_basis),
        dropout=float(args.dropout),
    )

    print(EXPERIMENT_TITLE)
    print(
        "Loss: STE only | "
        f"lambda_ste={exp_cfg.lambda_ste}, alpha_ste={exp_cfg.alpha_ste}, "
        f"num_basis={exp_cfg.num_basis}"
    )

    summaries = []
    for client_id, csv_path in base.select_clients(args):
        summaries.append(train_one_client(client_id, csv_path, args, exp_cfg))

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.output_root, "all_clients_test_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Finished {len(summaries)} client(s). Summary: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
