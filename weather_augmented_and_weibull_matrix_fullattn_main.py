"""Matrix-fused Weather-augmented AND-Weibull paper baseline.

This variant keeps the existing data/training pipeline and strict two-component
Shape + Residual policy, while adding paper-style W2/W3 and W6/W7 sequence
matrix fusion plus per-output-step Weibull attention.

Example:
    python weather_augmented_and_weibull_matrix_fullattn_main.py \
        --client-id 2 --epochs 60 --lambda-ste 1.0 \
        --alpha-ste 0.1 --num-basis 48
"""

import argparse
import copy
import json
import math
import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from utils.data_utils import ensure_dir, set_seed

import weather_augmented_and_weibull_baseline_main as baseline


EXPERIMENT_TITLE = (
    "Weather-augmented AND-Weibull matrix baseline | Shape + Residual"
)
WeatherAwareAEShiftExperimentConfig = baseline.WeatherAwareAEShiftExperimentConfig
PeriodicSineSequenceBranch = baseline.PeriodicSineSequenceBranch
LinearTrendBranch = baseline.LinearTrendBranch
SequenceAutoEncoder = baseline.SequenceAutoEncoder
compute_fourier_loss = baseline.compute_fourier_loss
compute_losses = baseline.compute_losses
build_prediction_dataframe = baseline.build_prediction_dataframe
BASELINE_CONFIG_PAYLOAD = baseline._baseline_config_payload
BASELINE_SAVE_DIAGNOSTIC_PLOTS = baseline.save_test_diagnostic_plots
ORIGINAL_RUN_EPOCH = baseline.base.run_epoch
_LAST_WEIBULL_ALPHA_MATRIX = None


class MatrixShapeFusion(nn.Module):
    """Paper-style shape fusion: S = W2(P) + W3(T) + b_shape."""

    def __init__(self, seq_len: int = 48):
        super().__init__()
        self.seq_len = int(seq_len)
        self.W2 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.W3 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.seq_len))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            eye = torch.eye(self.seq_len)
            self.W2.weight.copy_(0.5 * eye)
            self.W3.weight.copy_(0.5 * eye)
            self.bias.zero_()

    def forward(
        self, p_shift: torch.Tensor, t_shift: torch.Tensor
    ) -> torch.Tensor:
        p = p_shift.squeeze(-1)
        t = t_shift.squeeze(-1)
        s = self.W2(p) + self.W3(t) + self.bias
        return s.unsqueeze(-1)


class MatrixComponentFusion(nn.Module):
    """Paper-style output fusion: Y = W6(S) + W7(R) + b_out."""

    def __init__(self, seq_len: int = 48):
        super().__init__()
        self.seq_len = int(seq_len)
        self.W6 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.W7 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.seq_len))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            eye = torch.eye(self.seq_len)
            self.W6.weight.copy_(eye)
            self.W7.weight.copy_(eye)
            self.bias.zero_()

    def forward(
        self, s_shift: torch.Tensor, r_shift: torch.Tensor
    ) -> torch.Tensor:
        s = s_shift.squeeze(-1)
        r = r_shift.squeeze(-1)
        y = self.W6(s) + self.W7(r) + self.bias
        return y.unsqueeze(-1)


class FullWeibullAttention(nn.Module):
    """Causal per-output-step Weibull attention over relative lag j - i + 1."""

    def __init__(self, seq_len: int = 48):
        super().__init__()
        self.seq_len = int(seq_len)
        kappa_init = 2.0
        lambda_init = 4.0
        raw_kappa_init = math.log(math.expm1(kappa_init))
        raw_lambda_init = math.log(math.expm1(lambda_init))
        self.raw_kappa = nn.Parameter(
            torch.full((self.seq_len,), raw_kappa_init, dtype=torch.float32)
        )
        self.raw_lambda = nn.Parameter(
            torch.full((self.seq_len,), raw_lambda_init, dtype=torch.float32)
        )

    def forward(
        self, seq_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        horizon = int(seq_hidden.size(1))
        if horizon > self.seq_len:
            raise ValueError(
                f"Input horizon {horizon} exceeds configured seq_len {self.seq_len}."
            )

        raw_kappa = self.raw_kappa[:horizon]
        raw_lambda = self.raw_lambda[:horizon]
        kappa = F.softplus(raw_kappa) + 1e-4
        lambda_ = F.softplus(raw_lambda) + 1e-4
        kappa_j = kappa.view(horizon, 1)
        lambda_j = lambda_.view(horizon, 1)

        # Input index i runs from the oldest to the newest observation.  For
        # output step j, only i <= j is causally available and lag=1 denotes
        # the most recent available hidden state at i=j.
        output_j = torch.arange(horizon, device=seq_hidden.device).view(horizon, 1)
        input_i = torch.arange(horizon, device=seq_hidden.device).view(1, horizon)
        causal_mask = input_i <= output_j
        lag = (output_j - input_i + 1).clamp_min(1).to(seq_hidden.dtype)
        scaled = lag / lambda_j

        # Work in log space to preserve gradients for long lags instead of
        # underflowing exp(-(lag/lambda)^kappa) directly to zero.
        log_alpha = (
            torch.log(kappa_j)
            - torch.log(lambda_j)
            + (kappa_j - 1.0) * torch.log(scaled)
            - torch.pow(scaled, kappa_j)
        )
        log_alpha = log_alpha.masked_fill(~causal_mask, float("-inf"))
        alpha = torch.softmax(log_alpha, dim=1)

        context_seq = torch.einsum("ji,bid->bjd", alpha, seq_hidden)
        return context_seq, alpha


class WeatherAugmentedWeibullResidualSeqBranch(nn.Module):
    """AE residual + GHI/temperature with full Weibull attention."""

    def __init__(
        self,
        seq_len: int,
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
        self.weibull_attention = FullWeibullAttention(seq_len=seq_len)
        self.context_gate = nn.Linear(2 * int(lstm_hidden), int(lstm_hidden))
        self.residual_seq_head = nn.Sequential(
            nn.Linear(int(lstm_hidden) + int(shift_exog_dim), int(fc_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(fc_hidden), 1),
        )
        with torch.no_grad():
            self.context_gate.weight.zero_()
            self.context_gate.bias.zero_()

    def forward(
        self,
        base_resid_hist: torch.Tensor,
        weather_hist: torch.Tensor,
        weather_shift: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_input = torch.cat([base_resid_hist, weather_hist], dim=-1)
        seq_hidden, _ = self.lstm(encoder_input)
        context_seq, alpha_matrix = self.weibull_attention(seq_hidden)
        context_gate = torch.sigmoid(
            self.context_gate(torch.cat([seq_hidden, context_seq], dim=-1))
        )
        fused_hidden = context_gate * context_seq + (1.0 - context_gate) * seq_hidden
        decoder_input = torch.cat([fused_hidden, weather_shift], dim=-1)
        r_shift = self.residual_seq_head(decoder_input)
        return r_shift, alpha_matrix


class WeatherAwareAEShiftNetLoadModel(nn.Module):
    """Matrix-fused strict baseline with Shape and one Residual component."""

    def __init__(self, cfg: WeatherAwareAEShiftExperimentConfig):
        super().__init__()
        self.periodic_branch = PeriodicSineSequenceBranch(
            num_basis=cfg.num_basis,
            period=cfg.seq_len,
        )
        self.trend_branch = LinearTrendBranch(seq_len=cfg.seq_len)
        self.shape_fusion = MatrixShapeFusion(seq_len=cfg.seq_len)
        self.y_autoencoder = SequenceAutoEncoder(
            seq_len=cfg.seq_len,
            hidden=cfg.ae_y_hidden,
            bottleneck=cfg.ae_y_bottleneck,
        )
        self.residual_branch = WeatherAugmentedWeibullResidualSeqBranch(
            seq_len=cfg.seq_len,
            hist_input_dim=1 + 2,
            shift_exog_dim=2,
            lstm_hidden=cfg.resid_weibull_lstm_hidden,
            fc_hidden=cfg.resid_weibull_fc_hidden,
            dropout=cfg.dropout,
        )
        self.final_fusion = MatrixComponentFusion(seq_len=cfg.seq_len)

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
        s_shift = self.shape_fusion(p_shift, t_shift)

        y_ae_recon_hist = self.y_autoencoder(y_hist)
        base_resid_hist = y_hist - y_ae_recon_hist

        # Full weather order: [GHI, temperature, wind, daylight, GHI ramp].
        # Only GHI and temperature are exposed to the residual predictor.
        residual_weather_hist = weather_hist[..., 0:2]
        residual_weather_shift = weather_shift[..., 0:2]
        r_raw_shift, weibull_alpha_matrix = self.residual_branch(
            base_resid_hist,
            residual_weather_hist,
            residual_weather_shift,
        )
        r_shift = r_raw_shift  # Deliberately no daylight gate.

        y_additive_direct = s_shift + r_shift
        y_shift_pred = self.final_fusion(s_shift, r_shift)
        y_pred_future = y_shift_pred[:, -1, :]

        if (
            getattr(self, "_capture_weibull_attention", False)
            and getattr(self, "_captured_weibull_alpha_matrix", None) is None
        ):
            self._captured_weibull_alpha_matrix = (
                weibull_alpha_matrix.detach().cpu().clone()
            )

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
            "Y_additive_direct": y_additive_direct,
            "Y_matrix_fused": y_shift_pred,
            "weibull_alpha_matrix": weibull_alpha_matrix.detach(),
            "S_shift": s_shift,
            "P_shift": p_shift,
            "T_shift": t_shift,
            "R_raw_shift": r_raw_shift,
            "R_shift": r_shift,
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


def run_epoch(*args, **kwargs):
    """Capture the first batch's alpha matrix when outputs are collected."""

    global _LAST_WEIBULL_ALPHA_MATRIX
    model = kwargs.get("model", args[0] if args else None)
    collect_outputs = bool(kwargs.get("collect_outputs", False))
    if collect_outputs and model is not None:
        model._capture_weibull_attention = True
        model._captured_weibull_alpha_matrix = None
    try:
        result = ORIGINAL_RUN_EPOCH(*args, **kwargs)
    finally:
        if model is not None:
            model._capture_weibull_attention = False

    if collect_outputs and model is not None:
        captured = getattr(model, "_captured_weibull_alpha_matrix", None)
        if captured is not None:
            _LAST_WEIBULL_ALPHA_MATRIX = captured.numpy().copy()
    return result


def save_test_diagnostic_plots(
    pred_df: pd.DataFrame,
    test_component_stats,
    save_dir: str,
    plot_n: int,
):
    """Save standard diagnostics plus the full Weibull attention matrix."""

    BASELINE_SAVE_DIAGNOSTIC_PLOTS(
        pred_df,
        test_component_stats,
        save_dir=save_dir,
        plot_n=plot_n,
    )
    if _LAST_WEIBULL_ALPHA_MATRIX is None:
        return

    alpha = np.asarray(_LAST_WEIBULL_ALPHA_MATRIX, dtype=np.float32)
    np.save(os.path.join(save_dir, "test_weibull_attention_matrix.npy"), alpha)
    baseline.base.plt.figure(figsize=(8, 7))
    image = baseline.base.plt.imshow(alpha, aspect="auto", origin="lower", cmap="viridis")
    baseline.base.plt.colorbar(image, label="Attention weight")
    baseline.base.plt.title("Test Full Weibull Attention Matrix alpha[j, i]")
    baseline.base.plt.xlabel("Input hidden state index i")
    baseline.base.plt.ylabel("Output step index j")
    baseline.base.plt.tight_layout()
    baseline.base.plt.savefig(
        os.path.join(save_dir, "test_weibull_attention_heatmap.png"), dpi=200
    )
    baseline.base.plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description=EXPERIMENT_TITLE)
    parser.add_argument("--client-id", type=int, default=2)
    parser.add_argument(
        "--output-root",
        default=os.path.join(
            "runs", "weather_augmented_and_weibull_matrix_fullattn"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.train.lr)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=CFG.train.random_seed)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--ghi-threshold", type=float, default=10.0)
    parser.add_argument("--smooth-k", type=int, default=3)
    parser.add_argument("--lambda-ste", type=float, default=1.0)
    parser.add_argument("--alpha-ste", type=float, default=0.005)
    parser.add_argument("--num-basis", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--component-eval-every", type=int, default=1)
    parser.add_argument("--component-sample-n", type=int, default=300)
    parser.add_argument("--plot-n", type=int, default=500)
    parser.add_argument("--device", default=CFG.train.device)

    # Compatibility-only legacy options. They are accepted but remain inactive.
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


def _baseline_config_payload(
    payload: Dict, cfg: WeatherAwareAEShiftExperimentConfig
) -> Dict:
    payload = BASELINE_CONFIG_PAYLOAD(payload, cfg)
    payload = copy.deepcopy(payload)
    payload["experiment_title"] = EXPERIMENT_TITLE
    payload["input_policy"].update(
        {
            "model_output": "Matrix-fused Shape + Matrix-fused Residual",
            "shape_fusion": "MatrixShapeFusion: S = W2 * P + W3 * T + b",
            "final_fusion": "MatrixComponentFusion: Y = W6 * S + W7 * R + b",
            "weibull_attention": "Causal per-output-step Weibull attention over relative lag j - i + 1",
            "weibull_attention_simplified": False,
            "weibull_position_semantics": "lag=1 is the most recent causally available hidden state",
            "weibull_causal_mask": True,
            "weibull_log_space_softmax": True,
            "weibull_kappa_init": 2.0,
            "weibull_lambda_init": 4.0,
            "residual_hidden_fusion": "learned gate between context_seq and seq_hidden, initialized to 0.5",
            "final_addition": False,
            "matrix_fusion_enabled": True,
            "residual_weather_features": ["ghi_scaled", "temp_scaled"],
            "no_user_residual": True,
            "no_daylight_gate_on_residual": True,
            "loss": "STE only",
        }
    )
    payload["experiment_config"] = asdict(cfg)
    return payload


def install_shared_pipeline_hooks():
    """Bind the shared trainer to this matrix/full-attention model."""

    baseline.EXPERIMENT_TITLE = EXPERIMENT_TITLE
    baseline._baseline_config_payload = _baseline_config_payload
    baseline.install_shared_pipeline_hooks()
    baseline.base.WeatherAwareAEShiftNetLoadModel = WeatherAwareAEShiftNetLoadModel
    baseline.base.compute_losses = compute_losses
    baseline.base.run_epoch = run_epoch
    baseline.base.build_prediction_dataframe = build_prediction_dataframe
    baseline.base.save_test_diagnostic_plots = save_test_diagnostic_plots


def train_one_client(
    client_id: int,
    csv_path: str,
    args,
    base_cfg: WeatherAwareAEShiftExperimentConfig,
):
    return baseline.train_one_client(client_id, csv_path, args, base_cfg)


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
    for client_id, csv_path in baseline.base.select_clients(args):
        summaries.append(train_one_client(client_id, csv_path, args, exp_cfg))

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(
        args.output_root, "all_clients_test_metrics_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(
        f"Finished {len(summaries)} client(s). "
        f"Summary: {os.path.abspath(summary_path)}"
    )


if __name__ == "__main__":
    main()
