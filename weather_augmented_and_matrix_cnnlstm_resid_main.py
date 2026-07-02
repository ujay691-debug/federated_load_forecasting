"""Matrix-fused baseline with a CNN-LSTM-Attention residual predictor.

Only the residual predictor differs from
``weather_augmented_and_weibull_matrix_fullattn_main.py``.  Shape branches,
matrix fusion, AE residual construction, and the STE-only objective are shared.
"""

import argparse
import copy
import os
from dataclasses import asdict, dataclass
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from models.cnn_lstm import Attention, SamePadMaxPool1d
from utils.data_utils import ensure_dir, set_seed

import weather_augmented_and_weibull_matrix_fullattn_main as fullattn


EXPERIMENT_TITLE = (
    "Weather-augmented AND matrix baseline | CNN-LSTM-Attention Residual"
)
MatrixShapeFusion = fullattn.MatrixShapeFusion
MatrixComponentFusion = fullattn.MatrixComponentFusion
PeriodicSineSequenceBranch = fullattn.PeriodicSineSequenceBranch
LinearTrendBranch = fullattn.LinearTrendBranch
SequenceAutoEncoder = fullattn.SequenceAutoEncoder
compute_fourier_loss = fullattn.compute_fourier_loss
compute_losses = fullattn.compute_losses
build_prediction_dataframe = fullattn.build_prediction_dataframe
BASELINE_CONFIG_PAYLOAD = fullattn.BASELINE_CONFIG_PAYLOAD
BASELINE_SAVE_DIAGNOSTIC_PLOTS = fullattn.BASELINE_SAVE_DIAGNOSTIC_PLOTS
save_test_diagnostic_plots = BASELINE_SAVE_DIAGNOSTIC_PLOTS


@dataclass
class WeatherAwareAEShiftExperimentConfig(
    fullattn.WeatherAwareAEShiftExperimentConfig
):
    residual_weather_mode: str = "all5"
    cnn_resid_conv1_channels: int = 32
    cnn_resid_conv2_channels: int = 64
    cnn_resid_kernel: int = 3
    cnn_resid_lstm_hidden1: int = 32
    cnn_resid_lstm_hidden2: int = 16
    cnn_resid_attn_units: int = 20
    cnn_resid_fc_hidden: int = 16

    def __post_init__(self):
        if self.residual_weather_mode not in {"all5", "ghi_temp"}:
            raise ValueError(
                "residual_weather_mode must be 'all5' or 'ghi_temp', "
                f"got {self.residual_weather_mode!r}."
            )


class CNNLSTMAttentionResidualSeqBranch(nn.Module):
    """CNN-LSTM-Attention residual sequence predictor with weather exogenous inputs."""

    def __init__(
        self,
        hist_input_dim: int,
        shift_exog_dim: int,
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
            int(hist_input_dim),
            int(conv1_channels),
            kernel_size=int(kernel_size),
            stride=1,
            padding=int(kernel_size) // 2,
        )
        self.pool1 = SamePadMaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(
            int(conv1_channels),
            int(conv2_channels),
            kernel_size=int(kernel_size),
            stride=1,
            padding=int(kernel_size) // 2,
        )
        self.pool2 = SamePadMaxPool1d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(float(dropout))
        self.lstm1 = nn.LSTM(
            input_size=int(conv2_channels),
            hidden_size=int(lstm_hidden1),
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=int(lstm_hidden1),
            hidden_size=int(lstm_hidden2),
            batch_first=True,
        )
        self.attention = Attention(
            input_dim=int(lstm_hidden2),
            attn_units=int(attn_units),
        )
        self.residual_seq_head = nn.Sequential(
            nn.Linear(
                int(lstm_hidden2) + int(attn_units) + int(shift_exog_dim),
                int(fc_hidden),
            ),
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
        return self.residual_seq_head(decoder_input)


class WeatherAwareAEShiftNetLoadModel(nn.Module):
    """Matrix Shape/Output fusion with CNN-LSTM-Attention Residual."""

    def __init__(self, cfg: WeatherAwareAEShiftExperimentConfig):
        super().__init__()
        if cfg.residual_weather_mode not in {"all5", "ghi_temp"}:
            raise ValueError(
                "residual_weather_mode must be 'all5' or 'ghi_temp', "
                f"got {cfg.residual_weather_mode!r}."
            )
        self.residual_weather_mode = str(cfg.residual_weather_mode)
        weather_dim = 5 if self.residual_weather_mode == "all5" else 2

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
        self.residual_branch = CNNLSTMAttentionResidualSeqBranch(
            hist_input_dim=1 + weather_dim,
            shift_exog_dim=weather_dim,
            conv1_channels=cfg.cnn_resid_conv1_channels,
            conv2_channels=cfg.cnn_resid_conv2_channels,
            kernel_size=cfg.cnn_resid_kernel,
            lstm_hidden1=cfg.cnn_resid_lstm_hidden1,
            lstm_hidden2=cfg.cnn_resid_lstm_hidden2,
            attn_units=cfg.cnn_resid_attn_units,
            fc_hidden=cfg.cnn_resid_fc_hidden,
            dropout=cfg.dropout,
        )
        self.final_fusion = MatrixComponentFusion(seq_len=cfg.seq_len)

    def _select_residual_weather(self, weather: torch.Tensor) -> torch.Tensor:
        if self.residual_weather_mode == "all5":
            return weather
        if self.residual_weather_mode == "ghi_temp":
            return weather[..., 0:2]
        raise ValueError(
            "residual_weather_mode must be 'all5' or 'ghi_temp', "
            f"got {self.residual_weather_mode!r}."
        )

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
        residual_weather_hist = self._select_residual_weather(weather_hist)
        residual_weather_shift = self._select_residual_weather(weather_shift)
        r_raw_shift = self.residual_branch(
            base_resid_hist,
            residual_weather_hist,
            residual_weather_shift,
        )
        r_shift = r_raw_shift  # Deliberately no daylight gate.

        y_additive_direct = s_shift + r_shift
        y_shift_pred = self.final_fusion(s_shift, r_shift)
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
            "Y_additive_direct": y_additive_direct,
            "Y_matrix_fused": y_shift_pred,
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


def parse_args():
    parser = argparse.ArgumentParser(description=EXPERIMENT_TITLE)
    parser.add_argument("--client-id", type=int, default=2)
    parser.add_argument(
        "--output-root",
        default=os.path.join(
            "runs", "weather_augmented_and_matrix_cnnlstm_resid"
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
    parser.add_argument(
        "--residual-weather-mode",
        choices=["all5", "ghi_temp"],
        default="all5",
    )
    parser.add_argument("--cnn-resid-conv1-channels", type=int, default=32)
    parser.add_argument("--cnn-resid-conv2-channels", type=int, default=64)
    parser.add_argument("--cnn-resid-kernel", type=int, default=3)
    parser.add_argument("--cnn-resid-lstm-hidden1", type=int, default=32)
    parser.add_argument("--cnn-resid-lstm-hidden2", type=int, default=16)
    parser.add_argument("--cnn-resid-attn-units", type=int, default=20)
    parser.add_argument("--cnn-resid-fc-hidden", type=int, default=16)

    # Compatibility-only options accepted by older launch commands.
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
    weather_features = (
        [
            "ghi_scaled",
            "temp_scaled",
            "wind_scaled",
            "daylight",
            "ramp_ghi_scaled",
        ]
        if cfg.residual_weather_mode == "all5"
        else ["ghi_scaled", "temp_scaled"]
    )
    payload["experiment_title"] = EXPERIMENT_TITLE
    payload["input_policy"].update(
        {
            "model_output": "Matrix-fused Shape + Matrix-fused Residual",
            "shape_fusion": "MatrixShapeFusion: S = W2 * P + W3 * T + b",
            "final_fusion": "MatrixComponentFusion: Y = W6 * S + W7 * R + b",
            "residual_predictor": "CNN-LSTM-Attention",
            "residual_branch": "CNN-LSTM-Attention",
            "residual_weather_mode": cfg.residual_weather_mode,
            "residual_weather_features": weather_features,
            "weibull_attention": False,
            "no_user_residual": True,
            "no_daylight_gate_on_residual": True,
            "loss": "STE only",
        }
    )
    payload["experiment_config"] = asdict(cfg)
    return payload


def install_shared_pipeline_hooks():
    """Bind the shared trainer to the CNN-LSTM-Attention residual model."""

    baseline = fullattn.baseline
    baseline.EXPERIMENT_TITLE = EXPERIMENT_TITLE
    baseline.WeatherAwareAEShiftExperimentConfig = WeatherAwareAEShiftExperimentConfig
    baseline._baseline_config_payload = _baseline_config_payload
    baseline.install_shared_pipeline_hooks()
    baseline.base.WeatherAwareAEShiftExperimentConfig = WeatherAwareAEShiftExperimentConfig
    baseline.base.WeatherAwareAEShiftNetLoadModel = WeatherAwareAEShiftNetLoadModel
    baseline.base.compute_losses = compute_losses
    baseline.base.run_epoch = fullattn.ORIGINAL_RUN_EPOCH
    baseline.base.build_prediction_dataframe = build_prediction_dataframe
    baseline.base.save_test_diagnostic_plots = save_test_diagnostic_plots


def train_one_client(
    client_id: int,
    csv_path: str,
    args,
    base_cfg: WeatherAwareAEShiftExperimentConfig,
):
    return fullattn.baseline.train_one_client(client_id, csv_path, args, base_cfg)


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
        residual_weather_mode=str(args.residual_weather_mode),
        cnn_resid_conv1_channels=int(args.cnn_resid_conv1_channels),
        cnn_resid_conv2_channels=int(args.cnn_resid_conv2_channels),
        cnn_resid_kernel=int(args.cnn_resid_kernel),
        cnn_resid_lstm_hidden1=int(args.cnn_resid_lstm_hidden1),
        cnn_resid_lstm_hidden2=int(args.cnn_resid_lstm_hidden2),
        cnn_resid_attn_units=int(args.cnn_resid_attn_units),
        cnn_resid_fc_hidden=int(args.cnn_resid_fc_hidden),
    )

    print(EXPERIMENT_TITLE)
    print("Loss: STE only")
    print("Residual predictor: CNN-LSTM-Attention")
    print(f"Residual weather mode: {exp_cfg.residual_weather_mode}")

    summaries = []
    for client_id, csv_path in fullattn.baseline.base.select_clients(args):
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
