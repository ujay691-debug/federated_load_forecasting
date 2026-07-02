# Federated Load Forecasting

This repository contains Python experiments for multi-client electricity load forecasting with centralized, local, decentralized, and federated learning baselines. The main workflow uses CNN-LSTM style sequence models on half-hourly client data, supports weather and market-price features, and writes metrics, predictions, plots, and checkpoints under `runs/`.

## Highlights

- Federated aggregation baselines including FedAvg and H2A-style personalized aggregation.
- Centralized and local CNN-LSTM forecasting baselines for comparison.
- Direct and indirect net-load forecasting workflows, including `gc - gg` decomposition.
- Weather-aware additive and residual correction experiments for client-level forecasting.
- Reproducible output folders containing model checkpoints, CSV metrics, prediction files, and diagnostic plots.

## Repository Layout

```text
federated_load_forecasting/
+-- config.py                         # Shared experiment, data, feature, model, and training settings
+-- federated_main.py                 # Federated training entry point
+-- centralized_main.py               # Centralized baseline entry point
+-- decentralized_gcml_main.py        # Decentralized client-collaboration experiments
+-- cnn_lstm_attention_netload.py     # Local CNN-LSTM attention net-load baseline
+-- decomposed_net_load_main.py       # PV/load decomposition experiment
+-- net_load_experiment_suite.py      # Direct/indirect net-load experiment utilities
+-- run_gc_h2a_sweep.py               # H2A/FedAvg sweep runner
+-- client.py                         # Client-side training and evaluation helpers
+-- server.py                         # Federated server and aggregation helpers
+-- h2a_server.py                     # H2A aggregation server
+-- models/                           # Neural network modules
+-- utils/                            # Data processing, metrics, aggregation, runtime helpers
+-- per_client_merged/                # Default per-client input CSV files
+-- aemo_nsw1_rrp_2010_2013/          # AEMO price data used by selected experiments
+-- analysis_gc/                      # Analysis outputs and comparison artifacts
+-- runs/                             # Training outputs, checkpoints, predictions, and plots
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for a more detailed directory guide.

## Setup

Python 3.9 or newer is recommended. If you use CUDA, install the PyTorch build that matches your local CUDA driver first, then install the rest of the requirements.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Data Expectations

The default configuration reads nine client files from `per_client_merged/`:

```text
client_1_load_weather_30min.csv
client_2_load_weather_30min.csv
...
client_9_load_weather_30min.csv
```

Core columns used by the main workflows include:

- `timestamp`
- `gc` for grid consumption
- `gg` for local generation
- `net_load`
- optional weather and market features such as `temp2m_c`, `wind10m_ms`, `ghi_wm2`, `rh2m_pct`, and `rrp_aud_per_mwh`

Most defaults are defined in `config.py`, including client file paths, sequence length, forecast horizon, feature switches, training epochs, federated rounds, and output directories.

## Running Experiments

Run commands from the repository root.

```bash
# Federated training with the default settings in config.py
python federated_main.py

# Centralized direct net-load baseline
python centralized_main.py

# Local CNN-LSTM attention baseline for one client
python cnn_lstm_attention_netload.py --data-path per_client_merged/client_1_load_weather_30min.csv --save-dir runs/cnn_lstm_attention_client1 --epochs 60 --seq-len 48 --horizon 1

# PV/load decomposition experiment for a single client
python decomposed_net_load_main.py --client-id 2 --epochs 60 --output-root runs/pv_decomposition_client2

# H2A and FedAvg sweep for grid-consumption forecasting
python run_gc_h2a_sweep.py
```

More examples are in [docs/RUNNING.md](docs/RUNNING.md).

## Outputs

Experiment outputs are written under `runs/` by default. Common artifacts include:

- `config.json` for the resolved experiment settings
- `training_log.csv` or `federated_round_logs.csv`
- `test_metrics.csv` and summary CSV files
- `test_predictions.csv`
- `best_model.pth`, `final_model.pth`, or federated checkpoint files
- prediction and loss-curve plots

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
