# Project Structure

The repository is organized around experiment entry points, shared model code, data utilities, and generated analysis outputs.

## Core Configuration

```text
config.py
```

`config.py` defines the default data paths, target columns, feature switches, model hyperparameters, training settings, federated settings, and output directories. Most scripts import `CFG` from this file and either use it directly or override selected fields.

## Training Entry Points

```text
federated_main.py
centralized_main.py
decentralized_gcml_main.py
decomposed_net_load_main.py
cnn_lstm_attention_netload.py
run_gc_h2a_sweep.py
net_load_experiment_suite.py
```

- `federated_main.py` trains federated models and supports single-target as well as direct or indirect net-load workflows.
- `centralized_main.py` builds a centralized aggregate dataframe and trains a centralized baseline.
- `decentralized_gcml_main.py` contains decentralized client-collaboration experiments.
- `decomposed_net_load_main.py` trains PV/load decomposition models.
- `cnn_lstm_attention_netload.py` trains a local CNN-LSTM attention baseline for one client CSV.
- `run_gc_h2a_sweep.py` runs a predefined FedAvg/H2A grid-consumption sweep.
- `net_load_experiment_suite.py` contains shared utilities for comparing direct and indirect net-load forecasting.

## Model Code

```text
models/
├── cnn_lstm.py
└── h2a_hypernetwork.py
```

`models/cnn_lstm.py` defines CNN-LSTM forecasting modules and attention blocks. `models/h2a_hypernetwork.py` contains hypernetwork components used by H2A-style personalized aggregation.

## Shared Utilities

```text
utils/
├── aggregation.py
├── data_utils.py
├── metrics.py
└── runtime_env.py
```

- `aggregation.py` contains parameter aggregation helpers such as FedAvg.
- `data_utils.py` handles dataframe preparation, feature construction, sequence creation, scaling, and config saving.
- `metrics.py` computes forecasting metrics and writes metric CSVs and plots.
- `runtime_env.py` handles local runtime path setup for environments that need extra DLL paths.

## Federated Components

```text
client.py
server.py
h2a_server.py
```

- `client.py` wraps local training, evaluation, optimizers, losses, and client-level model construction.
- `server.py` implements the core federated server logic.
- `h2a_server.py` extends server-side aggregation with H2A-related personalization logic.

## Data Folders

```text
per_client_merged/
aemo_nsw1_rrp_2010_2013/
clients_with_aemo_rrp/
```

- `per_client_merged/` contains the default per-client load, weather, and net-load CSV inputs.
- `aemo_nsw1_rrp_2010_2013/` stores AEMO NSW1 price data used by selected price-aware experiments.
- `clients_with_aemo_rrp/` stores client datasets augmented with AEMO price features when available.

## Results and Analysis

```text
runs/
analysis_gc/
runs_cnn_lstm_netload_three_experiments/
runs_cnn_lstm_netload_multi_scope/
```

- `runs/` is the main output directory for experiment checkpoints, logs, metrics, predictions, and plots.
- `analysis_gc/` contains comparison tables and figures for grid-consumption experiments.
- `runs_cnn_lstm_*` folders contain saved outputs from earlier CNN-LSTM net-load studies.

## Additional Experiment Scripts

Several scripts explore variants of weather-aware additive decomposition, residual correction, expert modules, and price-aware baselines. Their filenames describe the corresponding experiment family, and most write outputs under `runs/`.
