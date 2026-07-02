# Running Guide

This guide lists the main commands for reproducing the forecasting experiments. Run all commands from the repository root.

## 1. Create the Python Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a CUDA GPU is available, install the PyTorch package variant that matches the local CUDA driver.

## 2. Check the Data Configuration

The default client files are configured in `config.py`:

```text
per_client_merged/client_1_load_weather_30min.csv
per_client_merged/client_2_load_weather_30min.csv
...
per_client_merged/client_9_load_weather_30min.csv
```

Before running an experiment, confirm that the configured files exist and include the required columns:

- `timestamp`
- `gc`
- `gg`
- `net_load`
- weather or price columns when their feature switches are enabled in `config.py`

## 3. Federated Training

The default federated experiment uses the settings in `config.py`.

```bash
python federated_main.py
```

Useful settings to review before running:

- `CFG.data.target_col`
- `CFG.data.seq_len`
- `CFG.data.horizon`
- `CFG.train.epochs`
- `CFG.federated.rounds`
- `CFG.federated.local_epochs`
- `CFG.federated.aggregation_method`
- `CFG.federated.save_dir`

## 4. Centralized Baseline

```bash
python centralized_main.py
```

This script builds a centralized aggregate dataframe and trains one CNN-LSTM baseline for direct net-load forecasting.

## 5. Local CNN-LSTM Attention Baseline

```powershell
python cnn_lstm_attention_netload.py `
  --data-path per_client_merged/client_1_load_weather_30min.csv `
  --save-dir runs/cnn_lstm_attention_client1 `
  --seq-len 48 `
  --horizon 1 `
  --epochs 60 `
  --batch-size 512
```

The same command can also be run on one line:

```bash
python cnn_lstm_attention_netload.py --data-path per_client_merged/client_1_load_weather_30min.csv --save-dir runs/cnn_lstm_attention_client1 --seq-len 48 --horizon 1 --epochs 60 --batch-size 512
```

The available input features are `net_load`, `ghi`, `temperature`, and `wind`.

```bash
python cnn_lstm_attention_netload.py --input-features net_load ghi temperature wind
```

## 6. Decomposed Net-Load Experiment

Train the PV/load decomposition workflow for one client:

```bash
python decomposed_net_load_main.py --client-id 2 --epochs 60 --output-root runs/pv_decomposition_client2
```

Train all configured clients with the default config:

```bash
python decomposed_net_load_main.py --output-root runs/pv_decomposition_all
```

## 7. H2A and FedAvg Sweep

```bash
python run_gc_h2a_sweep.py
```

This runner creates a grid-consumption sweep under `runs/gc_h2a_sweep/` and writes a CSV summary.

## 8. Output Files

Most experiments write outputs under `runs/`. Typical files include:

- `config.json`
- `training_log.csv`
- `federated_round_logs.csv`
- `test_metrics.csv`
- `test_predictions.csv`
- `best_model.pth`
- `final_model.pth`
- prediction plots and validation curves

Large output folders can be archived or moved out of the repository when preparing a lightweight code-only snapshot.
