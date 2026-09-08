# IDS - Intrusion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square&logo=scikit-learn)

**A real-time network intrusion detection system powered by unsupervised deep learning**

Detects anomalies in network traffic and system logs using LSTM Autoencoder with a concurrent multi-threaded streaming pipeline.

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [ML Models](#ml-models)
- [Dynamic Thresholding](#dynamic-thresholding)
- [Results](#results)

---

## Overview

IDS is a real-time Intrusion Detection System that combines two approaches:

- **Unsupervised Learning** (LSTM Autoencoder, Dense Autoencoder): Detects anomalies via reconstruction error - no attack labels required, enabling detection of **zero-day attacks**.
- **Supervised Learning** (Random Forest): Used as a performance baseline for comparison.

The system processes two data sources in parallel:

| Source | Description | Preprocessing |
|---|---|---|
| **Network Traffic** | IP flow features | MinMaxScaler |
| **System Log (HDFS)** | Event log sequences | TF-IDF Vectorizer (32 features) |

---

## System Architecture

### Concurrent 3-Thread Streaming Pipeline

```
demo_log.csv     --> [Thread 1: LogIngestionThread]   --> log_queue     --> [Thread 3a: InferenceThread-Log]     --> Alert
demo_traffic.csv --> [Thread 2: PacketAnalysisThread] --> traffic_queue --> [Thread 3b: InferenceThread-Traffic] --> Alert
```

| Thread | Responsibility |
|---|---|
| `LogIngestionThread` | Read log CSV, parse EventSequence, TF-IDF vectorize, push to queue |
| `PacketAnalysisThread` | Read traffic CSV, MinMax scale, create sliding window (step=5), push to queue |
| `InferenceThread` | Dequeue, LSTM-AE inference, compare threshold, emit real-time alert |

### System Overview

```
[CSV Data] --> [Ingestion Threads] --> [Queues] --> [Inference Threads] --> LSTM-AE + Dynamic Threshold --> Alerts
                                                                                    |
                                          Terminal Alerts | Dashboard PNG | Metrics Report
```

---

## Project Structure

```
IDS_Project/
|
+-- datas/                              # Input datasets
|   +-- train_traffic.csv               # Normal network traffic (20,000 flows, ~7MB)
|   +-- demo_traffic.csv                # Mixed traffic with attack labels (1,786 flows)
|   +-- train_log.csv                   # Normal HDFS logs (5,000 sequences, ~970KB)
|   +-- demo_log.csv                    # Mixed logs with labels (1,000 sequences)
|
+-- model/                              # Trained model artifacts
|   +-- autoencoder_traffic.h5          # LSTM Autoencoder - Traffic
|   +-- autoencoder_log.h5              # LSTM Autoencoder - Log
|   +-- dense_autoencoder_traffic.h5    # Dense Autoencoder - Traffic
|   +-- dense_autoencoder_log.h5        # Dense Autoencoder - Log
|   +-- network_scaler.pkl              # MinMaxScaler fitted on train_traffic
|   +-- log_vectorizer.pkl              # TF-IDF Vectorizer fitted on train_log
|   +-- rf_traffic.pkl                  # Random Forest - Traffic (supervised)
|   +-- rf_log.pkl                      # Random Forest - Log (supervised)
|
+-- src/
|   +-- pipeline/
|   |   +-- log_ingestion.py            # Thread 1: read logs, vectorize, push queue
|   |   +-- packet_analysis.py          # Thread 2: read traffic, scale, create sequences, push queue
|   |   +-- model_inference.py          # Thread 3: inference, measure latency & memory, emit alerts
|   |   +-- streaming_pipeline.py       # Orchestrator: initialize and coordinate 3 threads
|   |
|   +-- models/
|   |   +-- dense_autoencoder.py        # Dense Autoencoder definition, training, evaluation
|   |
|   +-- analysis/
|   |   +-- dynamic_threshold.py        # Dynamic threshold strategies
|   |   +-- metrics.py                  # Accuracy, Precision, Recall, F1, AUC-ROC, Latency, Memory
|   |   +-- comparison.py               # Supervised vs Unsupervised comparison
|   |
|   +-- dashboard/
|       +-- realtime_dashboard.py       # Dark-mode dashboard (8 subplots, exports PNG)
|
+-- results/                            # Output after running
|   +-- threshold_analysis_traffic.png  # MSE distribution + F1 vs Threshold (Traffic)
|   +-- threshold_analysis_log.png      # MSE distribution + F1 vs Threshold (Log)
|   +-- demo_dashboard.png              # Full analysis dashboard
|   +-- comparison_table.csv            # Model comparison metrics table
|
+-- train_models.py                     # Train Dense AE + RF, analyze thresholds, export results
+-- run_demo.py                         # Run the streaming pipeline demo
+-- README.md
```

---

## Installation

### Requirements

- Python 3.8+
- pip

### Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy joblib psutil matplotlib
```

On Linux/macOS with system Python:

```bash
pip3 install tensorflow scikit-learn pandas numpy joblib psutil matplotlib --break-system-packages
```

> **Note:** LSTM Autoencoder models (`.h5`) are pre-trained and stored in `model/`.
> Run `train_models.py` first if Dense Autoencoder or Random Forest files are missing.

---

## Usage

### Step 1 - Train and evaluate models (optional)

```bash
python train_models.py
```

This script will:

1. Load and preprocess traffic and log data
2. Evaluate the pre-trained LSTM Autoencoder
3. Train Dense Autoencoder (Traffic + Log, 25 epochs)
4. Train Random Forest Classifier (supervised baseline)
5. Analyze dynamic thresholds and plot MSE distribution and F1 vs Threshold
6. Export comparison table to `results/comparison_table.csv`
7. Export plots to `results/threshold_analysis_*.png`

### Step 2 - Run the streaming pipeline

```bash
python run_demo.py
```

This script will:

1. Validate all required model and data files
2. Launch the 3-thread pipeline concurrently
3. Print real-time alerts to the terminal (NORMAL / ANOMALY)
4. Print a full metrics report after completion
5. Render and save the dashboard to `results/demo_dashboard.png`

**Pipeline parameters** (configurable in `run_demo.py`):

| Parameter | Default | Description |
|---|---|---|
| `traffic_threshold` | `0.013591` | MSE threshold for traffic (mean + 2*std) |
| `log_threshold` | `0.014254` | MSE threshold for logs (mean + 2*std) |
| `time_steps` | `5` | Sliding window length |
| `delay_ms` | `15.0` | Delay between batches (ms) |
| `dynamic_threshold` | `True` | Enable adaptive thresholding |
| `timeout` | `600.0` | Maximum pipeline runtime (seconds) |

Press `Ctrl+C` to stop the pipeline safely (signal handler prevents TensorFlow core dump).

---

## ML Models

### LSTM Autoencoder (Sequence-to-Sequence)

The primary model, trained exclusively on **normal** data (unsupervised - no attack labels needed):

```
Input (time_steps=5, features)
  -> LSTM(32, return_sequences=False)
  -> RepeatVector(5)
  -> LSTM(32, return_sequences=True)
  -> TimeDistributed(Dense(features))
  -> Output (5, features)
```

| Parameter | Value |
|---|---|
| Loss | MSE (Mean Squared Error) |
| Threshold | mean(train_MSE) + 2 * std(train_MSE) |
| Detection rule | reconstruction_error > threshold -> **ANOMALY** |
| Sliding window | time_steps = 5 |

### Dense Autoencoder (Tabular)

A lighter model trained locally via `train_models.py`:

```
Input(dim)
  -> Dense(32) + BatchNorm + ReLU
  -> Dense(16) + ReLU                  <- Bottleneck
  -> Dense(32) + BatchNorm + ReLU
  -> Dense(dim, sigmoid)
  -> Output(dim)
```

| Parameter | Value |
|---|---|
| Epochs | 25 |
| Optimizer | Adam |
| Loss | MSE |

### Random Forest (Supervised Baseline)

- `n_estimators = 100`, trained **with labels** (60% of demo data).
- Detects only **known** attack patterns - used for comparison only, not in the main pipeline.

---

## Dynamic Thresholding

Three threshold strategies are supported:

| Strategy | Formula | Description |
|---|---|---|
| `mean_std` *(default)* | mu + k*sigma, k=2.0 | Based on train MSE distribution |
| `percentile` | percentile(train_errors, 95) | Percentile-based cutoff |
| `sliding_window` | Continuously updated | Adapts to real-time data drift |

**F1 vs Threshold analysis results:**

| Data | Best threshold (p90) | Best F1 |
|---|---|---|
| Network Traffic | 0.0107 | ~0.43 |
| System Log | 0.0111 | ~0.62 |

---

## Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Type |
|---|---|---|---|---|---|---|
| LSTM-AE (Traffic) | 0.5129 | 0.4320 | 0.3410 | 0.3812 | 0.5158 | Unsupervised |
| Dense-AE (Traffic) | 0.6756 | **0.9157** | 0.2901 | 0.4406 | **0.8272** | Unsupervised |
| **Random Forest (Traffic)** | **0.9916** | **0.9936** | **0.9873** | **0.9904** | N/A | Supervised |
| LSTM-AE (Log) | 0.5156 | 0.5101 | **0.7117** | 0.5943 | 0.5127 | Unsupervised |
| Dense-AE (Log) | 0.7420 | **0.9481** | 0.5120 | 0.6649 | **0.7538** | Unsupervised |
| **Random Forest (Log)** | **0.7550** | 0.9397 | 0.5450 | **0.6899** | N/A | Supervised |

### Key Takeaways

- **Random Forest (Supervised):** Highest F1, but only detects **known** attack patterns - unable to generalize to unseen threats.
- **Dense-AE Traffic:** Precision = 0.92, AUC-ROC = 0.83 - low false positive rate, suitable for production environments.
- **LSTM-AE Log:** Recall = 0.71 - highest anomaly capture rate on log data.
- **Core advantage of unsupervised approach:** No labeled attack data required, capable of detecting **zero-day attacks** and previously unseen malicious behavior.

---
