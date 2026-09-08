# INITNET IDS — Intrusion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square&logo=scikit-learn)

**Hệ thống phát hiện xâm nhập mạng dựa trên học máy không giám sát**

Phát hiện bất thường trong network traffic và system logs theo thời gian thực  
bằng LSTM Autoencoder — pipeline streaming đa luồng.

*Đồ án môn học — Nhóm INITNET*

</div>

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc Project](#cấu-trúc-project)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Mô hình ML](#mô-hình-ml)
- [Threshold Động](#threshold-động)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Phân công nhóm](#phân-công-nhóm)

---

## Giới thiệu

INITNET IDS là hệ thống phát hiện xâm nhập (Intrusion Detection System) kết hợp hai hướng tiếp cận:

- **Unsupervised Learning** (LSTM Autoencoder, Dense Autoencoder): Phát hiện bất thường dựa trên reconstruction error — không cần nhãn tấn công → có khả năng phát hiện **zero-day attacks**.
- **Supervised Learning** (Random Forest): Làm baseline để đối chiếu hiệu năng.

Hệ thống xử lý song song hai nguồn dữ liệu:

| Nguồn | Mô tả | Tiền xử lý |
|---|---|---|
| **Network Traffic** | Luồng IP (flow features) | MinMaxScaler |
| **System Log (HDFS)** | Chuỗi event log | TF-IDF Vectorizer (32 features) |

---

## Kiến trúc hệ thống

### Pipeline Streaming Đa Luồng (3 Thread)

```
demo_log.csv     ──► [Thread 1: LogIngestionThread]   ──► log_queue     ──► [Thread 3a: InferenceThread-Log]     ──► Alert Log
demo_traffic.csv ──► [Thread 2: PacketAnalysisThread]  ──► traffic_queue ──► [Thread 3b: InferenceThread-Traffic] ──► Alert Traffic
```

| Thread | Nhiệm vụ |
|---|---|
| `LogIngestionThread` | Đọc log CSV → parse EventSequence → TF-IDF vectorize → đẩy queue |
| `PacketAnalysisThread` | Đọc traffic CSV → MinMax scale → tạo sliding window (step=5) → đẩy queue |
| `InferenceThread` | Lấy từ queue → LSTM-AE inference → so threshold → phát cảnh báo real-time |

### Tổng quan hệ thống

```
┌──────────────────────────────────────────────────────────────┐
│                       StreamingPipeline                      │
│                                                              │
│  [CSV Data] ──► [Ingestion Threads] ──► [Queues]            │
│                                              │               │
│                                     [Inference Threads]      │
│                                              │               │
│                           ┌──────────────────┴──────────┐    │
│                           │  LSTM Autoencoder            │    │
│                           │  Dynamic Threshold           │    │
│                           │  Alert + Metrics             │    │
│                           └─────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
         Terminal Alerts   Dashboard PNG    Metrics Report
```

---

## Cấu trúc Project

```
IDS_Project/
│
├── datas/                              # Dữ liệu đầu vào
│   ├── train_traffic.csv               # Network traffic bình thường (20,000 flows, ~7MB)
│   ├── demo_traffic.csv                # Traffic hỗn hợp có nhãn tấn công (1,786 flows)
│   ├── train_log.csv                   # HDFS log bình thường (5,000 sequences, ~970KB)
│   └── demo_log.csv                    # Log hỗn hợp có nhãn (1,000 sequences)
│
├── model/                              # Các model đã train
│   ├── autoencoder_traffic.h5          # LSTM Autoencoder — Traffic (train trên Colab)
│   ├── autoencoder_log.h5              # LSTM Autoencoder — Log (train trên Colab)
│   ├── dense_autoencoder_traffic.h5    # Dense Autoencoder — Traffic
│   ├── dense_autoencoder_log.h5        # Dense Autoencoder — Log
│   ├── network_scaler.pkl              # MinMaxScaler đã fit trên train_traffic
│   ├── log_vectorizer.pkl              # TF-IDF Vectorizer đã fit trên train_log
│   ├── rf_traffic.pkl                  # Random Forest — Traffic (supervised)
│   └── rf_log.pkl                      # Random Forest — Log (supervised)
│
├── src/
│   ├── pipeline/
│   │   ├── log_ingestion.py            # Thread 1: đọc log, vectorize, đẩy queue
│   │   ├── packet_analysis.py          # Thread 2: đọc traffic, scale, tạo sequence, đẩy queue
│   │   ├── model_inference.py          # Thread 3: inference, đo latency & memory, phát alert
│   │   └── streaming_pipeline.py       # Orchestrator: khởi tạo và điều phối 3 thread
│   │
│   ├── models/
│   │   └── dense_autoencoder.py        # Định nghĩa, train, và evaluate Dense Autoencoder
│   │
│   ├── analysis/
│   │   ├── dynamic_threshold.py        # Các chiến lược tính threshold động
│   │   ├── metrics.py                  # Accuracy, Precision, Recall, F1, AUC-ROC, Latency, Memory
│   │   └── comparison.py               # So sánh Supervised vs Unsupervised
│   │
│   └── dashboard/
│       └── realtime_dashboard.py       # Dashboard dark-mode (8 subplot, xuất PNG)
│
├── results/                            # Output sau khi chạy
│   ├── threshold_analysis_traffic.png  # MSE distribution + F1 vs Threshold (Traffic)
│   ├── threshold_analysis_log.png      # MSE distribution + F1 vs Threshold (Log)
│   ├── demo_dashboard.png              # Dashboard tổng hợp sau demo
│   └── comparison_table.csv            # Bảng so sánh metrics tất cả model
│
├── train_models.py                     # [SCRIPT] Train Dense AE + RF, phân tích, xuất kết quả
├── run_demo.py                         # [SCRIPT] Demo pipeline streaming (dùng trước hội đồng)
└── README.md
```

---

## Cài đặt

### Yêu cầu

- Python 3.8+
- pip

### Cài dependencies

```bash
pip install tensorflow scikit-learn pandas numpy joblib psutil matplotlib
```

Hoặc trên Linux/macOS dùng Python system:

```bash
pip3 install tensorflow scikit-learn pandas numpy joblib psutil matplotlib --break-system-packages
```

> **Lưu ý:** Các model LSTM Autoencoder (`.h5`) đã được train sẵn trên Google Colab và lưu trong thư mục `model/`.
> Nếu chưa có file Dense AE hoặc Random Forest, chạy `train_models.py` để tạo trước.

---

## Hướng dẫn sử dụng

### Bước 1 — Train models & phân tích (tùy chọn)

```bash
python train_models.py
```

Script thực hiện tuần tự:

1. Load và tiền xử lý dữ liệu traffic & log
2. Evaluate LSTM Autoencoder từ model `.h5` đã có
3. Train Dense Autoencoder (Traffic + Log, 25 epochs)
4. Train Random Forest Classifier (supervised baseline)
5. Phân tích threshold động → vẽ biểu đồ MSE distribution và F1 vs Threshold
6. Xuất bảng so sánh → `results/comparison_table.csv`
7. Xuất biểu đồ → `results/threshold_analysis_*.png`

### Bước 2 — Chạy Demo (pipeline streaming)

```bash
python run_demo.py
```

Script thực hiện:

1. Kiểm tra toàn bộ file model và data cần thiết
2. Khởi động pipeline 3 thread song song
3. In cảnh báo real-time lên terminal (NORMAL / ANOMALY)
4. Sau khi hoàn tất → in báo cáo metrics đầy đủ
5. Vẽ và lưu dashboard → `results/demo_dashboard.png`

**Các tham số pipeline** (chỉnh trong `run_demo.py`):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `traffic_threshold` | `0.013591` | Ngưỡng MSE Traffic (mean + 2×std) |
| `log_threshold` | `0.014254` | Ngưỡng MSE Log (mean + 2×std) |
| `time_steps` | `5` | Độ dài sliding window |
| `delay_ms` | `15.0` | Delay giữa các batch (ms) |
| `dynamic_threshold` | `True` | Bật cập nhật threshold động |
| `timeout` | `600.0` | Thời gian chạy tối đa (giây) |

Nhấn `Ctrl+C` để dừng pipeline an toàn (có signal handler tránh core dump TensorFlow).

---

## Mô hình ML

### LSTM Autoencoder (Sequence-to-Sequence)

Mô hình chính, train trên dữ liệu **bình thường** (unsupervised — không cần nhãn tấn công):

```
Input (time_steps=5, features)
  → LSTM(32, return_sequences=False)
  → RepeatVector(5)
  → LSTM(32, return_sequences=True)
  → TimeDistributed(Dense(features))
  → Output (5, features)
```

| Thông số | Giá trị |
|---|---|
| Loss | MSE (Mean Squared Error) |
| Threshold | `mean(train_MSE) + 2 × std(train_MSE)` |
| Phát hiện | `reconstruction_error > threshold` → **ANOMALY** |
| Sliding window | `time_steps = 5` |

### Dense Autoencoder (Tabular)

Model nhẹ hơn, train trực tiếp qua `train_models.py`:

```
Input(dim)
  → Dense(32) + BatchNorm + ReLU
  → Dense(16) + ReLU                  ← Bottleneck
  → Dense(32) + BatchNorm + ReLU
  → Dense(dim, sigmoid)
  → Output(dim)
```

| Thông số | Giá trị |
|---|---|
| Epochs | 25 |
| Optimizer | Adam |
| Loss | MSE |

### Random Forest (Supervised — Baseline)

- `n_estimators = 100`, train **với nhãn** (60% demo data).
- Chỉ phát hiện được tấn công **đã biết** — dùng để so sánh, không dùng trong pipeline demo chính.

---

## Threshold Động

Hệ thống hỗ trợ 3 chiến lược tính threshold:

| Chiến lược | Công thức | Mô tả |
|---|---|---|
| `mean_std` *(mặc định)* | `μ + k×σ` với `k=2.0` | Dựa trên phân phối train MSE |
| `percentile` | `percentile(train_errors, 95)` | Tính theo phân vị |
| `sliding_window` | Cập nhật liên tục | Tự điều chỉnh theo thời gian thực |

**Kết quả phân tích F1 vs Threshold:**

| Data | Best threshold (p90) | F1 tốt nhất |
|---|---|---|
| Network Traffic | `0.0107` | ~0.43 |
| System Log | `0.0111` | ~0.62 |

---

## Kết quả thực nghiệm

### Bảng so sánh tổng hợp

| Mô hình | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Loại |
|---|---|---|---|---|---|---|
| LSTM-AE (Traffic) | 0.5129 | 0.4320 | 0.3410 | 0.3812 | 0.5158 | Unsupervised |
| Dense-AE (Traffic) | 0.6756 | **0.9157** | 0.2901 | 0.4406 | **0.8272** | Unsupervised |
| **Random Forest (Traffic)** | **0.9916** | **0.9936** | **0.9873** | **0.9904** | N/A | Supervised |
| LSTM-AE (Log) | 0.5156 | 0.5101 | **0.7117** | 0.5943 | 0.5127 | Unsupervised |
| Dense-AE (Log) | 0.7420 | **0.9481** | 0.5120 | 0.6649 | **0.7538** | Unsupervised |
| **Random Forest (Log)** | **0.7550** | 0.9397 | 0.5450 | **0.6899** | N/A | Supervised |

### Nhận xét

- **Random Forest (Supervised):** F1 vượt trội, nhưng chỉ phát hiện được tấn công **đã thấy trong tập train** — không phát hiện được zero-day.
- **Dense-AE Traffic:** Precision = 0.92, AUC-ROC = 0.83 — ít False Positive, phù hợp môi trường production.
- **LSTM-AE Log:** Recall = 0.71 — bắt được nhiều anomaly nhất trong dữ liệu log.
- **Ưu điểm then chốt của Unsupervised:** Không cần nhãn tấn công → phát hiện được **zero-day attacks** và hành vi bất thường chưa từng gặp.

---

## Phân công nhóm

| Thành viên | Nhiệm vụ |
|---|---|
| Member 1 | Data preprocessing, Train LSTM Autoencoder trên Google Colab |
| Member 2 | Pipeline streaming 3 thread (`pipeline/`), đo latency & memory |
| Member 3 | Dense Autoencoder (`models/`), so sánh Supervised, phân tích Threshold động |
| Member 4 | Dashboard (`dashboard/`), demo script (`run_demo.py`), báo cáo |

---

*Dự án phát triển cho mục đích học thuật trong khuôn khổ đồ án môn học.*
