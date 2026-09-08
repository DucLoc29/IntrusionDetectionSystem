"""
train_models.py
---------------
Train và đánh giá:
  1. Dense Autoencoder (Traffic)
  2. Dense Autoencoder (Log)  
  3. Supervised Classifier (Random Forest) — để so sánh
  4. Xuất bảng so sánh Supervised vs Unsupervised (LSTM-AE vs Dense-AE vs RF)
  5. Vẽ đồ thị threshold động
  
Chạy: python train_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import ast
import warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "datas")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from analysis.dynamic_threshold import DynamicThreshold
from analysis.metrics import (compute_classification_metrics,
                               print_metrics_report,
                               LatencyTracker, MemoryMonitor)
from analysis.comparison import SupervisedClassifier, run_comparison, print_comparison_table
from models.dense_autoencoder import (build_dense_autoencoder,
                                       compute_reconstruction_error,
                                       train_and_evaluate_dense)


# ============================================================
# 0. Tiền xử lý dữ liệu
# ============================================================
def load_traffic_data():
    print("\n[*] Đang tải dữ liệu Network Traffic...")
    from sklearn.preprocessing import MinMaxScaler

    df_train = pd.read_csv(os.path.join(DATA_DIR, "train_traffic.csv"))
    df_demo  = pd.read_csv(os.path.join(DATA_DIR, "demo_traffic.csv"))

    for df in [df_train, df_demo]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    y_demo = df_demo["Label"].values
    X_train = df_train.values
    X_demo  = df_demo.drop(columns=["Label"]).values

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_demo_s  = scaler.transform(X_demo)

    print(f"    Train: {X_train_s.shape} | Demo: {X_demo_s.shape}")
    return X_train_s, X_demo_s, X_demo, y_demo, scaler


def load_log_data():
    print("\n[*] Đang tải dữ liệu System Log...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    df_train = pd.read_csv(os.path.join(DATA_DIR, "train_log.csv"))
    df_demo  = pd.read_csv(os.path.join(DATA_DIR, "demo_log.csv"))
    df_train.dropna(inplace=True)
    df_demo.dropna(inplace=True)

    def parse_seq(val):
        try:
            p = ast.literal_eval(str(val))
            if isinstance(p, list): return " ".join(str(e) for e in p)
        except: pass
        return str(val)

    seq_col = "EventSequence"
    train_str = df_train[seq_col].apply(parse_seq).values
    demo_str  = df_demo[seq_col].apply(parse_seq).values
    y_demo    = df_demo["Label"].values

    vectorizer = TfidfVectorizer(max_features=32)
    X_train_s = vectorizer.fit_transform(train_str).toarray()
    X_demo_s  = vectorizer.transform(demo_str).toarray()

    print(f"    Train: {X_train_s.shape} | Demo: {X_demo_s.shape}")
    return X_train_s, X_demo_s, y_demo, vectorizer


# ============================================================
# 1. LSTM Autoencoder (đã train, chỉ load để lấy MSE)
# ============================================================
def evaluate_lstm_ae(model_path, X_train, X_demo, y_demo,
                     name="LSTM-AE", is_sequence=True):
    import tensorflow as tf
    print(f"\n[*] Đánh giá {name} từ model đã lưu...")

    # compile=False để tránh lỗi tương thích Keras 2 → Keras 3
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse')

    ts = 5
    def create_sequences(data, ts=5):
        # Tạo n-ts sequences, mỗi sequence gồm ts bước liên tiếp
        return np.array([data[i:i+ts] for i in range(len(data)-ts)])

    if is_sequence:
        X_train_seq = create_sequences(X_train, ts)
        X_demo_seq  = create_sequences(X_demo, ts)
        # create_sequences tạo len-ts samples → align nhãn từ index ts trở đi
        y_demo_seq  = y_demo[ts:]
    else:
        X_train_seq = X_train
        X_demo_seq  = X_demo
        y_demo_seq  = y_demo

    # Fit threshold từ train errors (dùng model thực)
    train_mse = compute_reconstruction_error(model, X_train_seq)
    dt = DynamicThreshold(strategy="mean_std", k=2.0)
    threshold = dt.fit(train_mse)

    demo_mse = compute_reconstruction_error(model, X_demo_seq)

    # Debug: in thống kê MSE để verify threshold
    print(f"    Train MSE: mean={train_mse.mean():.6f} std={train_mse.std():.6f}")
    print(f"    Demo  MSE: mean={demo_mse.mean():.6f} std={demo_mse.std():.6f} max={demo_mse.max():.6f}")
    print(f"    Threshold (mean+2std): {threshold:.6f} | Alerts: {(demo_mse > threshold).sum()}/{len(demo_mse)}")

    y_pred = ["ANOMALY" if e > threshold else "NORMAL" for e in demo_mse]

    metrics = compute_classification_metrics(y_demo_seq, y_pred, demo_mse, threshold)
    metrics["model_name"] = name
    metrics["threshold"] = threshold
    metrics["demo_mse"] = demo_mse
    metrics["train_mse"] = train_mse
    metrics["y_demo"] = y_demo_seq

    print_metrics_report(metrics, title=f"{name} — Kết quả đánh giá")
    return metrics, demo_mse, y_demo_seq, threshold, train_mse


# ============================================================
# 2. Dense Autoencoder
# ============================================================
def train_dense_ae(X_train, X_demo, y_demo, name, save_path,
                   epochs=25):
    print(f"\n[*] Training {name}...")
    model, metrics, demo_mse = train_and_evaluate_dense(
        X_train=X_train,
        X_demo=X_demo,
        y_demo=y_demo,
        threshold_strategy="mean_std",
        epochs=epochs,
        save_path=save_path,
    )
    metrics["model_name"] = name
    print_metrics_report(metrics, title=f"{name} — Kết quả đánh giá")
    return model, metrics, demo_mse


# ============================================================
# 3. Supervised (Random Forest)
# ============================================================
def train_supervised(X_train_raw, y_train, X_demo_raw, y_demo,
                     name="Random Forest (Supervised)", save_path=None):
    print(f"\n[*] Training {name}...")
    from sklearn.preprocessing import MinMaxScaler

    clf = SupervisedClassifier(n_estimators=100)
    clf.fit(X_train_raw, y_train)

    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    y_pred = clf.predict(X_demo_raw)

    def to_bin(arr):
        return np.array([0 if str(v).strip().upper() in ("BENIGN", "NORMAL", "0") else 1 for v in arr])

    yt = to_bin(y_demo)
    yp = to_bin(y_pred)

    metrics = {
        "model_name": name,
        "accuracy":  float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall":    float(recall_score(yt, yp, zero_division=0)),
        "f1":        float(f1_score(yt, yp, zero_division=0)),
        "n_samples": len(yt),
    }
    print_metrics_report(metrics, title=f"{name} — Kết quả đánh giá")

    if save_path:
        clf.save(save_path)
    return clf, metrics


# ============================================================
# 4. Phân tích Threshold Động
# ============================================================
def analyze_dynamic_threshold(mse_train, mse_demo, y_demo, title, save_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # So sánh các chiến lược
    strategies = DynamicThreshold.compare_strategies(
        mse_train,
        k_values=[1.5, 2.0, 2.5, 3.0],
        percentiles=[90, 95, 99]
    )

    print(f"\n📊 Phân tích Threshold Động — {title}")
    print("-" * 45)
    for strategy, thresh in strategies.items():
        # Tính F1 cho từng threshold
        from sklearn.metrics import f1_score
        def to_bin(arr):
            return np.array([0 if str(v).strip().upper() in ("BENIGN", "NORMAL", "0") else 1 for v in arr])
        yt = to_bin(y_demo[:len(mse_demo)])
        yp = (mse_demo > thresh).astype(int)
        f1 = f1_score(yt, yp[:len(yt)], zero_division=0)
        print(f"  {strategy:15s}: threshold={thresh:.6f} | F1={f1:.4f}")

    # Vẽ biểu đồ
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#1c2333")
        ax.tick_params(colors="#e6edf3")
        ax.grid(True, color="#21262d", linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Plot 1: Distribution of MSE
    axes[0].hist(mse_train, bins=50, color="#70a1ff", alpha=0.6, label="Train MSE")
    axes[0].hist(mse_demo,  bins=50, color="#ff6b81", alpha=0.6, label="Demo MSE")
    for strategy, thresh in strategies.items():
        if "2.0std" in strategy or "p95" in strategy:
            axes[0].axvline(thresh, linestyle="--", linewidth=1.2,
                            label=f"{strategy}={thresh:.4f}")
    axes[0].set_title(f"{title} — MSE Distribution", color="#e6edf3", fontsize=11)
    axes[0].set_xlabel("MSE", color="#e6edf3")
    axes[0].legend(fontsize=7, facecolor="#1c2333", labelcolor="#e6edf3")

    # Plot 2: F1 vs Threshold
    thresholds_range = np.linspace(mse_demo.min(), mse_demo.max(), 100)
    from sklearn.metrics import f1_score
    def to_bin(arr):
        return np.array([0 if str(v).strip().upper() in ("BENIGN", "NORMAL", "0") else 1 for v in arr])
    yt = to_bin(y_demo[:len(mse_demo)])
    f1_scores = []
    for t in thresholds_range:
        yp = (mse_demo > t).astype(int)
        f1_scores.append(f1_score(yt, yp[:len(yt)], zero_division=0))

    best_idx = np.argmax(f1_scores)
    axes[1].plot(thresholds_range, f1_scores, color="#2ed573", linewidth=2)
    axes[1].axvline(thresholds_range[best_idx], color="#ffa502", linestyle="--",
                    linewidth=1.5,
                    label=f"Best threshold={thresholds_range[best_idx]:.4f} (F1={f1_scores[best_idx]:.4f})")
    axes[1].set_title(f"{title} — F1-Score vs Threshold", color="#e6edf3", fontsize=11)
    axes[1].set_xlabel("Threshold", color="#e6edf3")
    axes[1].set_ylabel("F1-Score", color="#e6edf3")
    axes[1].legend(fontsize=8, facecolor="#1c2333", labelcolor="#e6edf3")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    💾 Saved → {save_path}")
    plt.close()
    return strategies, best_idx, thresholds_range, f1_scores


# ============================================================
# 5. So sánh tổng hợp
# ============================================================
def comparison_table(metrics_list: list, save_path=None):
    """In và lưu bảng so sánh."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Mô hình": m.get("model_name", "?"),
            "Accuracy":  f"{m.get('accuracy', 0):.4f}",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall":    f"{m.get('recall', 0):.4f}",
            "F1-Score":  f"{m.get('f1', 0):.4f}",
            "AUC-ROC":   f"{m.get('auc_roc', 0):.4f}" if m.get("auc_roc") else "N/A",
            "Loại":      m.get("type", "Unsupervised"),
        })
    df = pd.DataFrame(rows)

    sep = "=" * 80
    print(f"\n{sep}")
    print("  📊 BẢNG SO SÁNH TỔNG HỢP — SUPERVISED vs UNSUPERVISED")
    print(sep)
    print(df.to_string(index=False))
    print(sep)

    if save_path:
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n    💾 Đã lưu bảng so sánh → {save_path}")
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  INITNET IDS — TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    all_metrics = []

    # -------- TRAFFIC --------
    X_train_t, X_demo_t, X_demo_t_raw, y_demo_t, scaler_t = load_traffic_data()

    # Tải nhãn cho supervised (dùng demo_traffic để demo)
    df_demo_full = pd.read_csv(os.path.join(DATA_DIR, "demo_traffic.csv"))
    df_demo_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_demo_full.dropna(inplace=True)

    # 1a. LSTM-AE Traffic (đã có model)
    lstm_metrics_t, lstm_mse_t, y_demo_t_seq, lstm_thresh_t, lstm_train_mse_t = evaluate_lstm_ae(
        model_path=os.path.join(MODEL_DIR, "autoencoder_traffic.h5"),
        X_train=X_train_t, X_demo=X_demo_t, y_demo=y_demo_t,
        name="LSTM Autoencoder (Traffic)",
    )
    lstm_metrics_t["type"] = "Unsupervised"
    all_metrics.append(lstm_metrics_t)

    # 1b. Dense-AE Traffic
    _, dense_metrics_t, dense_mse_t = train_dense_ae(
        X_train=X_train_t, X_demo=X_demo_t, y_demo=y_demo_t,
        name="Dense Autoencoder (Traffic)",
        save_path=os.path.join(MODEL_DIR, "dense_autoencoder_traffic.h5"),
        epochs=25,
    )
    dense_metrics_t["type"] = "Unsupervised"
    all_metrics.append(dense_metrics_t)

    # 1c. Supervised RF Traffic
    # Train RF cần nhãn — dùng dữ liệu demo (split 70/30) cho demo mục đích so sánh
    from sklearn.model_selection import train_test_split
    X_sup_all = df_demo_full.drop(columns=["Label"]).values
    y_sup_all = df_demo_full["Label"].values
    X_sup_train, X_sup_test, y_sup_train, y_sup_test = train_test_split(
        X_sup_all, y_sup_all, test_size=0.4, random_state=42, stratify=y_sup_all
    )
    X_sup_train_s = scaler_t.transform(X_sup_train)
    X_sup_test_s  = scaler_t.transform(X_sup_test)

    rf_clf_t, rf_metrics_t = train_supervised(
        X_train_raw=X_sup_train_s, y_train=y_sup_train,
        X_demo_raw=X_sup_test_s,  y_demo=y_sup_test,
        name="Random Forest (Traffic)",
        save_path=os.path.join(MODEL_DIR, "rf_traffic.pkl"),
    )
    rf_metrics_t["type"] = "Supervised"
    all_metrics.append(rf_metrics_t)

    # Threshold analysis cho Traffic — dùng train_mse thực từ model
    analyze_dynamic_threshold(
        mse_train=lstm_train_mse_t,
        mse_demo=lstm_mse_t,
        y_demo=y_demo_t_seq,
        title="Network Traffic",
        save_path=os.path.join(RESULT_DIR, "threshold_analysis_traffic.png"),
    )

    # -------- LOG --------
    X_train_l, X_demo_l, y_demo_l, vectorizer_l = load_log_data()

    # 2a. LSTM-AE Log
    lstm_metrics_l, lstm_mse_l, y_demo_l_seq, lstm_thresh_l, lstm_train_mse_l = evaluate_lstm_ae(
        model_path=os.path.join(MODEL_DIR, "autoencoder_log.h5"),
        X_train=X_train_l, X_demo=X_demo_l, y_demo=y_demo_l,
        name="LSTM Autoencoder (Log)",
    )
    lstm_metrics_l["type"] = "Unsupervised"
    all_metrics.append(lstm_metrics_l)

    # 2b. Dense-AE Log
    _, dense_metrics_l, dense_mse_l = train_dense_ae(
        X_train=X_train_l, X_demo=X_demo_l, y_demo=y_demo_l,
        name="Dense Autoencoder (Log)",
        save_path=os.path.join(MODEL_DIR, "dense_autoencoder_log.h5"),
        epochs=25,
    )
    dense_metrics_l["type"] = "Unsupervised"
    all_metrics.append(dense_metrics_l)

    # 2c. Supervised RF Log
    X_sup_l_train, X_sup_l_test, y_l_train, y_l_test = train_test_split(
        X_demo_l, y_demo_l, test_size=0.4, random_state=42, stratify=y_demo_l
    )
    rf_clf_l, rf_metrics_l = train_supervised(
        X_train_raw=X_sup_l_train, y_train=y_l_train,
        X_demo_raw=X_sup_l_test,  y_demo=y_l_test,
        name="Random Forest (Log)",
        save_path=os.path.join(MODEL_DIR, "rf_log.pkl"),
    )
    rf_metrics_l["type"] = "Supervised"
    all_metrics.append(rf_metrics_l)

    # Threshold analysis Log — dùng train_mse thực từ model
    analyze_dynamic_threshold(
        mse_train=lstm_train_mse_l,
        mse_demo=lstm_mse_l,
        y_demo=y_demo_l_seq,
        title="System Log",
        save_path=os.path.join(RESULT_DIR, "threshold_analysis_log.png"),
    )

    # -------- BẢNG SO SÁNH --------
    df_comparison = comparison_table(
        all_metrics,
        save_path=os.path.join(RESULT_DIR, "comparison_table.csv"),
    )

    print("\n✅ Training hoàn thành!")
    print(f"   Model mới: {os.path.join(MODEL_DIR, 'dense_autoencoder_traffic.h5')}")
    print(f"   Model mới: {os.path.join(MODEL_DIR, 'dense_autoencoder_log.h5')}")
    print(f"   Kết quả : {RESULT_DIR}/")
