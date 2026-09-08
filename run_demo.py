"""
run_demo.py
-----------
Script demo chính — chạy trước hội đồng:
  1. Khởi động pipeline streaming 3 thread
  2. Hiển thị alert real-time trên terminal
  3. Sau khi xong → vẽ dashboard phân tích đầy đủ
  4. Xuất báo cáo kết quả + metrics

Chạy: python run_demo.py
"""

import os
import sys
import time
import signal
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Tắt log TF

# Xử lý Ctrl+C gọn gàng, tránh core dump của TensorFlow
_pipeline_ref = None
def _signal_handler(sig, frame):
    print("\n\n[!] Đã nhận Ctrl+C — Đang dừng pipeline an toàn...")
    if _pipeline_ref:
        _pipeline_ref.stop()
    os._exit(0)   # Bypass TF C++ cleanup để tránh core dump
signal.signal(signal.SIGINT, _signal_handler)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(PROJECT_ROOT, "datas")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pipeline.streaming_pipeline import StreamingPipeline
from dashboard.realtime_dashboard import RealtimeDashboard
from analysis.metrics import compute_classification_metrics, print_metrics_report
from analysis.dynamic_threshold import DynamicThreshold


# ============================================================
# Banner
# ============================================================
def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════╗
║       INITNET — Intrusion Detection System (IDS)         ║
║    Phát hiện bất thường bằng LSTM Autoencoder            ║
║    Pipeline Streaming Đa Luồng Thời Gian Thực            ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


# ============================================================
# Main
# ============================================================
def main():
    print_banner()

    # Kiểm tra file
    required = {
        "demo_traffic.csv": os.path.join(DATA_DIR,  "demo_traffic.csv"),
        "demo_log.csv":     os.path.join(DATA_DIR,  "demo_log.csv"),
        "autoencoder_traffic.h5": os.path.join(MODEL_DIR, "autoencoder_traffic.h5"),
        "autoencoder_log.h5":     os.path.join(MODEL_DIR, "autoencoder_log.h5"),
        "network_scaler.pkl":     os.path.join(MODEL_DIR, "network_scaler.pkl"),
        "log_vectorizer.pkl":     os.path.join(MODEL_DIR, "log_vectorizer.pkl"),
    }
    for name, path in required.items():
        if not os.path.exists(path):
            print(f"❌ Thiếu file: {name} ({path})")
            sys.exit(1)
    print("✅ Tất cả file cần thiết đã sẵn sàng.\n")

    # ── Dùng queue riêng để biết nguồn nào gửi kết quả
    # (traffic vs log phân biệt bằng thread name trong inference thread)
    dashboard = RealtimeDashboard(window_size=200)

    # Pipeline — không dùng shared callback
    # (mỗi inference thread riêng, thu kết quả sau khi pipeline kết thúc)
    pipeline = StreamingPipeline(
        traffic_csv=required["demo_traffic.csv"],
        log_csv=required["demo_log.csv"],
        model_traffic_path=required["autoencoder_traffic.h5"],
        model_log_path=required["autoencoder_log.h5"],
        scaler_path=required["network_scaler.pkl"],
        vectorizer_path=required["log_vectorizer.pkl"],
        traffic_threshold=0.013591,   # mean+2std từ train
        log_threshold=0.014254,       # mean+2std từ train
        dynamic_threshold=True,
        time_steps=5,
        delay_ms=15.0,                # delay nhỏ để chạy nhanh khi demo
    )

    print("[*] Đang khởi chạy pipeline streaming...\n")
    global _pipeline_ref
    _pipeline_ref = pipeline
    t_start = time.time()
    results = pipeline.run(timeout=600.0)
    elapsed = time.time() - t_start

    # ── Thu thập kết quả ──
    t_results = results.get("traffic_results", [])
    l_results = results.get("log_results", [])
    t_lat_stats = results.get("traffic_latency", {})
    l_lat_stats = results.get("log_latency", {})
    t_mem_stats = results.get("traffic_memory", {})

    print(f"\n[*] Pipeline chạy xong trong {elapsed:.1f}s")

    t_mse = t_lat = t_mem = t_preds = t_labels = None
    l_mse = l_lat = l_preds = l_labels = None

    # ── Metrics Traffic ──
    if t_results:
        t_mse    = np.array([r.mse for r in t_results])
        t_preds  = [r.prediction for r in t_results]
        t_labels = [r.true_label for r in t_results]
        t_lat    = [r.latency_ms for r in t_results]
        t_mem    = [r.memory_mb for r in t_results]

        dt = DynamicThreshold(strategy="mean_std")
        dt.fit(t_mse)
        m = compute_classification_metrics(t_labels, t_preds, t_mse, dt.threshold)
        print_metrics_report(m, title="Network Traffic — Kết quả Pipeline")

        print(f"  ⚡ Latency Traffic: mean={t_lat_stats.get('mean_ms', 0):.2f}ms "
              f"| p95={t_lat_stats.get('p95_ms', 0):.2f}ms "
              f"| max={t_lat_stats.get('max_ms', 0):.2f}ms")
        print(f"  💾 Memory  Traffic: mean={t_mem_stats.get('mean_rss_mb', 0):.1f}MB "
              f"| peak={t_mem_stats.get('max_rss_mb', 0):.1f}MB\n")

    # ── Metrics Log ──
    if l_results:
        l_mse    = np.array([r.mse for r in l_results])
        l_preds  = [r.prediction for r in l_results]
        l_labels = [r.true_label for r in l_results]
        l_lat    = [r.latency_ms for r in l_results]

        m2 = compute_classification_metrics(l_labels, l_preds, l_mse)
        print_metrics_report(m2, title="System Log — Kết quả Pipeline")

        print(f"  ⚡ Latency Log: mean={l_lat_stats.get('mean_ms', 0):.2f}ms "
              f"| p95={l_lat_stats.get('p95_ms', 0):.2f}ms\n")

    # ── Dashboard ──
    print("[*] Đang vẽ dashboard phân tích...")
    import matplotlib
    matplotlib.use("Agg")   # Lưu PNG, không cần display

    dashboard.load_from_results(results)
    save_path = os.path.join(RESULT_DIR, "demo_dashboard.png")
    try:
        dashboard.plot_static(
            traffic_mse=t_mse,
            log_mse=l_mse,
            traffic_threshold=0.013591,
            log_threshold=0.014254,
            traffic_labels=t_preds,
            log_labels=l_preds,
            latency_series=t_lat,
            memory_series=t_mem,
            save_path=save_path,
        )
        print(f"[Dashboard] ✅ Đã lưu → {save_path}")
    except Exception as e:
        print(f"[Dashboard] ⚠️ Lỗi khi vẽ: {e}")

    print(f"\n{'='*55}")
    print(f"  ✅ DEMO HOÀN TẤT | Thời gian: {elapsed:.1f}s")
    print(f"  📁 Kết quả: {RESULT_DIR}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
