"""
streaming_pipeline.py
---------------------
Pipeline chính: kết hợp 3 threads
  Thread 1 - LogIngestionThread    : đọc log CSV → queue
  Thread 2 - PacketAnalysisThread  : đọc traffic CSV → queue
  Thread 3 - ModelInferenceThread  : nhận từ queue → inference → alert

Mỗi loại dữ liệu (traffic, log) có queue và inference thread riêng.
"""

import queue
import time
import os
import sys
import numpy as np

# Đường dẫn project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pipeline.log_ingestion import LogIngestionThread
from pipeline.packet_analysis import PacketAnalysisThread
from pipeline.model_inference import ModelInferenceThread


class StreamingPipeline:
    """
    Pipeline phát hiện bất thường đa luồng thời gian thực.

    Sơ đồ:
      demo_log.csv   → LogIngestionThread    ──┐
                                               ├─► log_queue   → InferenceThread (Log)
      demo_traffic.csv → PacketAnalysisThread──┘
                                               ├─► traffic_queue → InferenceThread (Traffic)
    """

    def __init__(self,
                 # Paths
                 traffic_csv: str,
                 log_csv: str,
                 model_traffic_path: str,
                 model_log_path: str,
                 scaler_path: str,
                 vectorizer_path: str,
                 # Thresholds
                 traffic_threshold: float = 0.223306,
                 log_threshold: float = 0.05,
                 dynamic_threshold: bool = True,
                 # Config
                 time_steps: int = 5,
                 delay_ms: float = 30.0,
                 max_samples: int = None,
                 result_callback=None):

        self.traffic_csv = traffic_csv
        self.log_csv = log_csv
        self.traffic_threshold = traffic_threshold
        self.log_threshold = log_threshold
        self.dynamic_threshold = dynamic_threshold
        self.time_steps = time_steps
        self.delay_ms = delay_ms
        self.result_callback = result_callback

        # Load models
        print("[Pipeline] Đang nạp các mô hình...")
        import tensorflow as tf
        self.model_traffic = tf.keras.models.load_model(model_traffic_path, compile=False)
        self.model_traffic.compile(optimizer='adam', loss='mse')
        self.model_log = tf.keras.models.load_model(model_log_path, compile=False)
        self.model_log.compile(optimizer='adam', loss='mse')
        print(f"[Pipeline] ✅ Traffic model: {model_traffic_path}")
        print(f"[Pipeline] ✅ Log model    : {model_log_path}")

        # Queues
        self._traffic_queue = queue.Queue(maxsize=500)
        self._log_queue = queue.Queue(maxsize=500)

        # Threads
        self._threads = {}
        self._setup_threads(scaler_path, vectorizer_path)

    # ------------------------------------------------------------------
    def _setup_threads(self, scaler_path: str, vectorizer_path: str):
        # Thread 1: Log Ingestion
        self._threads["log_ingestion"] = LogIngestionThread(
            log_csv_path=self.log_csv,
            vectorizer_path=vectorizer_path,
            out_queue=self._log_queue,
            time_steps=self.time_steps,
            delay_ms=self.delay_ms,
        )

        # Thread 2: Packet Analysis
        self._threads["packet_analysis"] = PacketAnalysisThread(
            traffic_csv_path=self.traffic_csv,
            scaler_path=scaler_path,
            out_queue=self._traffic_queue,
            time_steps=self.time_steps,
            delay_ms=self.delay_ms,
        )

        # Thread 3a: Inference cho Traffic
        self._threads["inference_traffic"] = ModelInferenceThread(
            model=self.model_traffic,
            in_queue=self._traffic_queue,
            data_type="traffic",
            threshold=self.traffic_threshold,
            dynamic=self.dynamic_threshold,
            result_callback=self.result_callback,
            name="InferenceThread-Traffic",
        )

        # Thread 3b: Inference cho Log
        self._threads["inference_log"] = ModelInferenceThread(
            model=self.model_log,
            in_queue=self._log_queue,
            data_type="log",
            threshold=self.log_threshold,
            dynamic=self.dynamic_threshold,
            result_callback=self.result_callback,
            name="InferenceThread-Log",
        )

    # ------------------------------------------------------------------
    def run(self, timeout: float = 300.0):
        """
        Khởi chạy toàn bộ pipeline.

        Parameters
        ----------
        timeout : thời gian tối đa (giây) trước khi dừng
        """
        print("\n" + "=" * 60)
        print("  🚀 INITNET IDS — STREAMING PIPELINE BẮT ĐẦU")
        print("=" * 60)

        start_time = time.time()

        # Khởi động tất cả threads
        for name, thread in self._threads.items():
            thread.start()
            print(f"[Pipeline] Thread '{name}' đã khởi động.")

        # Chờ các ingestion thread hoàn thành
        self._threads["log_ingestion"].join(timeout=timeout)
        self._threads["packet_analysis"].join(timeout=timeout)

        # Chờ inference threads xử lý hết queue
        self._threads["inference_traffic"].join(timeout=60.0)
        self._threads["inference_log"].join(timeout=60.0)

        elapsed = time.time() - start_time

        print(f"\n[Pipeline] ✅ Pipeline kết thúc sau {elapsed:.1f}s")
        self._print_pipeline_summary()

        return self._collect_results()

    # ------------------------------------------------------------------
    def _print_pipeline_summary(self):
        sep = "=" * 60
        print(f"\n{sep}")
        print("  📊 TỔNG KẾT PIPELINE")
        print(sep)
        it = self._threads["inference_traffic"]
        il = self._threads["inference_log"]
        print(f"  Traffic — Tổng: {it.alert_count + it.normal_count} | "
              f"🚨 Alert: {it.alert_count} | ✅ Normal: {it.normal_count}")
        print(f"  Log     — Tổng: {il.alert_count + il.normal_count} | "
              f"🚨 Alert: {il.alert_count} | ✅ Normal: {il.normal_count}")
        print(sep + "\n")

    def _collect_results(self) -> dict:
        return {
            "traffic_results": self._threads["inference_traffic"].results,
            "log_results": self._threads["inference_log"].results,
            "traffic_latency": self._threads["inference_traffic"].latency_tracker.stats(),
            "log_latency": self._threads["inference_log"].latency_tracker.stats(),
            "traffic_memory": self._threads["inference_traffic"].memory_monitor.stats(),
            "log_memory": self._threads["inference_log"].memory_monitor.stats(),
        }

    # ------------------------------------------------------------------
    def stop(self):
        for thread in self._threads.values():
            if hasattr(thread, "stop"):
                thread.stop()
