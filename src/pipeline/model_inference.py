"""
model_inference.py
------------------
Thread 3: Model Inference
Nhận sequence từ queue, chạy LSTM Autoencoder,
tính reconstruction error, áp ngưỡng động, báo cáo kết quả.
Đồng thời đo latency và memory usage.
"""

import threading
import queue
import time
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.dynamic_threshold import DynamicThreshold
from analysis.metrics import LatencyTracker, MemoryMonitor, InferenceMetrics


class ModelInferenceThread(threading.Thread):
    """
    Thread 3: Nhận data từ queue → inference → quyết định báo động.
    """

    def __init__(self,
                 model,              # Keras model đã load
                 in_queue: queue.Queue,
                 data_type: str,    # "traffic" | "log"
                 threshold: float,
                 dynamic: bool = True,
                 result_callback=None,
                 name: str = "ModelInferenceThread"):
        super().__init__(name=name, daemon=True)
        self._model = model
        self._in_queue = in_queue
        self._data_type = data_type
        self._threshold = threshold
        self._dynamic = dynamic
        self._result_callback = result_callback
        self._stop_event = threading.Event()

        # Thống kê
        self.latency_tracker = LatencyTracker()
        self.memory_monitor = MemoryMonitor()
        self.results: list = []
        self._alert_count = 0
        self._normal_count = 0

        # Threshold động
        self._dyn_threshold = DynamicThreshold(strategy="sliding", k=2.0, window_size=50)
        self._dyn_threshold.threshold = threshold  # khởi tạo từ giá trị train

        # Sentinel counter
        self._done_count = 0
        self._expected_done = 1  # mặc định 1 nguồn

    def set_expected_sources(self, n: int):
        """Số lượng nguồn data (để biết khi nào kết thúc)."""
        self._expected_done = n

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    def run(self):
        print(f"[{self.name}] 🔴 Hệ thống sẵn sàng | Threshold ban đầu: {self._threshold:.6f}")

        while not self._stop_event.is_set():
            try:
                packet = self._in_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            # Sentinel – kết thúc stream
            if packet.get("done"):
                self._done_count += 1
                if self._done_count >= self._expected_done:
                    break
                continue

            sequence = packet["sequence"]       # np.ndarray
            label = packet.get("label", "?")
            idx = packet.get("index", -1)

            # --- Inference ---
            self.latency_tracker.start()
            mem_mb = self.memory_monitor.sample()

            lstm_input = sequence[np.newaxis, ...]  # (1, 5, dim)
            try:
                pred = self._model.predict(lstm_input, verbose=0)
                mse = float(np.mean(np.power(lstm_input - pred, 2)))
            except Exception as e:
                print(f"[{self.name}] ⚠️ Inference error: {e}")
                self._in_queue.task_done()
                continue

            latency_ms = self.latency_tracker.stop("inference")

            # --- Cập nhật threshold động ---
            if self._dynamic:
                current_threshold = self._dyn_threshold.update(mse)
            else:
                current_threshold = self._threshold

            # --- Phán quyết ---
            is_anomaly = mse > current_threshold
            status = "🚨 BÁO ĐỘNG" if is_anomaly else "✅ Normal"

            if is_anomaly:
                self._alert_count += 1
                print(f"[{self.name}] {status} | idx={idx:5d} | "
                      f"MSE={mse:.6f} | Threshold={current_threshold:.6f} | "
                      f"Latency={latency_ms:.2f}ms | Nhãn={label}")
            else:
                self._normal_count += 1

            # Lưu kết quả
            record = InferenceMetrics(
                index=idx,
                mse=mse,
                latency_ms=latency_ms,
                memory_mb=mem_mb,
                prediction="ANOMALY" if is_anomaly else "NORMAL",
                true_label=label,
            )
            self.results.append(record)

            # Callback (dùng cho dashboard)
            if self._result_callback:
                self._result_callback(record, current_threshold)

            self._in_queue.task_done()

        self._print_summary()

    # ------------------------------------------------------------------
    def _print_summary(self):
        total = self._alert_count + self._normal_count
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  [{self.name}] KẾT QUẢ TỔNG KẾT")
        print(sep)
        print(f"  Tổng số mẫu xử lý : {total}")
        print(f"  🚨 Bất thường      : {self._alert_count} ({self._alert_count/max(total,1)*100:.1f}%)")
        print(f"  ✅ Bình thường     : {self._normal_count} ({self._normal_count/max(total,1)*100:.1f}%)")
        lat = self.latency_tracker.stats()
        if lat:
            print(f"  Latency trung bình : {lat['mean_ms']:.2f} ms")
            print(f"  Latency P95        : {lat['p95_ms']:.2f} ms")
            print(f"  Latency max        : {lat['max_ms']:.2f} ms")
        mem = self.memory_monitor.stats()
        if mem:
            print(f"  Memory trung bình  : {mem['mean_rss_mb']:.1f} MB")
            print(f"  Memory đỉnh        : {mem['max_rss_mb']:.1f} MB")
        print(sep + "\n")

    # ------------------------------------------------------------------
    @property
    def alert_count(self) -> int:
        return self._alert_count

    @property
    def normal_count(self) -> int:
        return self._normal_count
