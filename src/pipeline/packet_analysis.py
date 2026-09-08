"""
packet_analysis.py
------------------
Thread 2: Packet Analysis
Đọc từng dòng network traffic từ file CSV (giả lập streaming),
chuẩn hóa → sliding window → đẩy vào queue.
"""

import threading
import queue
import time
import numpy as np
import pandas as pd
import joblib
import os


class PacketAnalysisThread(threading.Thread):
    """
    Thread 2: Đọc network traffic CSV, scale và tạo LSTM sequence.
    """

    def __init__(self,
                 traffic_csv_path: str,
                 scaler_path: str,
                 out_queue: queue.Queue,
                 time_steps: int = 5,
                 delay_ms: float = 50.0,
                 name: str = "PacketAnalysisThread"):
        super().__init__(name=name, daemon=True)
        self.traffic_csv_path = traffic_csv_path
        self.scaler_path = scaler_path
        self.out_queue = out_queue
        self.time_steps = time_steps
        self.delay_s = delay_ms / 1000.0
        self._stop_event = threading.Event()
        self._processed = 0
        self._window = []   # sliding window buffer

    # ------------------------------------------------------------------
    def stop(self):
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def processed(self) -> int:
        return self._processed

    # ------------------------------------------------------------------
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loại bỏ nhãn, làm sạch NaN/Inf."""
        label_col = "Label" if "Label" in df.columns else None
        labels = df[label_col].values if label_col else None

        feat_df = df.drop(columns=[label_col]) if label_col else df.copy()
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df.dropna(inplace=True)

        return feat_df, labels, label_col

    # ------------------------------------------------------------------
    def run(self):
        print(f"[{self.name}] Đang khởi động...")

        # Nạp scaler
        if not os.path.exists(self.scaler_path):
            print(f"[{self.name}] ❌ Không tìm thấy scaler: {self.scaler_path}")
            return
        scaler = joblib.load(self.scaler_path)

        # Đọc file traffic
        if not os.path.exists(self.traffic_csv_path):
            print(f"[{self.name}] ❌ Không tìm thấy traffic file: {self.traffic_csv_path}")
            return
        df_raw = pd.read_csv(self.traffic_csv_path)
        feat_df, labels, label_col = self._clean_features(df_raw)

        print(f"[{self.name}] ✅ Loaded {len(feat_df)} packets | Features: {feat_df.shape[1]}")

        for i, (idx, row) in enumerate(feat_df.iterrows()):
            if self._stop_event.is_set():
                break

            raw_vec = row.values.reshape(1, -1).astype(float)

            # Chuẩn hóa
            try:
                scaled_vec = scaler.transform(raw_vec)[0]
            except Exception as e:
                print(f"[{self.name}] Scaler error at row {idx}: {e}")
                continue

            self._window.append(scaled_vec)
            label = str(labels[i]) if labels is not None else "Unknown"

            # Khi đủ time_steps → tạo sequence 3D
            if len(self._window) >= self.time_steps:
                sequence = np.array(self._window[-self.time_steps:])  # (5, 78)
                packet = {
                    "type": "traffic",
                    "index": idx,
                    "sequence": sequence,
                    "label": label,
                    "timestamp": time.time(),
                }
                self.out_queue.put(packet)
                self._processed += 1

            time.sleep(self.delay_s)

        # Gửi sentinel
        self.out_queue.put({"type": "traffic", "done": True})
        print(f"[{self.name}] ✅ Hoàn thành. Đã xử lý {self._processed} packets.")
