"""
log_ingestion.py
----------------
Thread 1: Log Ingestion
Đọc từng dòng log từ file CSV (giả lập streaming),
parse EventSequence → TF-IDF vector → đẩy vào queue.
"""

import threading
import queue
import time
import ast
import numpy as np
import pandas as pd
import joblib
import os


class LogIngestionThread(threading.Thread):
    """
    Thread 1: Đọc log CSV, vectorize và đẩy vào inference queue.
    """

    def __init__(self,
                 log_csv_path: str,
                 vectorizer_path: str,
                 out_queue: queue.Queue,
                 time_steps: int = 5,
                 delay_ms: float = 50.0,
                 name: str = "LogIngestionThread"):
        super().__init__(name=name, daemon=True)
        self.log_csv_path = log_csv_path
        self.vectorizer_path = vectorizer_path
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
    def _parse_sequence(self, val) -> str:
        """Chuyển EventSequence list → chuỗi string cho TF-IDF."""
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return " ".join(str(e) for e in parsed)
        except Exception:
            pass
        return str(val)

    # ------------------------------------------------------------------
    def run(self):
        print(f"[{self.name}] Đang khởi động...")

        # Nạp vectorizer
        if not os.path.exists(self.vectorizer_path):
            print(f"[{self.name}] ❌ Không tìm thấy vectorizer: {self.vectorizer_path}")
            return
        vectorizer = joblib.load(self.vectorizer_path)

        # Đọc file log
        if not os.path.exists(self.log_csv_path):
            print(f"[{self.name}] ❌ Không tìm thấy log file: {self.log_csv_path}")
            return
        df = pd.read_csv(self.log_csv_path)
        df.dropna(inplace=True)

        # Xác định cột EventSequence
        seq_col = "EventSequence" if "EventSequence" in df.columns else df.columns[1]
        # Giữ nhãn nếu có (dùng cho so sánh)
        label_col = "Label" if "Label" in df.columns else None

        print(f"[{self.name}] ✅ Loaded {len(df)} log sequences | Column: '{seq_col}'")

        for idx, row in df.iterrows():
            if self._stop_event.is_set():
                break

            seq_str = self._parse_sequence(row[seq_col])
            label = str(row[label_col]) if label_col else "Unknown"

            # Vectorize (TF-IDF)
            vec = vectorizer.transform([seq_str]).toarray()[0]  # shape (32,)
            self._window.append(vec)

            # Khi đủ time_steps → tạo sequence 3D
            if len(self._window) >= self.time_steps:
                sequence = np.array(self._window[-self.time_steps:])  # (5, 32)
                packet = {
                    "type": "log",
                    "index": idx,
                    "sequence": sequence,
                    "label": label,
                    "timestamp": time.time(),
                }
                self.out_queue.put(packet)
                self._processed += 1

            time.sleep(self.delay_s)

        # Gửi sentinel để báo kết thúc
        self.out_queue.put({"type": "log", "done": True})
        print(f"[{self.name}] ✅ Hoàn thành. Đã xử lý {self._processed} sequences.")
