"""
metrics.py
----------
Đo lường hiệu năng hệ thống IDS:
  - Latency (ms) của từng giai đoạn inference
  - Memory usage (MB) theo thời gian thực
  - Các chỉ số phân loại: Accuracy, Precision, Recall, F1, AUC-ROC
"""

import time
import psutil
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LatencyRecord:
    timestamp: float
    latency_ms: float
    label: str = "inference"


@dataclass
class MemoryRecord:
    timestamp: float
    rss_mb: float       # Resident Set Size
    vms_mb: float       # Virtual Memory Size


@dataclass
class InferenceMetrics:
    """Tổng hợp một lần inference."""
    index: int
    mse: float
    latency_ms: float
    memory_mb: float
    prediction: str     # "ALERT" | "NORMAL"
    true_label: str = ""


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------
class LatencyTracker:
    """Đo latency của các bước xử lý."""

    def __init__(self):
        self._records: List[LatencyRecord] = []
        self._start: Optional[float] = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self, label: str = "inference") -> float:
        if self._start is None:
            return 0.0
        latency_ms = (time.perf_counter() - self._start) * 1000
        self._records.append(LatencyRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            label=label
        ))
        self._start = None
        return latency_ms

    def stats(self) -> dict:
        if not self._records:
            return {}
        lats = [r.latency_ms for r in self._records]
        return {
            "count": len(lats),
            "mean_ms": float(np.mean(lats)),
            "median_ms": float(np.median(lats)),
            "min_ms": float(np.min(lats)),
            "max_ms": float(np.max(lats)),
            "p95_ms": float(np.percentile(lats, 95)),
            "p99_ms": float(np.percentile(lats, 99)),
        }

    def get_series(self):
        ts = [r.timestamp for r in self._records]
        lats = [r.latency_ms for r in self._records]
        return ts, lats


# ---------------------------------------------------------------------------
# Memory monitor
# ---------------------------------------------------------------------------
class MemoryMonitor:
    """Theo dõi memory của process hiện tại."""

    def __init__(self):
        self._proc = psutil.Process(os.getpid())
        self._records: List[MemoryRecord] = []

    def sample(self) -> float:
        """Lấy mẫu ngay lập tức, trả về RSS(MB)."""
        info = self._proc.memory_info()
        rss_mb = info.rss / (1024 ** 2)
        vms_mb = info.vms / (1024 ** 2)
        self._records.append(MemoryRecord(
            timestamp=time.time(),
            rss_mb=rss_mb,
            vms_mb=vms_mb
        ))
        return rss_mb

    def stats(self) -> dict:
        if not self._records:
            return {}
        rss = [r.rss_mb for r in self._records]
        return {
            "count": len(rss),
            "mean_rss_mb": float(np.mean(rss)),
            "max_rss_mb": float(np.max(rss)),
            "min_rss_mb": float(np.min(rss)),
        }

    def get_series(self):
        ts = [r.timestamp for r in self._records]
        rss = [r.rss_mb for r in self._records]
        return ts, rss


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   mse_scores: np.ndarray = None,
                                   threshold: float = None) -> dict:
    """
    Tính toán đầy đủ các chỉ số phân loại.

    Parameters
    ----------
    y_true      : nhãn thật (0=Normal, 1=Anomaly hoặc string)
    y_pred      : nhãn dự đoán (0/1 hoặc string)
    mse_scores  : mảng MSE scores (để tính AUC-ROC)
    threshold   : ngưỡng đang dùng

    Returns
    -------
    dict chứa accuracy, precision, recall, f1, auc_roc, confusion_matrix
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    # Chuẩn hóa về binary
    def to_binary(arr):
        """0 = Normal (BENIGN/NORMAL), 1 = Attack/Anomaly (anything else)."""
        result = []
        for v in arr:
            sv = str(v).strip().upper()
            # Chỉ những nhãn này mới là Normal
            if sv in ("0", "NORMAL", "BENIGN"):
                result.append(0)
            else:
                result.append(1)
        return np.array(result)

    yt = to_binary(y_true)
    yp = to_binary(y_pred)

    metrics = {
        "accuracy":  float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall":    float(recall_score(yt, yp, zero_division=0)),
        "f1":        float(f1_score(yt, yp, zero_division=0)),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        "threshold": threshold,
        "n_samples": len(yt),
        "n_anomaly_true": int(yt.sum()),
        "n_anomaly_pred": int(yp.sum()),
    }

    if mse_scores is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(yt, mse_scores))
        except Exception:
            metrics["auc_roc"] = None

    return metrics


def print_metrics_report(metrics: dict, title: str = "Metrics Report"):
    """In báo cáo chỉ số ra console theo định dạng đẹp."""
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Accuracy  : {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision : {metrics.get('precision', 0):.4f}")
    print(f"  Recall    : {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score  : {metrics.get('f1', 0):.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC   : {metrics.get('auc_roc', 0):.4f}")
    if metrics.get("threshold") is not None:
        print(f"  Threshold : {metrics.get('threshold'):.6f}")
    print(f"  Samples   : {metrics.get('n_samples', 0)}")
    cm = metrics.get("confusion_matrix")
    if cm:
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
        print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print(sep + "\n")
