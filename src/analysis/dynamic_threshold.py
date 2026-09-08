"""
dynamic_threshold.py
--------------------
Phân tích và tính ngưỡng động (dynamic threshold) dựa trên
reconstruction error của Autoencoder.

Các chiến lược hỗ trợ:
  1. mean_std   : threshold = mean + k * std  (mặc định, k=2)
  2. percentile : threshold = percentile(errors, q)
  3. sliding    : cửa sổ trượt trên luồng thời gian thực
"""

import numpy as np
from collections import deque


class DynamicThreshold:
    """
    Tính và cập nhật ngưỡng phát hiện bất thường theo thời gian thực.
    """

    def __init__(self, strategy: str = "mean_std", k: float = 2.0,
                 percentile: float = 95.0, window_size: int = 100):
        """
        Parameters
        ----------
        strategy    : 'mean_std' | 'percentile' | 'sliding'
        k           : hệ số nhân std (dùng cho mean_std)
        percentile  : phân vị (dùng cho percentile strategy)
        window_size : kích thước cửa sổ (dùng cho sliding strategy)
        """
        if strategy not in ("mean_std", "percentile", "sliding"):
            raise ValueError(f"strategy phải là 'mean_std', 'percentile', hoặc 'sliding'. Nhận: {strategy}")

        self.strategy = strategy
        self.k = k
        self.percentile = percentile
        self.window_size = window_size

        # Cửa sổ trượt cho streaming
        self._window: deque = deque(maxlen=window_size)

        # Giá trị ngưỡng hiện tại
        self.threshold: float = float("inf")
        self._errors_history: list = []

    # ------------------------------------------------------------------
    # Tính ngưỡng từ tập dữ liệu tĩnh (batch)
    # ------------------------------------------------------------------
    def fit(self, errors: np.ndarray) -> float:
        """
        Tính threshold từ mảng lỗi reconstruction của tập train/demo.

        Returns
        -------
        float : giá trị ngưỡng được tính
        """
        errors = np.array(errors, dtype=float).flatten()
        self._errors_history = errors.tolist()

        if self.strategy == "mean_std":
            self.threshold = float(np.mean(errors) + self.k * np.std(errors))
        elif self.strategy == "percentile":
            self.threshold = float(np.percentile(errors, self.percentile))
        elif self.strategy == "sliding":
            # Khởi tạo window từ batch
            self._window.extend(errors.tolist())
            self.threshold = self._compute_sliding()

        return self.threshold

    # ------------------------------------------------------------------
    # Cập nhật online (streaming)
    # ------------------------------------------------------------------
    def update(self, new_error: float) -> float:
        """
        Cập nhật ngưỡng khi nhận thêm một điểm lỗi mới (streaming).
        Chỉ có hiệu lực khi strategy='sliding'.

        Returns
        -------
        float : threshold hiện tại
        """
        self._window.append(new_error)
        if self.strategy == "sliding" and len(self._window) >= 10:
            self.threshold = self._compute_sliding()
        return self.threshold

    def _compute_sliding(self) -> float:
        arr = np.array(list(self._window), dtype=float)
        return float(np.mean(arr) + self.k * np.std(arr))

    # ------------------------------------------------------------------
    # Phán quyết
    # ------------------------------------------------------------------
    def predict(self, error: float) -> bool:
        """Trả về True nếu error vượt ngưỡng (bất thường)."""
        return error > self.threshold

    # ------------------------------------------------------------------
    # Thống kê so sánh nhiều chiến lược
    # ------------------------------------------------------------------
    @staticmethod
    def compare_strategies(errors: np.ndarray,
                           k_values: list = None,
                           percentiles: list = None) -> dict:
        """
        So sánh các giá trị ngưỡng theo nhiều tham số.

        Returns
        -------
        dict : {'strategy_name': threshold_value}
        """
        if k_values is None:
            k_values = [1.5, 2.0, 2.5, 3.0]
        if percentiles is None:
            percentiles = [90, 95, 99]

        results = {}
        errors = np.array(errors, dtype=float).flatten()
        mu, sigma = np.mean(errors), np.std(errors)

        for k in k_values:
            results[f"mean+{k}std"] = float(mu + k * sigma)

        for q in percentiles:
            results[f"p{q}"] = float(np.percentile(errors, q))

        return results

    # ------------------------------------------------------------------
    # Lấy lịch sử để vẽ đồ thị
    # ------------------------------------------------------------------
    def get_history(self) -> np.ndarray:
        return np.array(self._errors_history)

    def __repr__(self) -> str:
        return (f"DynamicThreshold(strategy='{self.strategy}', "
                f"k={self.k}, threshold={self.threshold:.6f})")
