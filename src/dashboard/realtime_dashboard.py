"""
realtime_dashboard.py
---------------------
Dashboard matplotlib real-time hiển thị:
  - Reconstruction error theo thời gian (traffic & log)
  - Ngưỡng động
  - Latency theo thời gian
  - Memory usage
  - Alert count bar

Chạy được cả trong chế độ demo (animation từ kết quả pipeline)
và chế độ live (callback từ inference thread).
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # fallback nếu không có display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time


class RealtimeDashboard:
    """
    Dashboard thời gian thực cho INITNET IDS.
    Hiển thị reconstruction error, threshold động,
    latency, memory và bộ đếm cảnh báo.
    """

    def __init__(self, window_size: int = 100, title: str = "INITNET IDS — Real-time Monitor"):
        self.window_size = window_size
        self.title = title
        self._lock = threading.Lock()

        # Buffers cho traffic
        self._traffic_mse = deque(maxlen=window_size)
        self._traffic_threshold = deque(maxlen=window_size)
        self._traffic_labels = deque(maxlen=window_size)
        self._traffic_latency = deque(maxlen=window_size)

        # Buffers cho log
        self._log_mse = deque(maxlen=window_size)
        self._log_threshold = deque(maxlen=window_size)
        self._log_labels = deque(maxlen=window_size)
        self._log_latency = deque(maxlen=window_size)

        # Memory
        self._memory_rss = deque(maxlen=window_size)
        self._memory_time = deque(maxlen=window_size)

        # Counts
        self._traffic_alerts = 0
        self._traffic_normals = 0
        self._log_alerts = 0
        self._log_normals = 0

        self._fig = None
        self._axes = {}
        self._anim = None

    # ------------------------------------------------------------------
    # Callback để nhận dữ liệu từ inference thread
    # ------------------------------------------------------------------
    def on_traffic_result(self, record, threshold: float):
        with self._lock:
            self._traffic_mse.append(record.mse)
            self._traffic_threshold.append(threshold)
            self._traffic_latency.append(record.latency_ms)
            self._traffic_labels.append(record.prediction)
            self._memory_rss.append(record.memory_mb)
            self._memory_time.append(time.time())
            if record.prediction == "ANOMALY":
                self._traffic_alerts += 1
            else:
                self._traffic_normals += 1

    def on_log_result(self, record, threshold: float):
        with self._lock:
            self._log_mse.append(record.mse)
            self._log_threshold.append(threshold)
            self._log_latency.append(record.latency_ms)
            self._log_labels.append(record.prediction)
            if record.prediction == "ANOMALY":
                self._log_alerts += 1
            else:
                self._log_normals += 1

    # ------------------------------------------------------------------
    # Nạp kết quả batch (sau khi pipeline xong)
    # ------------------------------------------------------------------
    def load_from_results(self, pipeline_results: dict):
        """Nạp kết quả từ pipeline để vẽ đồ thị offline."""
        for r in pipeline_results.get("traffic_results", []):
            self._traffic_mse.append(r.mse)
            self._traffic_latency.append(r.latency_ms)
            self._traffic_labels.append(r.prediction)
            self._memory_rss.append(r.memory_mb)
            self._memory_time.append(r.index)
            if r.prediction == "ANOMALY":
                self._traffic_alerts += 1
            else:
                self._traffic_normals += 1

        for r in pipeline_results.get("log_results", []):
            self._log_mse.append(r.mse)
            self._log_latency.append(r.latency_ms)
            self._log_labels.append(r.prediction)
            if r.prediction == "ANOMALY":
                self._log_alerts += 1
            else:
                self._log_normals += 1

    # ------------------------------------------------------------------
    # Vẽ đồ thị tĩnh (sau khi chạy xong pipeline)
    # ------------------------------------------------------------------
    def plot_static(self,
                    traffic_mse: np.ndarray = None,
                    log_mse: np.ndarray = None,
                    traffic_threshold: float = None,
                    log_threshold: float = None,
                    traffic_labels: list = None,
                    log_labels: list = None,
                    latency_series: list = None,
                    memory_series: list = None,
                    save_path: str = None):
        """
        Vẽ đồ thị phân tích đầy đủ (8 subplot).
        """
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor("#0d1117")
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        ax_colors = "#1c2333"
        text_color = "#e6edf3"
        alert_color = "#ff4757"
        normal_color = "#2ed573"
        threshold_color = "#ffa502"

        def style_ax(ax, title):
            ax.set_facecolor(ax_colors)
            ax.set_title(title, color=text_color, fontsize=10, fontweight="bold")
            ax.tick_params(colors=text_color, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.grid(True, color="#21262d", linestyle="--", alpha=0.6)

        # --- 1. Traffic MSE ---
        ax1 = fig.add_subplot(gs[0, :2])
        style_ax(ax1, "🌐 Network Traffic — Reconstruction Error (MSE)")
        if traffic_mse is not None:
            x = np.arange(len(traffic_mse))
            colors = []
            for i, mse in enumerate(traffic_mse):
                lbl = traffic_labels[i] if traffic_labels else "NORMAL"
                colors.append(alert_color if lbl == "ANOMALY" else normal_color)
            ax1.bar(x, traffic_mse, color=colors, alpha=0.7, width=1.0)
            if traffic_threshold:
                ax1.axhline(traffic_threshold, color=threshold_color,
                            linewidth=1.5, linestyle="--", label=f"Threshold={traffic_threshold:.4f}")
            ax1.set_xlabel("Sample", color=text_color, fontsize=8)
            ax1.set_ylabel("MSE", color=text_color, fontsize=8)
            ax1.legend(fontsize=7, facecolor=ax_colors, labelcolor=text_color)

        # --- 2. Log MSE ---
        ax2 = fig.add_subplot(gs[1, :2])
        style_ax(ax2, "📋 System Log — Reconstruction Error (MSE)")
        if log_mse is not None:
            x = np.arange(len(log_mse))
            log_thresh = log_threshold or (np.mean(log_mse) + 2 * np.std(log_mse))
            colors = []
            for i, mse in enumerate(log_mse):
                lbl = log_labels[i] if log_labels else "NORMAL"
                colors.append(alert_color if lbl == "ANOMALY" else normal_color)
            ax2.bar(x, log_mse, color=colors, alpha=0.7, width=1.0)
            ax2.axhline(log_thresh, color=threshold_color,
                        linewidth=1.5, linestyle="--", label=f"Threshold={log_thresh:.4f}")
            ax2.set_xlabel("Sample", color=text_color, fontsize=8)
            ax2.set_ylabel("MSE", color=text_color, fontsize=8)
            ax2.legend(fontsize=7, facecolor=ax_colors, labelcolor=text_color)

        # --- 3. Alert counts (Traffic) ---
        ax3 = fig.add_subplot(gs[0, 2])
        style_ax(ax3, "🚨 Traffic Alert Summary")
        t_alert = self._traffic_alerts
        t_normal = self._traffic_normals
        if t_alert + t_normal == 0 and traffic_labels:
            t_alert = sum(1 for l in traffic_labels if l == "ANOMALY")
            t_normal = sum(1 for l in traffic_labels if l == "NORMAL")
        bars = ax3.bar(["Normal", "Alert"], [t_normal, t_alert],
                       color=[normal_color, alert_color], alpha=0.85)
        for bar, val in zip(bars, [t_normal, t_alert]):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", color=text_color, fontsize=9, fontweight="bold")
        ax3.set_ylabel("Count", color=text_color, fontsize=8)

        # --- 4. Alert counts (Log) ---
        ax4 = fig.add_subplot(gs[1, 2])
        style_ax(ax4, "🚨 Log Alert Summary")
        l_alert = self._log_alerts
        l_normal = self._log_normals
        if l_alert + l_normal == 0 and log_labels:
            l_alert = sum(1 for l in log_labels if l == "ANOMALY")
            l_normal = sum(1 for l in log_labels if l == "NORMAL")
        bars2 = ax4.bar(["Normal", "Alert"], [l_normal, l_alert],
                        color=[normal_color, alert_color], alpha=0.85)
        for bar, val in zip(bars2, [l_normal, l_alert]):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", color=text_color, fontsize=9, fontweight="bold")
        ax4.set_ylabel("Count", color=text_color, fontsize=8)

        # --- 5. Latency ---
        ax5 = fig.add_subplot(gs[2, :2])
        style_ax(ax5, "⚡ Inference Latency (ms)")
        if latency_series:
            ax5.plot(latency_series, color="#70a1ff", linewidth=1.2, alpha=0.85)
            mean_lat = np.mean(latency_series)
            ax5.axhline(mean_lat, color=threshold_color, linestyle="--",
                        linewidth=1.2, label=f"Mean={mean_lat:.2f}ms")
            ax5.fill_between(range(len(latency_series)), latency_series,
                             alpha=0.2, color="#70a1ff")
            ax5.set_xlabel("Sample", color=text_color, fontsize=8)
            ax5.set_ylabel("ms", color=text_color, fontsize=8)
            ax5.legend(fontsize=7, facecolor=ax_colors, labelcolor=text_color)

        # --- 6. Memory ---
        ax6 = fig.add_subplot(gs[2, 2])
        style_ax(ax6, "💾 Memory Usage (MB)")
        if memory_series:
            ax6.plot(memory_series, color="#eccc68", linewidth=1.5, alpha=0.9)
            ax6.fill_between(range(len(memory_series)), memory_series,
                             alpha=0.25, color="#eccc68")
            ax6.set_xlabel("Sample", color=text_color, fontsize=8)
            ax6.set_ylabel("MB", color=text_color, fontsize=8)

        # Title
        fig.suptitle(self.title, color=text_color,
                     fontsize=14, fontweight="bold", y=1.01)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[Dashboard] 💾 Đã lưu biểu đồ → {save_path}")

        plt.show()
        return fig
