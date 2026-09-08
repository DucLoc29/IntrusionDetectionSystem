"""
comparison.py
-------------
So sánh Supervised vs Unsupervised cho bài toán IDS.

Supervised  : Random Forest Classifier (train với nhãn)
Unsupervised: LSTM Autoencoder (train không nhãn, dùng reconstruction error)

Xuất bảng so sánh đầy đủ các chỉ số.
"""

import numpy as np
import pandas as pd
import joblib
import ast
import os
import time
import psutil


# ---------------------------------------------------------------------------
# Supervised: Random Forest
# ---------------------------------------------------------------------------
class SupervisedClassifier:
    """Wrapper Random Forest để so sánh với Autoencoder."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.label_encoder = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Trả về 0/1 (0=Normal, 1=Anomaly)."""
        preds = self.predict(X)
        return np.array([0 if str(p).upper() in ("BENIGN", "NORMAL", "0") else 1
                         for p in preds])

    def save(self, path: str):
        joblib.dump({"model": self.model, "encoder": self.label_encoder}, path)

    def load(self, path: str):
        obj = joblib.load(path)
        self.model = obj["model"]
        self.label_encoder = obj["encoder"]
        return self


# ---------------------------------------------------------------------------
# So sánh tổng hợp
# ---------------------------------------------------------------------------
def run_comparison(
    X_demo: np.ndarray,
    y_true: np.ndarray,
    unsupervised_mse: np.ndarray,
    unsupervised_threshold: float,
    supervised_model: "SupervisedClassifier",
    X_demo_raw_for_supervised: np.ndarray = None
) -> pd.DataFrame:
    """
    So sánh supervised vs unsupervised.

    Returns
    -------
    pd.DataFrame: bảng so sánh
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)

    def to_binary(arr):
        """0 = Normal (BENIGN/NORMAL), 1 = Attack/Anomaly (anything else)."""
        return np.array([
            0 if str(v).strip().upper() in ("BENIGN", "NORMAL", "0") else 1
            for v in arr
        ])

    yt = to_binary(y_true)

    rows = []

    # --- Unsupervised ---
    y_unsup = (unsupervised_mse > unsupervised_threshold).astype(int)
    rows.append({
        "Phương pháp": "Unsupervised (LSTM-AE)",
        "Accuracy":    accuracy_score(yt, y_unsup),
        "Precision":   precision_score(yt, y_unsup, zero_division=0),
        "Recall":      recall_score(yt, y_unsup, zero_division=0),
        "F1-Score":    f1_score(yt, y_unsup, zero_division=0),
        "AUC-ROC":     roc_auc_score(yt, unsupervised_mse) if len(np.unique(yt)) > 1 else None,
        "Cần nhãn train": "❌ Không",
        "Phát hiện unknown": "✅ Có",
    })

    # --- Supervised ---
    X_sup = X_demo_raw_for_supervised if X_demo_raw_for_supervised is not None else X_demo
    y_sup = supervised_model.predict_binary(X_sup)
    rows.append({
        "Phương pháp": "Supervised (Random Forest)",
        "Accuracy":    accuracy_score(yt, y_sup),
        "Precision":   precision_score(yt, y_sup, zero_division=0),
        "Recall":      recall_score(yt, y_sup, zero_division=0),
        "F1-Score":    f1_score(yt, y_sup, zero_division=0),
        "AUC-ROC":     roc_auc_score(yt, supervised_model.model.predict_proba(X_sup)[:, 1])
                       if len(np.unique(yt)) > 1 else None,
        "Cần nhãn train": "✅ Có",
        "Phát hiện unknown": "❌ Không",
    })

    df = pd.DataFrame(rows)
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")

    return df


def print_comparison_table(df: pd.DataFrame, title: str = "So sánh Supervised vs Unsupervised"):
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(df.to_string(index=False))
    print(sep + "\n")
