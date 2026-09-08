"""
dense_autoencoder.py
--------------------
Dense (Standard) Autoencoder cho dữ liệu tabular.
Dùng để so sánh với LSTM Autoencoder.

Hỗ trợ cả Traffic và Log.
"""

import numpy as np
import os


def build_dense_autoencoder(input_dim: int,
                             encoding_dim: int = 16,
                             hidden_dim: int = 32):
    """
    Xây dựng Dense Autoencoder với kiến trúc:
      input_dim → hidden_dim → encoding_dim → hidden_dim → input_dim

    Parameters
    ----------
    input_dim    : số chiều đầu vào
    encoding_dim : kích thước bottleneck
    hidden_dim   : kích thước lớp ẩn
    """
    import tensorflow as tf

    inp = tf.keras.Input(shape=(input_dim,), name="input")

    # Encoder
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="enc_1")(inp)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Dense(encoding_dim, activation="relu", name="enc_2")(x)

    # Decoder
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dec_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    out = tf.keras.layers.Dense(input_dim, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inp, out, name="DenseAutoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse")
    return model


def build_lstm_autoencoder(time_steps: int, input_dim: int,
                            latent_dim: int = 32):
    """
    LSTM Autoencoder (để tham chiếu lại kiến trúc cũ).
    """
    import tensorflow as tf

    inp = tf.keras.Input(shape=(time_steps, input_dim), name="input")
    enc = tf.keras.layers.LSTM(latent_dim, activation="relu",
                                return_sequences=False, name="lstm_enc")(inp)
    rep = tf.keras.layers.RepeatVector(time_steps, name="repeat")(enc)
    dec = tf.keras.layers.LSTM(latent_dim, activation="relu",
                                return_sequences=True, name="lstm_dec")(rep)
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(input_dim), name="output")(dec)

    model = tf.keras.Model(inp, out, name="LSTMAutoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse")
    return model


def compute_reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """
    Tính MSE reconstruction error cho từng mẫu.

    Returns
    -------
    np.ndarray shape (n_samples,)
    """
    X_pred = model.predict(X, verbose=0)
    # Hỗ trợ cả 2D (dense) và 3D (lstm)
    if X.ndim == 3:
        mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))
    else:
        mse = np.mean(np.power(X - X_pred, 2), axis=1)
    return mse


def train_and_evaluate_dense(
    X_train: np.ndarray,
    X_demo: np.ndarray,
    y_demo: np.ndarray,
    threshold_strategy: str = "mean_std",
    epochs: int = 30,
    batch_size: int = 64,
    save_path: str = None
) -> dict:
    """
    Train Dense AE, tính metrics và trả về kết quả đầy đủ.
    """
    import tensorflow as tf
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from analysis.dynamic_threshold import DynamicThreshold
    from analysis.metrics import compute_classification_metrics

    input_dim = X_train.shape[1]
    print(f"\n[Dense AE] Input dim: {input_dim}, Epochs: {epochs}")

    model = build_dense_autoencoder(input_dim)
    model.summary()

    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    # Tính reconstruction error trên train để fit threshold
    train_mse = compute_reconstruction_error(model, X_train)

    # Tính threshold
    dt = DynamicThreshold(strategy=threshold_strategy)
    threshold = dt.fit(train_mse)

    # Tính reconstruction error trên demo
    demo_mse = compute_reconstruction_error(model, X_demo)

    # Dự đoán
    y_pred = ["ANOMALY" if e > threshold else "NORMAL" for e in demo_mse]

    # Metrics
    metrics = compute_classification_metrics(y_demo, y_pred, demo_mse, threshold)
    metrics["history"] = history.history
    metrics["model_name"] = "Dense Autoencoder"
    metrics["threshold_strategy"] = threshold_strategy
    metrics["threshold"] = threshold
    metrics["train_mse_mean"] = float(np.mean(train_mse))
    metrics["train_mse_std"] = float(np.std(train_mse))
    metrics["demo_mse"] = demo_mse

    if save_path:
        model.save(save_path)
        print(f"[Dense AE] Model saved → {save_path}")

    return model, metrics, demo_mse
