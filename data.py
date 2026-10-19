from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Tuple

DataTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _scale(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 255.0)[..., np.newaxis]


def load_dataset(name: str) -> dict:
    if name == "mnist":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    elif name == "fashion_mnist":
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: '{name}'. Choose 'mnist' or 'fashion_mnist'.")

    x_tr = _scale(x_tr)
    x_te = _scale(x_te)
    x_all = np.concatenate([x_tr, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)

    return {
        "x_train": x_tr,
        "y_train": y_tr,
        "x_test":  x_te,
        "y_test":  y_te,
        "x_all":   x_all,
        "y_all":   y_all,
    }


def get_tf_dataset(x: np.ndarray, batch_size: int, shuffle: bool = True,
                   seed: int = 42) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, x))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
