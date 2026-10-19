from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

import config
from metrics import clustering_accuracy, compute_nmi


def encode_to_latent(encoder, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    z_means = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        z_mean, _, _ = encoder(batch, training=False)
        z_means.append(z_mean.numpy())
    return np.concatenate(z_means, axis=0)


def run_kmeans_clustering(
    encoder,
    x: np.ndarray,
    y_true: np.ndarray,
    k_values: list[int] | None = None,
    seed: int = config.GLOBAL_SEED,
) -> dict[str, dict[str, float]]:
    if k_values is None:
        k_values = config.K_VALUES

    z = encode_to_latent(encoder, x)

    results: dict[str, dict[str, float]] = {}
    for k in k_values:
        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=seed,
        )
        y_pred = km.fit_predict(z)

        acc = clustering_accuracy(y_true, y_pred)
        nmi = compute_nmi(y_true, y_pred)

        results[f"k{k}"] = {"acc": acc, "nmi": nmi}

    return results
