from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
from skimage.metrics import structural_similarity as skimage_ssim

try:
    from sewar.full_ref import vifp as sewar_vifp
    _SEWAR_AVAILABLE = True
except ImportError:
    _SEWAR_AVAILABLE = False


def compute_mse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    return float(np.mean((x_true - x_pred) ** 2))


def compute_ssim(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    scores = []
    for img_t, img_p in zip(x_true, x_pred):
        s = skimage_ssim(img_t.squeeze(), img_p.squeeze(), data_range=1.0)
        scores.append(s)
    return float(np.mean(scores))


def _uqi_single(x: np.ndarray, y: np.ndarray, window_size: int = 8) -> float:
    H, W = x.shape
    scores = []
    for r in range(0, H - window_size + 1, window_size):
        for c in range(0, W - window_size + 1, window_size):
            xb = x[r:r + window_size, c:c + window_size].ravel()
            yb = y[r:r + window_size, c:c + window_size].ravel()
            mu_x   = np.mean(xb)
            mu_y   = np.mean(yb)
            var_x  = np.var(xb, ddof=1)
            var_y  = np.var(yb, ddof=1)
            cov_xy = np.mean((xb - mu_x) * (yb - mu_y))
            denom  = (var_x + var_y) * (mu_x**2 + mu_y**2)
            if denom < 1e-12:
                scores.append(1.0)
            else:
                scores.append((4 * cov_xy * mu_x * mu_y) / denom)
    return float(np.mean(scores)) if scores else 1.0


def compute_uqi(x_true: np.ndarray, x_pred: np.ndarray,
                window_size: int = 8) -> float:
    scores = []
    for img_t, img_p in zip(x_true, x_pred):
        s = _uqi_single(img_t.squeeze(), img_p.squeeze(), window_size)
        scores.append(s)
    return float(np.mean(scores))


def compute_vif(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    if not _SEWAR_AVAILABLE:
        return float("nan")

    scores = []
    for img_t, img_p in zip(x_true, x_pred):
        t = (img_t.squeeze() * 255).astype(np.uint8)
        p = (img_p.squeeze() * 255).astype(np.uint8)
        try:
            v = sewar_vifp(t, p)
            scores.append(float(v))
        except Exception:
            scores.append(float("nan"))
    return float(np.nanmean(scores))


def compute_image_metrics(x_true: np.ndarray, x_pred: np.ndarray,
                          max_samples: int = 1000) -> dict[str, float]:
    n = min(len(x_true), max_samples)
    idx = np.random.choice(len(x_true), n, replace=False)
    xt  = x_true[idx]
    xp  = x_pred[idx]

    return {
        "mse":  compute_mse(xt, xp),
        "ssim": compute_ssim(xt, xp),
        "uqi":  compute_uqi(xt, xp),
        "vif":  compute_vif(xt, xp),
    }


def _make_cost_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                      n_true: int, n_pred: int) -> np.ndarray:
    cost = np.zeros((n_true, n_pred), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost[t, p] += 1
    return cost


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    n_true = int(y_true.max()) + 1
    n_pred = int(y_pred.max()) + 1

    cost = _make_cost_matrix(y_true, y_pred, n_true, n_pred)
    row_ind, col_ind = linear_sum_assignment(-cost)
    correct = cost[row_ind, col_ind].sum()
    return float(correct) / float(len(y_true))


def compute_nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(normalized_mutual_info_score(y_true, y_pred,
                                              average_method="arithmetic"))
