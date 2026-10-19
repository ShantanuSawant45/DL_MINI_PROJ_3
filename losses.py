from __future__ import annotations

import tensorflow as tf
from typing import Callable


def _kl_divergence(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
    kl = -0.5 * tf.reduce_sum(
        1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1,
    )
    return tf.reduce_mean(kl)


def _bce_reconstruction(x_true: tf.Tensor, x_pred: tf.Tensor) -> tf.Tensor:
    x_true_flat = tf.reshape(x_true, [tf.shape(x_true)[0], -1])
    x_pred_flat = tf.reshape(x_pred, [tf.shape(x_pred)[0], -1])
    bce = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(x_true_flat, x_pred_flat),
        axis=-1,
    )
    n_pixels = tf.cast(tf.shape(x_true_flat)[1], tf.float32)
    return tf.reduce_mean(bce * n_pixels)


def standard_vae_loss(
    x_true: tf.Tensor,
    x_pred: tf.Tensor,
    z_mean: tf.Tensor,
    z_log_var: tf.Tensor,
) -> tf.Tensor:
    bce = _bce_reconstruction(x_true, x_pred)
    kl  = _kl_divergence(z_mean, z_log_var)
    return bce + kl


def make_ssim_weighted_loss(n: float) -> Callable:
    def ssim_weighted_vae_loss(
        x_true: tf.Tensor,
        x_pred: tf.Tensor,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
    ) -> tf.Tensor:
        elbo_loss = standard_vae_loss(x_true, x_pred, z_mean, z_log_var)

        if n == 0:
            return elbo_loss

        ssim_vals = tf.image.ssim(x_true, x_pred, max_val=1.0)
        weight = tf.reduce_mean(tf.pow(1.0 - ssim_vals, float(n)))
        return elbo_loss * weight

    ssim_weighted_vae_loss.__name__ = f"ssim_weighted_n{n}"
    return ssim_weighted_vae_loss
