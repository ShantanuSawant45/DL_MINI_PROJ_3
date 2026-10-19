from __future__ import annotations

import time
import numpy as np
import tensorflow as tf

import config
from data import load_dataset, get_tf_dataset
from model import build_vae
from losses import standard_vae_loss, make_ssim_weighted_loss


def train_vae(
    dataset: str,
    latent_dim: int,
    loss_type: str = "standard",
    n_exponent: float = 1.0,
    seed: int = config.GLOBAL_SEED,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    verbose: bool = True,
) -> tuple:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data     = load_dataset(dataset)
    train_ds = get_tf_dataset(data["x_train"], batch_size=batch_size,
                              shuffle=True, seed=seed)

    encoder, decoder, vae = build_vae(latent_dim)

    if loss_type == "standard":
        loss_fn = standard_vae_loss
    elif loss_type == "ssim_weighted":
        loss_fn = make_ssim_weighted_loss(n_exponent)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. "
                         f"Choose 'standard' or 'ssim_weighted'.")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables

    @tf.function
    def train_step(x_batch: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(x_batch, training=True)
            x_hat                = decoder(z, training=True)
            loss                 = loss_fn(x_batch, x_hat, z_mean, z_log_var)
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0
        for x_batch, _ in train_ds:
            batch_loss  = train_step(x_batch)
            epoch_loss += float(batch_loss)
            n_batches  += 1

        if verbose:
            mean_loss = epoch_loss / max(n_batches, 1)
            print(f"  [{dataset} | latent={latent_dim} | {loss_type}] "
                  f"Epoch {epoch:03d}/{epochs}  loss={mean_loss:.4f}")

    elapsed_sec = time.perf_counter() - t0

    if verbose:
        print(f"  Training finished in {elapsed_sec:.1f}s\n")

    return encoder, decoder, vae, elapsed_sec
