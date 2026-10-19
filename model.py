from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

import config


class Sampling(layers.Layer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:  # type: ignore[override]
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_encoder(latent_dim: int,
                  dropout_rate: float,
                  name: str = "encoder") -> Model:
    kernel_init = "glorot_uniform"

    inputs = layers.Input(shape=config.IMAGE_SHAPE, name="encoder_input")

    x = layers.Conv2D(64, 4, padding="same", kernel_initializer=kernel_init,
                      name="enc_conv1")(inputs)
    x = layers.LeakyReLU(name="enc_lrelu1")(x)
    x = layers.MaxPool2D((2, 2), name="enc_pool1")(x)

    x = layers.Conv2D(64, 4, padding="same", kernel_initializer=kernel_init,
                      name="enc_conv2")(x)
    x = layers.LeakyReLU(name="enc_lrelu2")(x)
    x = layers.MaxPool2D((2, 2), name="enc_pool2")(x)

    x = layers.Conv2D(64, 4, padding="same", kernel_initializer=kernel_init,
                      name="enc_conv3")(x)
    x = layers.LeakyReLU(name="enc_lrelu3")(x)
    x = layers.MaxPool2D((1, 1), name="enc_pool3")(x)

    x = layers.Flatten(name="enc_flatten")(x)
    x = layers.Dropout(rate=dropout_rate, name="enc_dropout")(x)

    z_mean    = layers.Dense(latent_dim, kernel_initializer=kernel_init,
                             name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, kernel_initializer=kernel_init,
                             name="z_log_var")(x)
    z         = Sampling(name="z_sampling")([z_mean, z_log_var])

    return Model(inputs, [z_mean, z_log_var, z], name=name)


def build_decoder(latent_dim: int, name: str = "decoder") -> Model:
    kernel_init = "glorot_uniform"

    latent_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")

    x = layers.Dense(7 * 7 * 64, kernel_initializer=kernel_init,
                     name="dec_dense")(latent_inputs)
    x = layers.Reshape((7, 7, 64), name="dec_reshape")(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same",
                               kernel_initializer=kernel_init,
                               name="dec_deconv1")(x)
    x = layers.LeakyReLU(name="dec_lrelu1")(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same",
                               kernel_initializer=kernel_init,
                               name="dec_deconv2")(x)
    x = layers.LeakyReLU(name="dec_lrelu2")(x)

    x = layers.Conv2DTranspose(1, 4, strides=1, padding="same",
                               kernel_initializer=kernel_init,
                               name="dec_deconv3")(x)
    outputs = layers.Activation("sigmoid", name="decoder_output")(x)

    return Model(latent_inputs, outputs, name=name)


def build_vae(latent_dim: int,
              dropout_is_keep_prob: bool = False) -> tuple[Model, Model, Model]:
    paper_val = config.DROPOUT_PAPER_VALUE
    if dropout_is_keep_prob:
        dropout_rate = 1.0 - paper_val
    else:
        dropout_rate = paper_val

    encoder = build_encoder(latent_dim, dropout_rate)
    decoder = build_decoder(latent_dim)

    vae_input               = layers.Input(shape=config.IMAGE_SHAPE, name="vae_input")
    z_mean, z_log_var, z    = encoder(vae_input)
    reconstructed           = decoder(z)

    vae = Model(vae_input, reconstructed, name="vae")

    vae.encoder = encoder      # type: ignore[attr-defined]
    vae.decoder = decoder      # type: ignore[attr-defined]

    return encoder, decoder, vae
