#!/usr/bin/env python3
"""
Vanilla Autoencoder module
"""
import numpy as np
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates an autoencoder:

    - input_dims: integer containing the dimensions of the model input
    - hidden_layers: list containing the number of nodes for each
    hidden layer in the encoder, respectively
        - the hidden layers should be reversed for the decoder
    - latent_dims: integer containing the dimensions of the
    latent space representation
    Returns: encoder, decoder, auto
        - encoder: encoder model
        - decoder: decoder model
        - auto: full autoencoder model
    """
    # encoder
    encoder_input = tf.keras.Input(shape=(input_dims,))
    encoder = tf.keras.layers.Dense(hidden_layers[0],
                                    activation='relu')(encoder_input)
    for layer in hidden_layers[1:]:
        encoder = tf.keras.layers.Dense(layer, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(latent_dims, activation='relu')(encoder)
    encoder = tf.keras.Model(inputs=encoder_input, outputs=encoder)

    # decoder
    decoder_input = tf.keras.Input(shape=(latent_dims,))
    decoder = tf.keras.layers.Dense(hidden_layers[-1],
                                    activation='relu')(decoder_input)
    for layer in reversed(hidden_layers[:-1]):
        decoder = tf.keras.layers.Dense(layer, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(input_dims, activation='sigmoid')(decoder)
    decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder)

    # autoencoder
    encoder_output = encoder(encoder_input)
    decoder_output = decoder(encoder_output)
    auto = tf.keras.Model(inputs=encoder_input, outputs=decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
