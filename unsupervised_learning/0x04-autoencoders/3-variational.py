#!/usr/bin/env python3
"""
Variational autoencoder module
"""
import numpy as np
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder:

    - input_dims: integer containing the dimensions of the model input
    - hidden_layers: list containing the number of nodes for each
    hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    - latent_dims: integer containing the dimensions of the
    latent space representation
    Returns: encoder, decoder, auto
        - encoder: encoder model, which should output the latent
        representation, the mean, and the log variance, respectively
        - decoder: decoder model
        - auto: full autoencoder model
    """
    pass
