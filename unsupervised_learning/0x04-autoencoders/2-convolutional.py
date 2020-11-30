#!/usr/bin/env python3
"""
Convolutional autoencoder module
"""
import numpy as np
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates a convolutional autoencoder:

    - input_dims: tuple of integers containing the dimensions
    of the model input
    - filters: list containing the number of filters for each
    convolutional layer in the encoder, respectively
        - the filters should be reversed for the decoder
    - latent_dims: tuple of integers containing the dimensions
    of the latent space representation
    Returns: encoder, decoder, auto
        - encoder: encoder model
        - decoder: decoder model
        - auto: full autoencoder model
    """
    pass
