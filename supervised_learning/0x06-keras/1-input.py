#!/usr/bin/env python3
"""Input module
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library:
    - nx: number of input features to the network
    - layers: list containing the number of nodes in each layer of the network
    - activations: list containing the activation functions used for each layer
    - lambtha: L2 regularization parameter
    - keep_prob: probability that a node will be kept for dropout
    """
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha))(inputs)
    for i in range(1, len(layers) - 1):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(x)

    x = K.layers.Dropout(1 - keep_prob)(x)
    predictions = K.layers.Dense(layers[len(layers) - 1])(x)
    model = K.Model(inputs=inputs, outputs=predictions)

    return model
