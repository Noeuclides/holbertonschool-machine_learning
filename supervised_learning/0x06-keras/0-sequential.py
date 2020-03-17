#!/usr/bin/env python3
"""Sequential module
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
    model = K.models.Sequential()
    model.add(
        K.layers.Dense(
            layers[0],
            activation=activations[0],
            kernel_regularizer=K.regularizers.l2(lambtha),
            input_dim=nx))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - 0.2))
        model.add(
            K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))

    return model
