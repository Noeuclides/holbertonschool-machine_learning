#!/usr/bin/env python3
"""forward propagation module
"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    - x: placeholder for the input data
    - layer_sizes: list containing the number of nodes in each layer
    - activations: list containing the activation functions for each layer
    Returns: the prediction of the network in tensor form
    """
    y = create_layer(x, layer_sizes[0], activations[0])
    for l in range(1, len(layer_sizes)):
        y = create_layer(y, layer_sizes[l], activations[l])

    return y
