#!/usr/bin/env python3
"""Module to L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization:
    - Y: one-hot numpy.ndarray of shape (classes, m) that contains
    the correct labels for the data
        - classes: number of classes
        - m: number of data points
    - weights: dictionary of the weights and biases of the neural network
    - cache: dictionary of the outputs of each layer of the neural network
    - alpha: learning rate
    - lambtha: L2 regularization parameter
    - L: number of layers of the network
    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    The weights and biases of the network should be updated in place
    """
    _, m = Y.shape
    key_A = 'A{}'.format(L)
    dz = cache[key_A] - Y
    for layer in range(L, 0, -1):
        key_A = 'A{}'.format(layer - 1)
        key_w = 'W{}'.format(layer)
        key_b = 'b{}'.format(layer)

        dw = np.matmul(cache[key_A], dz.T) / m
        dw_L2 = dw.T + lambtha * weights[key_w] / m
        db = np.sum(dz, axis=1, keepdims=True)
        
        # Update parameters
        weights[key_w] = weights[key_w] - alpha * dw_L2
        weights[key_b] = weights[key_b] - alpha * db

        # derivative of the tanh activation function
        derivative = 1 - cache[key_A] ** 2
        dz = np.matmul(weights[key_w].T, dz) * derivative
