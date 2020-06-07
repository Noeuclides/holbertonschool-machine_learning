#!/usr/bin/env python3
"""module make a predicton
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network:

    - network: network model to make the prediction with
    - data: input data to make the prediction with
    - verbose: boolean that determines if output should be printed
    during the prediction process
    Returns: the prediction for the data
    """
    result = network.predict(data, verbose=verbose)

    return result
