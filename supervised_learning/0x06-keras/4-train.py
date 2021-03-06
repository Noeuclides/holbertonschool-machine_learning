#!/usr/bin/env python3
"""train module
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent:
    - network: model to train
    - data: numpy.ndarray of shape (m, nx) containing the input data
    - labels: one-hot numpy.ndarray of shape (m, classes) with the labels
    - batch_size: size of the batch used for mini-batch gradient descent
    - epochs: number of passes through data for mini-batch gradient descent
    - verbose: determines if output should be printed during training
    - shuffle: determines whether to shuffle the batches every epoch.
    """
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
