#!/usr/bin/env python3
"""train module
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    trains a model using mb gradient descent with early stopping:
    - network: model to train
    - data: numpy.ndarray of shape (m, nx) containing the input data
    - labels: one-hot numpy.ndarray of shape (m, classes) with the labels
    - batch_size: size of the batch used for mini-batch gradient descent
    - epochs: number of passes through data for mini-batch gradient descent
    - validation_data: data to validate the model with
    - early_stopping: boolean that indicates whether early
    stopping should be used:
    - early stopping should only be performed if validation_data exists
    - early stopping should be based on validation loss

    - verbose: determines if output should be printed during training
    - shuffle: determines whether to shuffle the batches every epoch.
    """
    callback = None
    if validation_data:
        callback = [K.callbacks.EarlyStopping(patience=patience)]

    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callback,
                          verbose=verbose, shuffle=shuffle)
    return history
