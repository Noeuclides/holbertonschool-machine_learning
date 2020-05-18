#!/usr/bin/env python3
"""trains
"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier
    - X_train: numpy.ndarray containing the training input data
    - Y_train: numpy.ndarray containing the training labels
    - X_valid: numpy.ndarray containing the validation input data
    - Y_valid: numpy.ndarray containing the validation labels
    - layer_sizes: list containing the number of nodes in each layer
    - activations: list containing the activation functions for each layer
    - alpha: learning rate
    - iterations: number of iterations to train over
    - save_path designates where to save the model
    Returns: the path where the model was saved
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    init = tf.global_variables_initializer()
    train = {x: X_train, y: Y_train}
    validation = {x: X_valid, y: Y_valid}
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            loss_train = sess.run(loss, feed_dict=train)
            acc_train = sess.run(accuracy, feed_dict=train)
            loss_val = sess.run(loss, feed_dict=validation)
            acc_val = sess.run(accuracy, feed_dict=validation)
            sess.run(train_op, feed_dict=train)
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValidation Cost: {}".format(loss_val))
                print("\tValidation Accuracy: {}".format(acc_val))
        save_path = saver.save(sess, save_path)
    return save_path
