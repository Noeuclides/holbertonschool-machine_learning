#!/usr/bin/env python3
"""Convolutional forward prop module
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer of a nn:
    - A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) with the output
    of the previous layer.
    - W: numpy.ndarray (kh, kw, c_prev, c_new) with the kernels for the conv.
    - b: numpy.ndarray (1, 1, 1, c_new) with the biases applied to the conv.
    - activation: activation function applied to the convolution
    - padding: string that is either same or valid
    - stride: tuple of (sh, sw)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    pad_height, pad_width = 0, 0
    if padding == 'same':
        pad_height = int(((sh * h_prev) - sh + kh - h_prev) / 2)
        pad_width = int(((sw * w_prev) - sw + kw - w_prev) / 2)

    A_prev = np.pad(
        array=A_prev,
        pad_width=[
            (0, 0),
            (pad_height, pad_height),
            (pad_width, pad_width),
            (0, 0)],
        mode='constant',
        constant_values=0)

    output_h = int(((h_prev + (2 * pad_height) - kh) / sh) + 1)
    output_w = int(((w_prev + (2 * pad_width) - kw) / sw) + 1)
    convolution = np.zeros((m, output_h, output_w, c_new))

    prev_m = np.arange(0, m)
    for h in range(output_h):
        for w in range(output_w):
            for c in range(c_new):
                convole = A_prev[prev_m, h * sh:kh +
                                 (h * sh), w * sw:kw + (w * sw)] *
                W[:, :, :, c]
                summatory = np.sum(convole, axis=(1, 2, 3))
                convolution[prev_m, h, w, c] = activation(
                    summatory + b[0, 0, 0, c])
    return convolution
