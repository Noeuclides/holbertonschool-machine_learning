#!/usr/bin/env python3
"""Convolutional forward prop module
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer of a
    neural network:
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        - m: number of examples
        - h_prev: height of the previous layer
        - w_prev: width of the previous layer
        - c_prev: number of channels in the previous layer
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
        - kh: filter height
        - kw: filter width
        - c_prev: number of channels in the previous layer
        - c_new: number of channels in the output
    - b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    - activation: activation function applied to the convolution
    - padding: string that is either same or valid, indicating the
    type of padding used
    - stride: tuple of (sh, sw) containing the strides for the convolution
        - sh: stride for the height
        - sw: stride for the width
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "valid":
        ph, pw = 0, 0
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)

    out_h = int((h_prev - kh) / sh + 1)
    out_w = int((w_prev - kw) / sw + 1)

    out_image = np.ndarray((m, out_h, out_w, c_new))
    pad_images = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    for i in range(0, out_h, sh):
        for j in range(0, out_w, sw):
            for channel in range(c_new):
                out_image[
                    :,
                    int(i / sh),
                    int(j / sw),
                    channel] = activation(
                    np.sum(pad_images[:, i:i + kh, j:j + kw, :] *
                           W[:, :, :, channel] + b, axis=(1, 2, 3)))

    return out_image
