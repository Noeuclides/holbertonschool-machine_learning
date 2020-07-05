#!/usr/bin/env python3
"""
Module to do back propagations on CNN
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network:
    - dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the unactivated output of
    the convolutional layer
        - m: number of examples
        - h_new: height of the output
        - w_new: width of the output
        - c_new: number of channels in the output
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        - h_prev: height of the previous layer
        - w_prev: width of the previous layer
        - c_prev: number of channels in the previous layer
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        - kh: filter height
        - kw: filter width
    - b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    - padding: string that is either same or valid, indicating the type
    of padding used
    - stride: tuple of (sh, sw) containing the strides for the convolution
        - sh: stride for the height
        - sw: stride for the width

    Returns: the partial derivatives with respect to the previous
    layer (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    _, h_new, w_new, _ = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((sh*h_prev)-sh+kh-h_prev)/2))
        pw = int(np.ceil(((sw*w_prev)-sw+kw-w_prev)/2))
    else:
        ph, pw = 0, 0

    out_img = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    dA = np.zeros(out_img.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for img in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for ch in range(c_new):
                    row = i * sh
                    col = j * sw
                    dA[img, row:row + kh,
                       col:col + kw, :] += dZ[img, i, j, ch] * W[:, :, :, ch]
                    pad_img = out_img[img, row:row + kh, col:col + kw, :]
                    dW[:, :, :, ch] += pad_img * dZ[img, i, j, ch]

    # if padding == 'same':
        # dA = dA[:, ph:-ph, pw:-pw, :]

    return dA, dW, db
