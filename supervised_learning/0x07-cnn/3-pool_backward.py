#!/usr/bin/env python3
"""
Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network:
    - dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
        - m: number of examples
        - h_new: height of the output
        - w_new: width of the output
        - c: number of channels
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
    output of the previous layer
        - h_prev: height of the previous layer
        - w_prev: width of the previous layer
    - kernel_shape: tuple of (kh, kw) containing the size of the kernel
    for the pooling
        - kh: kernel height
        - kw: kernel width
    - stride: tuple of (sh, sw) containing the strides for the pooling
        - sh: stride for the height
        - sw: stride for the width
    - mode: string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev)
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for img in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for ch in range(c):
                    row = i * sh
                    col = j * sw

                    if mode == 'max':
                        pool = A_prev[img, row:row + kh,
                                      col:col + kw, ch]
                        mask = (pool == np.max(pool))
                        dA_prev[img, row:row + kh,
                                col:col + kw, ch] += dA[img, i, j, ch] * mask
                    if mode == 'avg':
                        avg = dA[img, i, j, ch] / (kh * kw)
                        mask = np.ones(kernel_shape) * avg
                        dA_prev[img, row:row + kh, col:col + kw, ch] += mask
    return dA_prev
