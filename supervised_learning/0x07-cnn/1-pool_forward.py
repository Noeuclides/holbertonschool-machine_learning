#!/usr/bin/env python3
"""foward prpagation over pooling
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network:
    - A_prev_numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        - m: number of examples
        - h_prev: height of the previous layer
        - w_prev: width of the previous layer
        - c_prev: number of channels in the previous layer
    - kernel_shape: tuple of (kh, kw) containing the size of
    the kernel for the pooling
        - kh: kernel height
        - kw: kernel width
    - stride: tuple of (sh, sw) containing the strides for the pooling
        - sh is the stride for the height
        - sw is the stride for the width
    - mode: string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    ph = int(1 + (h_prev - kh) / sh)
    pw = int(1 + (w_prev - kw) / sw)

    output = np.zeros((m, ph, pw, c_prev))
    m_prev = np.arange(0, m)
    for h in range(ph):
        for w in range(pw):
            for c in range(c_prev):
                pool = A_prev[m_prev, h * sh:h *
                              sh + kh, w * sw:w * sw + kw, c]
                if mode == 'max':
                    output[m_prev, h, w, c] = np.max(pool)
                elif mode == "avg":
                    output[m_prev, h, w, c] = np.mean(pool)

    return output
