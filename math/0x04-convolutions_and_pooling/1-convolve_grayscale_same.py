#!/usr/bin/env python3
"""Convolution module
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a valid convolution on grayscale images:
    - images: numpy.ndarray, shape (m, h, w), with multiple grayscale images
    - kernel: numpy.ndarray, shape (kh, kw), with the kernel for the conv.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = int(kh / 2)
    pw = int(kw / 2)

    if kh % 2 != 0:
        out_h = (h - kh + 2 * ph + 1)
    else:
        out_h = (h - kh + 2 * ph)

    if kw % 2 != 0:
        out_w = (w - kw + 2 * pw + 1)
    else:
        out_w = (w - kw + 2 * pw)

    output = np.zeros((m, out_h, out_w))
    out_m = np.arange(m)

    pad_img = np.pad(images, [(0,), (ph,), (pw,)],
                     mode='constant', constant_values=0)

    for i in range(out_h):
        for j in range(out_w):
            output[out_m, i, j] =
            (pad_img[out_m, i:i + kh, j:j + kw] * kernel).sum(axis=(1, 2))

    return output
