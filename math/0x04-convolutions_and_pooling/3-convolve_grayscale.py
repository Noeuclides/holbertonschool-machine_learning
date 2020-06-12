#!/usr/bin/env python3
"""Convolution module
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images:
    - images: numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        - m: number of images
        - h: height in pixels of the images
        - w: width in pixels of the images
    - kernel: numpy.ndarray with shape (kh, kw) containing the
    kernel for the convolution
        - kh: height of the kernel
        - kw: width of the kernel
    - padding: either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            - ph: padding for the height of the image
            - pw: padding for the width of the image
        the image should be padded with 0’s
    - stride: tuple of (sh, sw)
        - sh: stride for the height of the image
        - sw: stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    if padding == 'same':
        ph = int((kh - 1) / 2)
        pw = int((kw - 1) / 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    sh, sw = stride

    images_pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    _, h_pad, w_pad = images_pad.shape

    out_w = int((w + 2*pw - kw) / sw + 1)
    out_h = int((h + 2*ph - kh) / sh + 1)

    convolve_out = np.ndarray((m, out_h, out_w))

    for i in range(0, h_pad, sh):
        for j in range(0, w_pad, sw):
            if kh + i <= h_pad and kw + j <= w_pad:
                convolve_out[:, int(i / sh), int(j / sw)] = np.sum(
                    images_pad[:, i:kh+i, j:kw+j] * kernel,
                    axis=(1, 2))

    return convolve_out
