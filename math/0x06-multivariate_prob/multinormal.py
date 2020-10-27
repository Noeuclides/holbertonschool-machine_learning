#!/usr/bin/env python3
"""
Multivariate normal distribution
"""
import numpy as np


class MultiNormal:
    """
    represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        - data: numpy.ndarray of shape (d, n) containing the data set:
            - n: number of data points
            - d: number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(data.T, axis=0)
        data_mean = data.T - mean
        cov = (np.dot(data_mean.T, data_mean)) * (1 / data.shape[1])
        self.mean = np.array([mean]).T
        self.cov = cov
