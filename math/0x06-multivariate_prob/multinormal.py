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
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        mean = np.mean(data.T, axis=0)
        data_mean = data.T - mean
        cov = (np.dot(data_mean.T, data_mean)) / (data.shape[1] - 1)
        self.mean = mean.reshape(data.shape[0], 1)
        self.cov = cov

    def pdf(self, x):
        """
        calculates the PDF at a data point:

        - x: numpy.ndarray of shape (d, 1) containing the data point
        whose PDF should be calculated
            - d: number of dimensions of the Multinomial instance
        Returns the value of the PDF
        """
