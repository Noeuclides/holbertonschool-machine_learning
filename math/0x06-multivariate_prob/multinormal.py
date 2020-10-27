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
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a 2D numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        den = ((2 * np.pi) ** (x.shape[0] / 2)) * np.sqrt(det)
        inv = np.linalg.inv(self.cov)
        diff = np.dot((x - self.mean).T, inv)
        exponent = -1 * np.dot(diff, (x - self.mean)) / 2
        pdf = (1 / den) * np.exp(exponent)

        return pdf[0][0]
