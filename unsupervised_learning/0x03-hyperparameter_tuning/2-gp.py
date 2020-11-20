#!/usr/bin/env python3
"""
Gaussian Process module
"""
import numpy as np


class GaussianProcess:
    """
    class to perform gaussian processes
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor:
        - X_init: numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        - Y_init: numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        - t: number of initial samples
        - l: length parameter for the kernel
        - sigma_f: standard deviation given to the output of the black-box
        function
        Sets the public instance attributes X, Y, l, and sigma_f
        corresponding to the respective constructor inputs
        Sets the public instance attribute K, representing the current
        covariance kernel matrix for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices:
        - X1: numpy.ndarray of shape (m, 1)
        - X2: numpy.ndarray of shape (n, 1)
        Returns: the covariance kernel matrix as a numpy.ndarray of
        shape (m, n)

        rbf: k(x, x') = stdev * exp(- ||x - x'|| ** 2 / (2*stdev**2))
        """
        m, n = X1.shape[0], X2.shape[0]

        # get mxn matrices
        X1_dot = np.sum(X1 ** 2, axis=1).reshape(m, 1) * np.ones((1, n))
        X2_dot = np.sum(X2 ** 2, axis=1) * np.ones((m, 1))

        dist = X1_dot + X2_dot - 2 * np.dot(X1, X2.T)

        return self.sigma_f ** 2 * np.exp(- dist / (2 * self.l ** 2))

    def predict(self, X_s):
        """
        predicts the mean and standard deviation of points in a
        Gaussian process:

        - X_s: numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and standard deviation should be calculated
            - s: number of sample points
        Returns: mu, sigma
            - mu: numpy.ndarray of shape (s,) containing the mean
            for each point in X_s, respectively
            - sigma: numpy.ndarray of shape (s,) containing the variance
            for each point in X_s, respectively
        """
        s = X_s.shape[0]
        kernel = self.kernel(self.X, self.X)
        kernel_s = self.kernel(self.X, X_s)
        kernel_ss = self.kernel(X_s, X_s)
        k_inv = np.linalg.inv(kernel)
        mean_s = np.dot(np.dot(kernel_s.T, k_inv), self.Y)
        cov_s = kernel_ss - np.dot(np.dot(kernel_s.T, k_inv), kernel_s)

        return mean_s.reshape((s,)), np.diag(cov_s)

    def update(self, X_new, Y_new):
        """
        updates a Gaussian Process:
v
        - X_new: numpy.ndarray of shape (1,) that represents
        the new sample point
        - Y_new: numpy.ndarray of shape (1,) that represents
        the new sample function value
        Updates the public instance attributes X, Y, and K
        """
        self.X = np.concatenate((self.X, np.expand_dims(X_new, axis=1)))
        self.Y = np.concatenate((self.Y, np.expand_dims(Y_new, axis=1)))
        self.K = self.kernel(self.X, self.X)
