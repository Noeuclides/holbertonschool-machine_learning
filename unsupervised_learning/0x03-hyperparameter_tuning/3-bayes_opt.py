#!/usr/bin/env python3
"""
Bayesian optimization module
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        class constructor:

        - f: black-box function to be optimized
        - X_init: numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        - Y_init: numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        - t: number of initial samples
        - bounds: tuple of (min, max) representing the bounds of the space
        in which to look for the optimal point
        - ac_samples: number of samples that should be analyzed
        during acquisition
        - l: length parameter for the kernel
        - sigma_f: standard deviation given to the output of the
        black-box function
        - xsi: exploration-exploitation factor for acquisition
        - minimize: bool determining whether optimization should be
        performed for minimization (True) or maximization (False)

        Sets the following public instance attributes:
            - f: the black-box function
            - gp: an instance of the class GaussianProcess
            - X_s: a numpy.ndarray of shape (ac_samples, 1) containing
            all acquisition sample points, evenly spaced between min and max
            - xsi: the exploration-exploitation factor
            - minimize: a bool for minimization versus maximization
        """
        min, max = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.expand_dims(np.linspace(min, max, ac_samples), axis=1)
        self.xsi = xsi
        self.minimize = minimize
