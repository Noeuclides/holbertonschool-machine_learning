#!/usr/bin/env python3
"""module that defines a single neuron
"""

import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """define a single neuron performing binary classification
    """

    def __init__(self, nx):
        """class constructor
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """setter method of the weights
        """
        return self.__W

    @property
    def b(self):
        """setter method of the bias
        """
        return self.__b

    @property
    def A(self):
        """setter method of the activated output
        """
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron
        """
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression
        """
        loss1 = np.matmul(Y, np.log(A).T)
        loss2 = np.matmul(1 - Y, np.log(1.0000001 - A).T)
        m = Y.shape[1]
        cost = np.sum(-(loss1 + loss2)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron’s predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent on the neuron
        """
        dw = np.matmul(X, (A - Y).T) / X.shape[1]
        self.__W = self.__W - alpha * dw.T
        db = np.sum(A - Y) / X.shape[1]
        self.__b = self.__b - alpha * db.T
        return self.__W, self.__b

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """Train the neuron
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose and graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        iteration = []
        for i in range(iterations + 1):
            self.__A = self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            if i % step == 0 or i == iterations:
                costs.append(cost)
                iteration.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            self.__W, self.__b = self.gradient_descent(X, Y, self.__A, alpha)

        if graph:
            plt.plot(iteration, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
