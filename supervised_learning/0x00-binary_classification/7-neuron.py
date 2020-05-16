#!/usr/bin/env python3
"""module that defines a single neuron
"""
import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """define a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        - nx: number of input features to the neuron

        Private instance attributes:
        -__W: The weights vector for the neuron.
        Initialized using a random normal distribution.
        - __b: The bias for the neuron.
        Initialized to 0.
        - __A: The activated output of the neuron (prediction).
        Initialized to 0.
        Each private attribute have its getter function
        (no setter function).
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
        """
        Calculates the forward propagation of the neuron
        - X: numpy.ndarray with shape (nx, m) that contains the input data
        - nx: number of input features to the neuron
        - m: number of examples
        Updates the private attribute __A
        The neuron use a sigmoid activation function
        - sig(z) = 1 / (1 + exp(-z))
        Returns the private attribute __A
        """
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Loss Function:
        L(A, Y) = -(Y * log(A) + (1 - Y) * log(1 - A)))
        Cost function:
        J(w, b) = (1 / m) * ∑(i=1; i<=m) L(A, Y)
        - Y: numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        - A: numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        Returns the cost
        """
        loss1 = np.matmul(Y, np.log(A).T)
        # 1.0000001 - A instead of 1 - A to avoid division by zero errors
        loss2 = np.matmul(1 - Y, np.log(1.0000001 - A).T)
        m = Y.shape[1]
        cost = np.sum(-(loss1 + loss2)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        - X: numpy.ndarray with shape (nx, m) that contains the input data
        - nx: number of input features to the neuron
        -m: number of examples
        Y: numpy.ndarray with shape (1, m) containing the correct labels
        for the input data.
        Returns the neuron’s prediction and the cost of the network:
        - prediction: numpy.ndarray with shape (1, m) containing the predicted
        labels for each example.
        - label values are 1 if the output of the network is >= 0.5 0 otherwise
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neuron.
        dz2 = A2 - Y
        dw2 = (1 / m) * dz2 * A1.T
        w1 := w1 - alpha * dw1
        b := b - alpha*db
        - X: numpy.ndarray with shape (nx, m) that contains the input data
        - nx: number of input features to the neuron
        - m: number of examples
        - Y: numpy.ndarray with shape (1, m) containing the correct labels
        for the input data.
        - A: numpy.ndarray with shape (1, m) containing the activated output
        of the neuron for each example.
        - alpha: learning rate
        Updates the private attributes __W and __b
        """
        dw = np.matmul(X, (A - Y).T) / X.shape[1]
        self.__W = self.__W - alpha * dw.T
        db = np.sum(A - Y) / X.shape[1]
        self.__b = self.__b - alpha * db.T
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Train the neuron
        - X: numpy.ndarray with shape (nx, m) that contains the input data
        - nx: number of input features to the neuron
        - m: number of examples
        - Y: numpy.ndarray with shape (1, m) that contains the correct labels
        for the input data.
        - iterations: number of iterations to train over
        - alpha: learning rate
        - verbose: boolean that defines whether or not to print information
        about the training.
        - graph: boolean that defines whether or not to graph information
        about the training once the training has completed.

        Updates the private attributes __W, __b, and __A
        Returns the evaluation of the training data.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
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
