#!/usr/bin/env python3
"""module that defines a deep neural network
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    class that defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """class constructor
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if activation != 'sig' or activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        l_prev = nx
        for l in range(len(layers)):
            if not isinstance(layers[l], int):
                raise TypeError('layers must be a list of positive integers')
            key = 'W{}'.format(l + 1)
            bias = 'b{}'.format(l + 1)
            self.__weights[key] = np.random.randn(
                layers[l], l_prev) * np.sqrt(2 / l_prev)
            self.__weights[bias] = np.zeros((layers[l], 1))
            l_prev = layers[l]

    @property
    def L(self):
        """get number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """get dictionary to hold all intermediary values of the network.
        """
        return self.__cache

    @property
    def weights(self):
        """get dictionary to hold all weights and biased of the network.
        """
        return self.__weights

    @property
    def activation(self):
        """type of activation function used in the hidden layers
        """
        return self.__activation

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network
        """
        self.__cache['A0'] = X
        for layer in range(self.__L):
            key = 'W{}'.format(layer + 1)
            bias = 'b{}'.format(layer + 1)
            w = self.__weights[key]
            key_cache = 'A{}'.format(layer)
            cache = self.__cache[key_cache]
            z = np.matmul(w, cache) + self.__weights[bias]
            A = 'A{}'.format(layer + 1)
            if layer == self.__L - 1:
                den = np.sum(np.exp(z), axis=0, keepdims=True)
                self.__cache[A] = np.exp(z) / den
            else:
                self.__cache[A] = self.activationFunction(z)
        out = 'A{}'.format(self.__L)
        return self.__cache[out], self.__cache

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression
        """
        loss1 = Y * np.log(A)
        m = Y.shape[1]
        cost = -1 * np.sum(loss1) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network’s predictions
        """
        A, _ = self.forward_prop(X)
        max = np.amax(A, axis=0, keepdims=True)
        key = 'A{}'.format(self.__L)
        return np.where(self.__cache[key] == max, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network
        """
        W_copy = self.weights.copy()
        for layer in range(self.__L, 0, -1):
            key = 'A{}'.format(layer)
            input = 'A{}'.format(layer - 1)
            w = 'W{}'.format(layer)
            out = 'A{}'.format(self.__L)
            bias = 'b{}'.format(layer)
            if layer == self.__L:
                dz = cache[out] - Y
                dw = np.matmul(dz, cache[input].T) / Y.shape[1]
            else:
                w1 = 'W{}'.format(layer + 1)
                back = np.matmul(W_copy[w1].T, dz)
                derivative = self.derivative(cache[key])
                dz = back * derivative
                dw = np.matmul(dz, cache[input].T) / Y.shape[1]
            db = np.sum(dz, axis=1, keepdims=True) / Y.shape[1]
            self.__weights[w] = W_copy[w] - alpha * dw
            self.__weights[bias] = W_copy[bias] - alpha * db
        return self.__weights

    def activationFunction(self, z):
        """activation function sigmoid or tanh
        """
        if self.__activation == 'sig':
            activation = 1 / (1 + np.exp(-z))
        else:
            tanhnum = np.exp(z) - np.exp(-z)
            tanhden = np.exp(z) + np.exp(-z)
            activation = tanhnum / tanhden
        return activation

    def derivative(self, cache):
        """derivate of the activation function
        """
        if self.__activation == 'sig':
            derivative = cache * (1 - cache)
        else:
            derivative = 1 - cache ** 2
        return derivative

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Train the deep neural network
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
        iteration = np.arange(0, iterations)
        for i in range(iterations):
            A, self.__cache = self.forward_prop(X)
            key = "A{}".format(self.__L)
            cost = self.cost(Y, A)
            costs.append(cost)
            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            self.gradient_descent(Y, self.__cache, alpha)
        if graph:
            plt.plot(iteration, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Save the instance object to a file in pickle format
        """
        if '.pkl' not in filename:
            filename = filename + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            return None
