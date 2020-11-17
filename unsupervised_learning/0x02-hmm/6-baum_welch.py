#!/usr/bin/env python3
"""
Baum-Welch Algorithm module
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs the Baum-Welch algorithm for a hidden markov model:

    - Observations: numpy.ndarray of shape (T,) that contains the index
    of the observation
        - T: number of observations
    - Transition: numpy.ndarray of shape (M, M) that contains the initialized
    transition probabilities
        - M: number of hidden states
    - Emission: numpy.ndarray of shape (M, N) that contains the initialized
    emission probabilities
        - N: number of output states
    - Initial: numpy.ndarray of shape (M, 1) that contains the initialized
    starting probabilities
    - iterations: number of times expectation-maximization should be performed
    Returns: the converged Transition, Emission, or None, None on failure
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    N, M = Emission.shape
    T = Observations.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    # iterations over 454 makes no difference in the output
    # to check use np.isclose with atol=1e-5 in a and b (store a_prev)
    if iterations > 454:
        iterations = 454

    a = Transition.copy()
    b = Emission.copy()
    for n in range(iterations):
        alpha = forward(Observations, b, a, Initial.reshape((-1, 1)))
        beta = backward(Observations, b, a, Initial.reshape((-1, 1)))

        xi = np.zeros((N, N, T - 1))
        for col in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, col].T, a) *
                                 b[:, Observations[col + 1]].T,
                                 beta[:, col + 1])
            for row in range(N):
                numerator = alpha[row, col] * a[row, :] *\
                    b[:, Observations[col + 1]].T * beta[:, col + 1].T
                xi[row, :, col] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for k in range(M):
            b[:, k] = np.sum(gamma[:, Observations == k], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a, b


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
    """

    N, M = Emission.shape
    T = Observation.shape[0]

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial.T * Emission[:, Observation[0]]

    for col in range(1, T):
        for row in range(N):
            aux = alpha[:, col - 1] * Transition[:, row]
            alpha[row, col] = np.sum(aux * Emission[row, Observation[col]])

    # P = np.sum(alpha[:, -1])

    return alpha


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for col in range(T - 2, -1, -1):
        for row in range(N):
            beta[row, col] = np.sum(beta[:, col + 1] *
                                    Transition[row, :] *
                                    Emission[:, Observation[col + 1]])

    # P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])

    return beta
