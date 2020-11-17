#!/usr/bin/env python3
"""
Backward Algorithm module
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    performs the backward algorithm for a hidden markov model:

    - Observation: numpy.ndarray of shape (T,) that contains the index
    of the observation
        - T: number of observations
    - Emission: numpy.ndarray of shape (N, M) containing the emission
    probability
    of a specific observation given a hidden state
        - Emission[i, j]: probability of observing j given the hidden state i
        - N: number of hidden states
        - M: number of all possible observations
    - Transition: 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities
        - Transition[i, j]: probability of transitioning from the hidden
        state i to j
    - Initial: numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state
    Returns: P, B, or None, None on failure
        - Pis: likelihood of the observations given the model
        - B: numpy.ndarray of shape (N, T) containing the backward path
        probabilities
            - B[i, j]: probability of generating the future observations from
            hidden state i at time j
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if (np.sum(Emission, axis=1) != 1).any():
        return None, None
    if (np.sum(Transition, axis=1) != 1).any():
        return None, None
    if (np.sum(Initial, axis=0) != 1).any():
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for col in range(T - 2, -1, -1):
        for row in range(N):
            beta[row, col] = np.sum(beta[:, col + 1] *
                                    Transition[row, :] *
                                    Emission[:, Observation[col + 1]])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])

    return P, beta
