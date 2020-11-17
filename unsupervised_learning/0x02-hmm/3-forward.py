#!/usr/bin/env python3
"""
Forward Algorithm module
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model:

    - Observation: numpy.ndarray of shape (T,) that contains the index
    of the observation
        - T: number of observations
    - Emission: numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
        - Emission[i, j]: probability of observing j given the hidden state i
        - N: number of hidden states
        - M: number of all possible observations
    - Transition: 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
        - Transition[i, j]: probability of transitioning from the hidden
        state i to j
    - Initial: numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state
    Returns: P, F, or None, None on failure
        - P: likelihood of the observations given the model
        - F: numpy.ndarray of shape (N, T) containing the forward path
        probabilities
            - F[i, j]: probability of being in hidden state i at time j
            given the previous observations
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

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial.T * Emission[:, Observation[0]]

    for col in range(1, T):
        for row in range(N):
            aux = alpha[:, col - 1] * Transition[:, row]
            alpha[row, col] = np.sum(aux * Emission[row, Observation[col]])

    P = np.sum(alpha[:, -1])

    return P, alpha
