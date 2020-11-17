#!/usr/bin/env python3
"""
Viretbi Algorithm module
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    calculates the most likely sequence of hidden states for a hidden
    markov model:

    - Observation: numpy.ndarray of shape (T,) that contains the index
    of the observation
        - T: number of observations
    - Emission: numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
        - Emission[i, j]: probability of observing j given the hidden state i
        - N: number of hidden states
        - M: number of all possible observations
    - Transition: 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities
        - Transition[i, j]: probability of transitioning from the hidden
        state i to j
    - Initial: numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state
    Returns: path, P, or None, None on failure
        - path: list of length T containing the most likely sequence of
        hidden states
        - P: probability of obtaining the path sequence
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

    omega = np.zeros((N, T))
    aux = (Initial * Emission[:, Observation[0]].reshape(-1, 1))
    omega[:, 0] = aux.reshape(-1)

    backpointer = np.zeros((N, T))
    backpointer[:, 0] = 0
    for col in range(1, T):
        for row in range(N):
            prev = omega[:, col - 1]
            trans = Transition[:, row]
            em = Emission[row, Observation[col]]
            result = prev * trans * em
            omega[row, col] = np.amax(result)
            backpointer[row, col - 1] = np.argmax(result)

    path = []
    # Find the most probable last hidden state
    last_state = np.argmax(omega[:, T - 1])
    path.append(int(last_state))

    # backtracking algorithm gotten from first read
    for i in range(T - 2, -1, -1):
        path.append(int(backpointer[int(last_state), i]))
        last_state = backpointer[int(last_state), i]

    # Flip the path array since we were backtracking
    path.reverse()

    min_prob = np.amax(omega, axis=0)
    min_prob = np.amin(min_prob)

    return path, min_prob
