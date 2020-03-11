#!/usr/bin/env python3
"""f1 score
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix
    """
    sens = sensitivity(confusion)
    prcs = precision(confusion)
    f1_score = sens * prcs / (sens + prcs)

    return 2 * f1_score
