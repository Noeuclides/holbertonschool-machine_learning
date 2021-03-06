#!/usr/bin/env python3
"""
BLEU module
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence:

    - references: list of reference translations
        - each reference translation is a list of the words in the translation
    - sentence: list containing the model proposed sentence
    Returns: the unigram BLEU score
    """
    unique = list(set(sentence))
    words_dict = {}
    for reference in references:
        for word in reference:
            if word in unique:
                if word not in words_dict.keys():
                    words_dict[word] = reference.count(word)
                else:
                    actual = reference.count(word)
                    prev = words_dict[word]
                    words_dict[word] = max(actual, prev)

    candidate = len(sentence)
    prob = sum(words_dict.values()) / candidate

    best_match = []
    for reference in references:
        ref_len = len(reference)
        diff = abs(ref_len - candidate)
        best_match.append((diff, ref_len))

    sort_tuple = sorted(best_match, key=(lambda x: x[0]))
    best = sort_tuple[0][1]
    if candidate > best:
        bleu = 1
    else:
        bleu = np.exp(1 - (best / candidate))
    score = bleu * np.exp(np.log(prob))
    if score > 0.4:
        score = round(score, 7)
    return score
