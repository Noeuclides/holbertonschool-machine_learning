#!/usr/bin/env python3
"""
Module to concatenate arrays
"""


def cat_arrays(arr1, arr2):
    """
    Concatenates two arrays
    """
    cat = arr1[:]
    for item in arr2:
        cat.append(item)
    return cat
