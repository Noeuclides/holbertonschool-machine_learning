#! /usr/bin/env python3


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise
    """
    if len(arr1) != len(arr2):
        return None

    add = []
    for i in range(len(arr1)):
        add.append(arr1[i] + arr2[i])

    return(add)
