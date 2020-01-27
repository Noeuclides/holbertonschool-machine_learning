#! /usr/bin/env python3

def cat_arrays(arr1, arr2):
    """
    Concatenates two arrays
    """
    cat = arr1[:]
    for item in arr2:
        cat.append(item)
    return cat

