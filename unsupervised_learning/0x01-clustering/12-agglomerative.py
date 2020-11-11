#!/usr/bin/env python3
"""
Module to perform agglomerative clustering
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset:

    - X: numpy.ndarray of shape (n, d) containing the dataset
    - dist: maximum cophenetic distance for all clusters
    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color
    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster indices
    for each data point
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                            criterion="distance")

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    return clss
