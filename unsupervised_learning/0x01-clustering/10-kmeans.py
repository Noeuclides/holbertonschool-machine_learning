#!/usr/bin/env python3
"""
Module to get kmeans with sklearn
"""
import sklearn.cluster


def kmeans(X, k):
    """
    performs K-means on a dataset:

    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: number of clusters
    Returns: C, clss
        - C: numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
        - clss: numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to
    """
    kmean = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    return kmean.cluster_centers_, kmean.labels_
