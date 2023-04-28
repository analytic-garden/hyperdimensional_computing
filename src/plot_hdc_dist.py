#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_hdc_dist.py - plot hdc distances for varying dimensions
author: Bill Thompson
license: GPL 3
copyright: 2023-04-26
"""
import numpy as np
import matplotlib.pyplot as plt
from hdc import hdv, cos
import seaborn as sns
import pandas as pd

def calc_hdv_distances(num_dimensions: int, num_samples: int = 1000) -> np.ndarray:
    """Calculate a matrix of hdv cosine similarities

    Parameters
    ----------
    num_dimensions : int
        size of the hdvs
    num_samples : int, optional
        number of random hdvs to sample, by default 1000

    Returns
    -------
    np.ndarray
        a matrix of cosine similarities. dist[i,j] -> cos(hdvs[i], hdvs[j])
    """
    # create a bunch of hdvs
    hdvs = np.zeros(num_samples, dtype = object)
    for i in range(num_samples):
        hdvs[i] = hdv(N = num_dimensions)

    # get distances
    dist = np.zeros((num_samples, num_samples))
    for i in range(num_samples-1):
        dist[i, i] = 1
        for j in range(i+1, num_samples):
            dist[i, j] = cos(hdvs[i], hdvs[j])
            dist[j, i] = dist[i, j]

    return dist

def main():
    dist10 = calc_hdv_distances(10)          # 10d
    dist10_000 = calc_hdv_distances(10_000)  # 100000d

    # plot off diagonal elements
    df = pd.DataFrame({"dim: 10": np.ndarray.flatten(dist10), 
                       "dim: 10_000": np.ndarray.flatten(dist10_000)})
    sns.histplot(data = df, bins = 30)
    plt.xlabel("cos")
    plt.show()

    # plot everything
    sns.heatmap(dist10)
    plt.title("dim: 10")
    plt.show()

    sns.heatmap(dist10_000)
    plt.title("dim: 10_000")
    plt.show()

if __name__ == "__main__":
    main()
