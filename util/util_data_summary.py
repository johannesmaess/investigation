import numpy as np
import torch
import matplotlib.pyplot as plt

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def nth_largest(n):
    def nth_largest_(x):
        return np.partition(x, -n)[-n]
    nth_largest_.__name__ = 'EV-%s' % n
    return nth_largest_

def heatshow(data, print_data=False, lim=None, ax=plt):
    if lim is None: lim = np.abs(data).max()
    if print_data: print(data[:, (np.abs(data).sum(axis=0) != 0)].round(2))
    return ax.imshow(data, vmin=-lim, vmax=lim, cmap="RdYlGn")