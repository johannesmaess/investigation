import numpy as np
import torch

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

