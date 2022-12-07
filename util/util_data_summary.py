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


def distribution_histogram(comp_vals, bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, .1, 1, 10, 100, 1e4], cutoff = 1e-2):
    """
    Plots the distribution of Eigenvalues/Singularvalues 
        - per weight
        - per datapoint
        - per gamma
        - within user-defined bins.

    cutoff is the lower bound for Vals to be shown. 
    The Val is ommitted for all gammas, if it is not > cutoff for at least one gamma.
    """
    bins = np.array(bins)
    assert len(bins.shape) == 1 and np.all(np.diff(bins) > 0), "Bin boundaries should be monotonically increasing"
    assert not np.any(comp_vals < bins[0]), "Bin Range starts too high"
    assert not np.any(comp_vals > bins[-1]), "Bin Range ends too low"

    for data_for_weight in comp_vals:
        num_points, num_gamma, _ = data_for_weight.shape
        fig, axs = plt.subplots(num_gamma, num_points, figsize=(10*num_points, 4*num_gamma))
        axs = np.array(axs).reshape((num_gamma, num_points))

        for i_point, data_for_point in enumerate(data_for_weight):
            if cutoff: # show only those Svals, that had a relevant value for gamma=0
                mask = np.any(data_for_point > cutoff, axis=0)
                data_for_point = data_for_point[:, mask]
                bins = [0] + list(bins[bins >= cutoff])
                
            for i_gamma, data_for_gamma in enumerate(data_for_point):
                (counts, bins) = np.histogram(data_for_gamma, bins=bins)
                ax = axs[i_gamma, i_point]

                x = np.arange(len(counts))
                bar_lbls = [f"$ < \sigma <$ {bins[i+1]:.1e}".replace('.0e', 'e').replace('e+00', '').replace('e+0', 'e+').replace('e-0', 'e-') for i in range(len(bins)-1)]

                ax.set_xticks(x, bar_lbls)
                ax.bar(x, counts)
                # ax.plot(x, counts)
                
        plt.suptitle(f"Distribution of Singular values.\nFor {len(axs[0])} data points (columns) and {len(axs)} gammas (rows).", fontsize=30)
        plt.show()


def distribution_boxplot(comp_vals, gammas, cutoff = 1e-2):
    """
    Plots the distribution of Eigenvalues/Singularvalues 
        - per weight
        - per datapoint
        - per gamma (visualized as multiple box's in one plot).

    cutoff is the lower bound for Vals to be shown. 
    The Val is ommitted for all gammas, if it is not > cutoff for at least one gamma.
    """

    for data_for_weight in comp_vals:
        num_points, num_gamma, _ = data_for_weight.shape
        fig, axs = plt.subplots(1, num_points, figsize=(10*num_points, 4), sharey=True)
        if num_points==1: axs=[axs]
        axs[0].set_ylabel("Singular values")

        for ax, data_for_point in zip(axs, data_for_weight):
            ax.set_yscale('log')
            ax.set_ylim(1e-9, 1000)

            if cutoff is not None: # show only those Svals, that had a relevant value for gamma=0
                mask = np.any(data_for_point > cutoff, axis=0)
                data_for_point = data_for_point[:, mask]
                
                ax.set_ylim(cutoff, 1000)

            # sns.violinplot(data_for_gamma, ax=ax)
            ax.boxplot(data_for_point.T)
            
            ax.set_xlabel("$\gamma$")
            ticks = [str(g) if g < 1e8 else '$\inf$' for g in gammas]
            ax.set_xticks(np.arange(len(ticks))+1, ticks)
                
        plt.show()