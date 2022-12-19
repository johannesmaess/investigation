import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plot_hists_on_ax(ax, vals_for_point, bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, .1, 1, 10, 100, 1e4, 1e5, 1e6, 1e7], max_val=1e7): 
    bins = np.array(bins)
    if max_val: bins = bins[bins <= max_val]
    assert len(bins.shape) == 1 and np.all(np.diff(bins) > 0), "Bin boundaries should be monotonically increasing"
    assert not np.any(vals_for_point < bins[0]),  "Bin Range starts too high"
    assert not np.any(vals_for_point > bins[-1]), "Bin Range ends too low"

    divider = make_axes_locatable(ax)
    bins = [0] + list(bins[bins >= vals_for_point.min()])
    x = np.arange(len(bins) - 1)
    
    def pretty(num):
        if 0.01 < num <= 100: return f"{num:.3f}".rstrip('0').rstrip('.')                                       # floating
        return f"{num:.1e}".replace('.0e', 'e').replace('e+00', '').replace('e+0', 'e+').replace('e-0', 'e-')   # scientific
    bar_lbls = ["$ < \sigma \leq$ " + pretty(bins[i+1]) for i in range(len(bins)-1)]

    for i_gamma, vals_for_gamma in enumerate(vals_for_point):
        counts, _ = np.histogram(vals_for_gamma, bins=bins)
        bars = ax.bar(x, counts)
        ax.bar_label(bars, padding=3)

        ax_title = f'Mean: {vals_for_gamma.mean():.1f}\n95th Perc: {np.percentile(vals_for_gamma, 95).round(1)}\nStdv: {vals_for_gamma.std():.1f}'
        ax.annotate(ax_title, xy=(0.85, 0.8), xycoords='axes fraction')

        # add another split to the plot (except after the last iteration):
        if i_gamma == len(vals_for_point)-1: 
            ax.set_xticks(x, bar_lbls)
        else:
            ax.set_xticks([])
            ax = divider.append_axes("bottom", size="100%", pad=0, sharey=ax)


def distribution_plot(vals, gammas, mode='hist', cutoff = 1e-2, aggregate_over=None, agg=False, **hist_kwargs):
    """
    Plots the distribution of Eigenvalues/Singularvalues 
        - per weight
        - per datapoint
        - per gamma
        - within user-defined bins.

    cutoff is the lower bound for Vals to be shown. 
    The Val is ommitted for all gammas, if it is not > cutoff for at least one gamma.
    """
    lbl_point=None
    sharey='row'
    if aggregate_over=='points' or agg:
        n_trans, n_point, n_gamma, n_vals = vals.shape
        vals = vals.transpose((0, 2, 3, 1)).reshape((n_trans, 1, n_gamma, n_point * n_vals))
        lbl_point=f"[{0}, {n_point-1}]"
        sharey=False

    n_trans, n_point, n_gamma, _ = vals.shape

    # order plots horizontally. and additionally vertically, if multiple LRP-transformation & reference points  are to be passed.
    n_ax = (1, n_trans) if n_point==1 else (n_trans, n_point)
    figsize = [10*n_ax[1], 4*n_ax[0] * (0.9*n_gamma)**(mode=='hist')]

    fig, axs = plt.subplots(*n_ax, figsize=figsize, sharey=sharey)
    axs = np.array(axs).reshape((n_trans, n_point)) # for correct loop iteration

    # hist_kwargs['max_val'] = vals.max() * 10

    for i_trans, vals_for_trans in enumerate(vals):
        for i_point, vals_for_point in enumerate(vals_for_trans):
            if cutoff: # show only those Svals, that had a relevant value for gamma=0
                mask = np.any(vals_for_point > cutoff, axis=0)
                vals_for_point = vals_for_point[:, mask]

            ax = axs[i_trans, i_point]
            ax.title.set_text(f"w{i_trans} p{lbl_point or i_point}. Distribution of {vals_for_point.shape[1]} non-zero Singular values for {n_gamma} gammas.")

            if mode=='hist':       
                plot_hists_on_ax(ax, vals_for_point, **hist_kwargs)
            elif mode in ['box', 'violin']:
                if mode=='box':    ax.boxplot(vals_for_point.T)
                if mode=='violin': ax.violinplot(vals_for_point.T)

                ax.set_yscale('log')
                ax.set_ylim(max(1e-9, cutoff/1.5), vals.max()*1.5)

                ax.set_xlabel("$\gamma$")
                ax.set_xticks(1+np.arange(len(gammas)), gammas)
            else: raise f"Invalid mode: {mode}"

    plt.subplots_adjust(wspace=0.1, hspace=0.16) 
    plt.show()

