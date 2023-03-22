import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from util.util_pickle import load_data
from math import ceil, floor
from util.naming import *

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

def pretty_num(num):
    if num=='inf': return num
    if 0.01 < num <= 100: return f"{num:.3f}".rstrip('0').rstrip('.')                                       # floating
    return f"{num:.1e}".replace('.0e', 'e').replace('e+00', '').replace('e+0', 'e+').replace('e-0', 'e-')   # scientific

def plot_hists_on_ax(ax, vals_for_point, gammas, bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, .1, 1, 10, 100, 1e4, 1e5, 1e6, 1e7], max_val=1e7): 
    bins = np.array(bins)
    if max_val: bins = bins[bins <= max_val]
    assert len(bins.shape) == 1 and np.all(np.diff(bins) > 0), "Bin boundaries should be monotonically increasing"
    assert not np.any(vals_for_point < bins[0]),  "Bin Range starts too high"
    assert not np.any(vals_for_point > bins[-1]), "Bin Range ends too low"

    divider = make_axes_locatable(ax)
    bins = np.array([0] + list(bins[bins >= vals_for_point.min()]))
    x = np.arange(len(bins) - 1)
    ticks = np.arange(len(bins)) - .5
    
    # bar_lbls = ["$ < \sigma \leq$ " + pretty_num(bins[i+1]) for i in range(len(bins)-1)]
    tick_lbls = [pretty_num(b) for b in bins]

    for i_gamma, vals_for_gamma in enumerate(vals_for_point):
        counts, _ = np.histogram(vals_for_gamma, bins=bins)
        bars = ax.bar(x, counts)
        ax.bar_label(bars, padding=3)

        # plot gamma and data stats as text in top right corner
        ax.annotate(f"$\gamma=${pretty_num(gammas[i_gamma])}", xy=(0.85, 0.9), xycoords='axes fraction', fontsize=16)
        stats = f'Mean: {vals_for_gamma.mean():.1f}\n95th Perc: {np.percentile(vals_for_gamma, 95).round(1)}\nStdv: {vals_for_gamma.std():.1f}'
        ax.annotate(stats, xy=(0.85, 0.7), xycoords='axes fraction')

        ax.axvline(np.argmax(bins >= 1)-.5, color="green")

        # add another split to the plot (except after the last iteration):
        if i_gamma == len(vals_for_point)-1: 
            # ax.set_xticks(x, bar_lbls)
            ax.set_xticks(ticks, tick_lbls)
        else:
            ax.set_xticks([])
            ax = divider.append_axes("bottom", size="100%", pad=0, sharey=ax)

def prep_data(vals, gammas=None, norm_g0=False, norm_s1=False, end_at_0=False, dice=()):
    if type(vals) == tuple:   vals = load_data(*vals); assert vals is not False, "Could not load pickleid"
    else:                     vals = vals.copy()
    if gammas is None:        gammas = match_gammas(vals)

    assert not (norm_s1 and norm_g0)

    if end_at_0:
        vals = vals - vals[:, :, -1:, :]
    
    if norm_s1: 
        vals /= vals[:, :, :, :1]
        
    # divide every (n-th singular) value by (the n-th singular value at gamma=0)
    if norm_g0: 
        vals /= vals[:, :, :1, :]

    # take dice of svals and gammas
    if dice != ():
        selectors = [slice(0, None)] * 4
        for i, slic in enumerate(dice): 
            if   type(slic) == list:  selectors[i] = slic
            elif type(slic) == int:   selectors[i] = slice(0, slic)
            elif type(slic) == tuple:
                if slic == (): continue # select everything
                selectors[i] = slice(*slic)
            else: assert False, f"Invalid dice: {dice}"

        vals = vals[selectors[0], selectors[1], selectors[2], selectors[3]]
        gammas = gammas[selectors[2]]

    return vals, gammas

def distribution_plot(vals, gammas=None, dice=(),
                      mode='hist', cutoff = 1e-2, aggregate_over=None, agg=False,
                      # divide every spectra by its first singular value
                      norm_s1=False, 
                      # divide every (n-th singular) value by (the n-th singular value at gamma=0)
                      norm_g0=False,
                      **hist_kwargs):
    """
    Plots the distribution of Eigenvalues/Singularvalues 
        - per weight
        - per datapoint
        - per gamma
        - within user-defined bins.

    cutoff is the lower bound for Vals to be shown. 
    The Val is ommitted for all gammas, if it is not > cutoff for at least one gamma.
    """

    vals, gammas = prep_data(vals, gammas, norm_g0, norm_s1, dice=dice)
    # if norm_s1:  ylabel='$\\frac{ \sigma_i(\gamma) }{ \sigma_1(\gamma) }$'
    # if norm_g0:  ylabel='$\\frac{ \sigma_i(\gamma) }{ \sigma_i(0) }$'

    lbl_point=None
    sharey='row'
    if aggregate_over=='points' or agg:
        n_trans, n_point, n_gamma, n_vals = vals.shape
        vals = vals.transpose((0, 2, 3, 1)).reshape((n_trans, 1, n_gamma, n_point * n_vals))
        lbl_point=f"[{0}, {n_point-1}]"
        sharey=False

    n_trans, n_point, n_gamma, _ = vals.shape

    # order plots horizontally. and additionally vertically, if multiple LRP-transformation & reference points  are to be passed.
    if n_point==1:
        n_ax = (1, n_trans) 
        sharey = False
    else:
        n_ax = (n_trans, n_point)
    figsize = [10*n_ax[1], 4*n_ax[0] * (0.9*n_gamma)**(mode=='hist')]

    fig, axs = plt.subplots(*n_ax, figsize=figsize, sharey=sharey)
    axs = np.array(axs).reshape((n_trans, n_point)) # for correct loop iteration

    hist_kwargs['max_val'] = vals.max() * 20

    for i_trans, vals_for_trans in enumerate(vals):
        for i_point, vals_for_point in enumerate(vals_for_trans):
            if cutoff: # show only those Svals, that had a relevant value for gamma=0
                mask = np.any(vals_for_point > cutoff, axis=0)
                vals_for_point = vals_for_point[:, mask]

            ax = axs[i_trans, i_point]
            ax.title.set_text(f"w{i_trans} p{lbl_point or i_point}. Distribution of {vals_for_point.shape[1]} non-zero Singular values for {n_gamma} gammas.")

            if mode=='hist':       
                plot_hists_on_ax(ax, vals_for_point, gammas, **hist_kwargs)
            elif mode in ['box', 'violin']:
                if mode=='box':    ax.boxplot(vals_for_point.T)
                if mode=='violin': ax.violinplot(vals_for_point.T)

                ax.set_yscale('log')
                ax.set_ylim(max(1e-9, cutoff/1.5), vals.max()*1.5)

                ax.set_xlabel("$\gamma$")
                ax.set_xticks(1+np.arange(len(gammas)), gammas)

                ax.axhline(1, color="green")
            else: raise Exception(f"Invalid mode: {mode}")

    plt.subplots_adjust(wspace=0.1, hspace=0.16) 
    plt.show()



### Visualization of LRP Gridsearch results
def gridlrp_load_results(batch_tags, e_tag_filter=''):
    # load data
    loaded = [load_data('d3', tag) for tag in tqdm(batch_tags)]
    gs = np.vstack([gs for gs, _, _ in loaded])
    es = np.vstack([es for _, es, _ in loaded])
    es_tag = np.array(loaded[0][2])

    # order by second, then first column:
    order = np.argsort(gs[:, 1] * 1e4 + gs[:, 0] * 1e-4)
    gs_ordered = gs[order]
    es_ordered = es[order]
    # check order 
    increasing_or_0 = np.diff(gs_ordered, axis=0) >= 0
    increasing_or_0[gs_ordered[1:] == 0] = True
    assert np.all(increasing_or_0)
    
    if e_tag_filter:
        mask = np.array([e_tag_filter in tag for tag in es_tag])
        es_ordered = es_ordered[:, mask]
        es_tag = es_tag[mask]
    
    return gs_ordered, es_ordered, es_tag

def gridlrp_print_best_and_worst(es, es_tag):
    for i, tag in enumerate(es_tag):
        best_idx, worst_idx = np.argmax(-es_ordered[:, i]), np.argmin(-es_ordered[:, i])
        gs_best, e_best = gs_ordered[best_idx], es_ordered[best_idx, i]
        gs_worst, e_worst = gs_ordered[worst_idx], es_ordered[worst_idx, i]
        print(f'{tag}')
        print('\tBest\t', gs_best, e_best)
        print('\tWorst\t', gs_worst, e_worst)

def gridlrp_plot_metric_terrain(gs, e_flat, e_tag, ax, log=False, ylim=None, xlim=None, n_contours=11):
    if log:
        # filter out 0s
        mask = np.logical_not(np.any(gs == 0, axis=1))
        gs = gs[mask]
        e_flat = e_flat[mask]

    g1s, g2s = np.unique(gs[:, 0]), np.unique(gs[:, 1])
    
    e = e_flat.reshape((len(g2s), len(g1s)))
    plt.colorbar(ax.pcolormesh(g1s, g2s, e), ax=ax)
    if n_contours:
        levels = np.quantile(e, np.linspace(0, 1, n_contours))
        ax.contour(g1s, g2s, e, levels=levels, cmap='Greys')

    best_idx, worst_idx = np.argmin(e_flat), np.argmax(e_flat)
    gs_best,  e_best  = gs[best_idx],  e_flat[best_idx]
    gs_worst, e_worst = gs[worst_idx], e_flat[worst_idx]
    ax.plot(*gs_best,  marker='o', ms=15, c='g')
    ax.plot(*gs_worst, marker='o', ms=15, c='r')
    
    ax.set_aspect('equal')
    ax.set_xlabel('gamma early')
    ax.set_ylabel('gamma late')
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    if xlim is None: xlim = g1s.min(), g1s.max()
    if ylim is None: ylim = g2s.min(), g2s.max()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(e_tag)


def gridlrp_plot_metric_surface(gs, e_flat, e_tag, ax):

    g1s, g2s = np.unique(gs[:, 0]), np.unique(gs[:, 1])

    e = e_flat.reshape((len(g2s), len(g1s)))
    # plt.colorbar(ax.pcolormesh(g2s, g1s, e.T))
    
    g2s_tick = np.arange(len(g2s))
    g1s_tick = np.arange(len(g1s))
    
    x = np.outer(g2s_tick, np.ones_like(g1s))
    y = np.outer(g1s_tick, np.ones_like(g2s)).T
    
    ax.plot_surface(x, y, e.T, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_title(e_tag)
    
    ax.set_aspect('equal')
    ax.set_ylabel('gamma early')
    ax.set_xlabel('gamma late')
    
    plot_every=3
    ax.set_yticks(g2s_tick[::plot_every])
    ax.set_yticklabels(g2s[::plot_every].round(2))
    ax.set_xticks(g1s_tick[::plot_every])
    ax.set_xticklabels(g1s[::plot_every].round(2))
    
    ax.set_zlim(e.min(), e.max())
    ax.set_zscale('log')

def gridlrp_plot_metric_terrain_for_tags(batch_tags, log=False, in_3d=False, e_tag_filter=''):
    if in_3d: print("Warn: I didn't test the 3d surface plot implementation yet.") # TODO test the 3d surface plot implementation

    gs, es, es_tag = gridlrp_load_results(batch_tags=batch_tags, e_tag_filter=e_tag_filter)

    n, m = 1, len(es_tag)
    if m%3==0: n, m = 3, int(m/3)
    fig, axs = plt.subplots(m, n, figsize=(n*6, m*6+2), subplot_kw=({}, {"projection": "3d"})[in_3d])
    axs = np.array(axs).flatten()

    for e_flat, e_tag, ax in zip(es.T, es_tag, axs):
        if in_3d: gridlrp_plot_metric_surface(gs, e_flat, e_tag, ax, log=log)
        else:     gridlrp_plot_metric_terrain(gs, e_flat, e_tag, ax, log=log)

    axs[1].set_title(f"Terrain of {len(axs)} metrics. Lower (blue) is better. \n\n\n" + axs[1].get_title());


### Visualization of LLRP results (learning LRP by gradient descent)

def llrp_plot_training_for_tags(llrp_tags):
    n_lines_per_plot = 9

    n_cols = 1 + len(load_data('d3', llrp_tags[0])[4])
    n_cols = 1 + int(len(load_data('d3', llrp_tags[0])[4]) / n_lines_per_plot)
    n_rows = len(llrp_tags)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

    for tag, axs_row in zip(llrp_tags, axs):
        gs, gs_t, es, es_t, es_tag = load_data('d3', tag)

        axs_row[0].plot(gs_t, gs)
        axs_row[0].set_title(tag[58:])

        for i, ax in enumerate(axs_row):
            if i==0: continue
            ax.sharey(axs[0][i])

        for i, (e, e_tag) in enumerate(zip(np.array(es).T, es_tag)):
            ax = axs_row[1 + int(i/n_lines_per_plot)]
            ax.plot(es_t, e, label=e_tag)
            ax.legend()

    plt.show()

### More clever visualizations of Svals ### 

def plot_sval_func(vals, gammas=None, dice=(),
                           sval_func=lambda pvals: pvals[:, 0] / pvals[:, -1], # by default, we compute the condition number 
                           minima = False # in this mode, we don't plot the lines for every gamma, but just summarize at which gammas the line's minima are, in boxplots.
                           ):
    """
    Plot the gamma that maximises the fraction between the last to first singular value.
    """

    vals, gammas = prep_data(vals, gammas, dice=dice)

    # calculate which gamma yields the highest ratio sval_min / sval_max
    res=[]

    for wvals in vals[:]:                               # wwavls contain: per weight, per point, per gamma, n singular values
        res.append([])
        for pvals in wvals:                             # pvals contain:              per point, per gamma, n singular values
            # filter for non-zero singular values
            # print("vals", pvals.shape, end=" -> ")
            pvals=pvals[:, np.any(pvals>0, axis=0)]     # pvals contain:              per point, per gamma, k<n singular values
            # print(pvals.shape)            
            func_vals = sval_func(pvals)

            if minima==True:
                gamma_idx = func_vals.argmin()
                g = gammas[gamma_idx]
                res[-1].append(1e8 if g=='inf' else g)
            else:
                res[-1].append(func_vals)

    res = np.array(res)

    if minima:
        plt.boxplot(res.T)
        plt.ylim((-.5,5))
        plt.xlabel("Matrix No.")
        plt.ylabel("$\gamma$")
    else:
        fig, axs = plt.subplots(1, len(res), figsize=(5*len(res), 6))
        for r, ax in zip(res, axs):
            ax.set_yscale('log')
            ax.set_xlabel("$\gamma$")
            ax.set_ylabel("$\\frac{\sigma_i(\gamma)}{\sigma_1(\gamma)}$")

            if np.any([g=='inf' for g in gammas]):
                print(gammas)
                ax.plot(r.T)
                ax.set_xticks(np.arange(len(gammas)))
                ax.set_xticklabels(gammas)
            else:
                ax.plot(gammas, r.T)
                ax.set_xscale('log')
                ax.set_xlim((1e-3, 1e3))

    plt.show()

def plot_condition_number(*args, percentile=0, **kwargs):
    if type(percentile) in (int, float):
        l_percentile, u_percentile = percentile, 1 - percentile
    else:
        l_percentile, u_percentile = percentile
        u_percentile = 1 - u_percentile
    
    assert l_percentile < u_percentile

    def sval_func(pvals):
        l_idx = (pvals.shape[1] - 1) * l_percentile
        u_idx = (pvals.shape[1] - 1) * u_percentile

        l_idx = int(floor(l_idx))
        u_idx = int(ceil(u_idx))
        
        return pvals[:, l_idx] / pvals[:, u_idx]
    
    return plot_sval_func(*args, sval_func=sval_func, **kwargs)

def plot_determinant(vals, gammas=None):
    """
    This fucntion uses the (not so established) notion of a "generalized determinant" for non-quadratic, and non-full rank matrixes: the product of all singular values that are non-zero.
    """

    vals, gammas = prep_data(vals, gammas, norm_s1=True)

    # calculate the product of all ( non-zero singular values / maximum singular value )
    vals[vals==0] = 1
    res = np.product(vals, axis=3)

    # calculate its change relative to start
    res /= res[:, :, :1]

    fig, axs = plt.subplots(1, len(res), figsize=(5*len(res), 6))

    for r, ax in zip(res, axs):
        ax.plot(gammas, r.T)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((1e-3, 1e3))
        ax.set_xlabel("$\gamma$")
        ax.set_ylabel("$\\frac{\sigma_i(\gamma)}{\sigma_1(\gamma)}$")
