import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from util.util_pickle import load_data
from math import ceil, floor
from util.naming import *
from util.common import *
from functools import partial
from scipy import stats

# differnetiation and smoothing methods
from scipy.spatial.distance import cdist
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from util.quantus import batch_auc
# from util.util_gamma_rule import plot_vals_lineplot


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

def dice_to_selectors(dice):
    selectors = [slice(0, None)] * 4
    for i, slic in enumerate(dice): 
        if   type(slic) in [list, np.ndarray]:  selectors[i] = slic
        elif type(slic) == int:                 selectors[i] = slice(0, slic)
        elif type(slic) == tuple:
            if slic == (): continue # select everything
            selectors[i] = slice(*slic)
        else: assert False, f"Invalid dice: {dice}"
    return selectors

def gaussian_filter_1d(size, sigma):
    """Create a 1D Gaussian filter."""
    # Create an array of size length, centered around zero
    x = np.linspace(-(size // 2), size // 2, size)

    # Create a 1D Gaussian filter with mean 0 and standard deviation sigma
    filter_ = stats.norm.pdf(x, 0, sigma)

    # Normalize the filter so that it sums to 1 (for proper convolution)
    return filter_ / np.sum(filter_)

def edw_interpolation(x, y, x_new, power=2):
    """
    Exponential Distance Weighting interpolation.

    Parameters
    ----------
    x : array-like
        1-D array of coordinates.
    y : array-like
        1-D array of values to interpolate.
    x_new : array-like
        1-D array of coordinates at which to interpolate.
    power : float, default=2
        The power parameter for the IDW.

    Returns
    -------
    y_new : array-like
        1-D array of interpolated values.
    """
    # Compute distances from x_new to each x
    distances = cdist(x_new.reshape(-1, 1), x.reshape(-1, 1), metric='euclidean')

    # Compute weights (inverse of distances to the power)
    weights = np.exp(-power * distances)

    # Identify the indices of the top two maximum weights for each row
    idx_top2 = np.argpartition(weights, -2, axis=1)[:, -2:]

    # Create a mask of the same shape as weights, with False at the positions of the top two weights
    mask = np.ones(weights.shape, dtype=bool)
    for i in range(weights.shape[0]):
        mask[i, idx_top2[i]] = False

    # Set weights that are not among the top two to zero
    weights[mask] = 0

    # Normalize weights
    weights /= weights.sum(axis=1, keepdims=True)
    weights[np.isnan(weights)] = 1

    # Compute weighted sum of y values
    y_new = np.dot(weights, y)

    return y_new

def calc_derivative(vals, gammas, along_axis=2, derivative_mode='filter', derivative=0, 
                    # savgol specific
                    polyorder=2, window_length=5,
                    # modifications
                    logspace=True
                    ):
    if derivative==0: return vals, gammas

    if derivative_mode=='filter': # compute derivative using discrete differentiation filter
        assert logspace, "derivative 'filter' only supported in logspace."
        # Define the convolution filter
        filter = np.array([-1, 0, 1])
        
        while derivative>0:
            vals = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='valid'), along_axis, vals)
            gammas = gammas[1:-1]
            derivative -= 1

    elif derivative_mode=='savgol':
        assert logspace, "derivative 'savgol' only supported in logspace."
        vals = np.apply_along_axis(lambda m: -savgol_filter(m, window_length, polyorder, deriv=derivative, mode='nearest'), along_axis, vals) # minus, because for some reason the results started to have a signflip...

    elif derivative_mode=='formula':
        assert along_axis==2, "derivative with 'formula' only on axis 2 supported."

        if logspace:
            mask0 = gammas > 0
            gammas = gammas[mask0]
            vals = vals[:, :, mask0, :]
            g = np.log(gammas)
        else:
            g = gammas
            
        if derivative == 1:
            # (y[i+1] - y[i]) / (x[i+1] - x[i])
            vals = (vals[:, :, 1:] - vals[:, :, :-1]) / (g[None, None, 1:, None] - g[None, None, :-1, None])
            gammas = gammas[1:]

        if derivative == 2:
            # https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
            y1 = vals[:, :, :-2]
            y2 = vals[:, :, 1:-1]
            y3 = vals[:, :, 2:]

            x1 = g[None, None, :-2, None]
            x2 = g[None, None, 1:-1, None]
            x3 = g[None, None, 2:, None]

            vals = 2*y1 / ((x2-x1)*(x3-x1)) \
                - 2*y2 / ((x3-x2)*(x2-x1)) \
                + 2*y3 / ((x3-x2)*(x3-x1))
            
            gammas = gammas[1:-1]
    else:
        assert 0, "invalid derivative mode: " + derivative_mode

    return vals, gammas
     
def prep_data(vals, gammas=None
              , norm_g0=False, norm_s1=False, end_at_0=False
              , mean=None, hmean=None
              , dice=()
              , derivative=0, derivative_logspace=True, derivative_mode=['filter', 'formula', 'savgol'][0]
              , conv_filter=None
              , approx_method=''
              , approx_logspace=True
              , approx_power=3, approx_size=3
              , gammas_dense=np.logspace(-5, 3, 100)
              , edw=False):
    if type(vals) == tuple:   vals = load_data(*vals); assert vals is not False, "Could not load pickleid"
    else:                     vals = vals.copy()
    if gammas is None: gammas = match_gammas(vals)
    else:              gammas = gammas.copy()

    # take dice of svals and gammas (part 1)
    if dice != ():
        selectors = dice_to_selectors(dice)
        vals = vals[selectors[0], selectors[1], :, selectors[3]]

    if approx_method:
        a,b,c,d = vals.shape

        # Create an array to store the interpolated vals
        vals_dense = np.zeros((a, b, len(gammas_dense), d))

        if approx_logspace:
            mask0 = gammas > 0
            gammas = gammas[mask0]
            vals = vals[:, :, mask0, :]

        gammas_old = np.log(gammas)       if approx_logspace else gammas
        gammas_new = np.log(gammas_dense) if approx_logspace else gammas_dense

        # Loop over all entries in the other dimensions
        for i in range(a):
            for j in range(b):
                for k in range(d):
                    vals_old = vals[i, j, :, k]
                    if approx_method=='polyfit': vals_new = np.polyval(np.polyfit(deg=approx_power, x=gammas_old, y=vals_old), gammas_new)
                    if approx_method=='edw':     vals_new = edw_interpolation(gammas_old, vals_old, gammas_new, power=approx_power)
                    if approx_method=='spline':  vals_new = UnivariateSpline(gammas_old, vals_old, k=approx_power)(gammas_new)
                    if approx_method=='loess':
                        # Perform LOESS smoothing on the gammas and vals for this entry
                        smooth_vals = lowess(vals_old, gammas_old, frac=approx_power)

                        # Interpolate the smoothed vals at the denser gamma values
                        vals_new = np.interp(gammas_new, smooth_vals[:, 0], smooth_vals[:, 1])
                    vals_dense[i, j, :, k] = vals_new
        
        gammas = gammas_dense
        vals = vals_dense

    # implement conv filter
    if conv_filter==True: conv_filter = gaussian_filter_1d(9, 1.5)
    if conv_filter is not None:
        # shorten gammas
        l = int(len(conv_filter) / 2)
        r = l + len(conv_filter) % 2 - 1
        gammas = gammas[l:-r]

        conv_operator = partial(np.convolve, v=conv_filter, mode='valid')
        vals = np.apply_along_axis(conv_operator, arr=vals, axis=2)

    # if passed, calculate nth derivative.
    vals, gammas = calc_derivative(vals, gammas, 
                                   derivative=derivative, along_axis=2, 
                                    # savgol specific:
                                    polyorder=approx_power, window_length=approx_size,
                                    # modifications
                                    logspace=derivative_logspace
                                    )

    assert not (norm_s1 and norm_g0)

    if end_at_0:
        vals = vals - vals[:, :, -1:, :]
    
    if norm_s1: 
        vals /= vals[:, :, :, :1]
        
    # divide every (n-th singular) value by (the n-th singular value at gamma=0)
    if norm_g0: 
        vals /= vals[:, :, :1, :]

    # take dice of svals and gammas (part 2)
    if dice != ():
        vals = vals[:, :, selectors[2], :]
        gammas = gammas[selectors[2]]
        
    
    assert (mean is None) or (hmean is None), "We can only calculate mean or hmean."

    # calculate mean over...
    if mean=='points': 
        vals = np.nanmean(vals, axis=1, keepdims=True)
    else:
        assert mean is None, "Only mean over points is supported so far."

    # calculate harmonic mean over...
    if hmean=='points': 
        if end_at_0: 
            vals[:, :, -1] = 1 # the entries for gamma=inf are end_at_0d to 0. The hmean can not be calculated for 0 entries.
            vals = np.clip(vals, a_min=1e-3, a_max=None)
        vals = stats.hmean(vals, axis=1, keepdims=True, nan_policy='omit')
        if end_at_0: 
            vals[:, :, -1] = 0 # We thus adopt the policy hmean(..., 0, ...) = 0.
    else:
        assert hmean is None, "Only hmean over points is supported so far."

    return vals, gammas

def distribution_plot(vals, gammas=None, dice=(),
                      mode='hist', cutoff = 1e-2, aggregate_over=None, agg=False,
                      # divide every spectra by its first singular value
                      norm_s1=False, 
                      # divide every (n-th singular) value by (the n-th singular value at gamma=0)
                      norm_g0=False,
                      show=False,
                      sharey=None,
                      uniform_spacing=False, # if True, don't consider the gamma values to position boxes/violins
                      showextrema=True, showmedians=True, # violin plot args
                      axs=None, # provide (the right amount of) axs to draw on
                      ticks=None,
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
    sharey_auto='row'
    if aggregate_over=='points' or agg:
        n_trans, n_point, n_gamma, n_vals = vals.shape
        vals = vals.transpose((0, 2, 3, 1)).reshape((n_trans, 1, n_gamma, n_point * n_vals))
        lbl_point=f"[{0}, {n_point-1}]"
        sharey_auto=False

    n_trans, n_point, n_gamma, _ = vals.shape

    # order plots horizontally. and additionally vertically, if multiple LRP-transformation & reference points  are to be passed.
    if n_point==1:
        n_ax = (1, n_trans) 
        sharey_auto = False
    else:
        n_ax = (n_trans, n_point)
    figsize = [10*n_ax[1], 4*n_ax[0] * (0.9*n_gamma)**(mode=='hist')]

    if axs is not None: fig = None
    else: fig, axs = plt.subplots(*n_ax, figsize=figsize, sharey=(sharey or sharey_auto))
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
                if mode=='box' and uniform_spacing==True:    
                    if ticks: print("non manual ticks supported with 'uniform_spacing'.")
                    ax.boxplot(vals_for_point.T)
                    ax.set_xticks(np.arange(len(gammas)), [pretty_num(g) for g in gammas])
                    
                    ax.set_ylim(max(1e-9, cutoff/1.5), vals.max()*1.5)
                    ax.axhline(1, color="green")
                    
                else: # uniform_spacing==False or mode=='violin' 
                    if False: # omit gamma=0
                        mask = gammas > 0
                        x = gammas[mask]
                        y = vals_for_point[mask].T
                        ax.set_xticks(np.log(x), [pretty_num(g) for g in x])
                            
                    else: # plot gamma=0 in the position of a small value
                        x = gammas.copy()
                        if x[0] == 0:
                            x[0] = x[1] / 10
                        y = vals_for_point.T
                        ax.set_xticks(np.log(x), [pretty_num(g) for g in gammas])
                            
                        
                    if mode=='violin': ax.violinplot(positions=np.log(x), dataset=y, \
                                                     showextrema=showextrema, showmedians=showmedians)
                    elif mode=='box':     ax.boxplot(positions=np.log(x),       x=y, showfliers= showextrema)

                    if ticks is not None:
                        ax.set_xticks(np.log(ticks))
                        ax.set_xticklabels([pretty_num(t) for t in ticks])

                ax.set_xlabel("$\gamma$")
                ax.set_yscale('log')
            else: raise Exception(f"Invalid mode: {mode}")

    plt.subplots_adjust(wspace=0.1, hspace=0.16) 
    
    if show:
        plt.show()
        
    return fig, axs



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

def plot_sval_func(vals, gammas=None, dice=(), alpha=.75,
                           sval_func=lambda pvals, gammas: (pvals[:, 0] / pvals[:, -1], gammas), # by default, we compute the condition number 
                           minima = False, # in this mode, we don't plot the lines for every gamma, but just summarize at which gammas the line's minima are, in boxplots.
                           xscale='log', yscale='log',
                           sharey=True,
                           **prep_data_kwargs,
                           ):
    """
    Plot the gamma that maximises the fraction between the last to first singular value.
    """
    vals, gammas = prep_data(vals, gammas, dice=dice, **prep_data_kwargs)

    # calculate which gamma yields the highest ratio sval_min / sval_max
    res, minis = [], []

    for wvals in vals[:]:                               # wwavls contain: per weight, per point, per gamma, n singular values
        res.append([])
        minis.append([])
        for pvals in wvals:                             # pvals contain:              per point, per gamma, n singular values
            # filter for non-zero singular values
            # print("vals", pvals.shape, end=" -> ")
            pvals=pvals[:, np.any(pvals>0, axis=0)]     # pvals contain:              per point, per gamma, k<n singular values
            # print(pvals.shape)            
            func_vals, func_gammas = sval_func(pvals, gammas)

            if minima==True:
                gamma_idx = func_vals.argmin()
                g = func_gammas[gamma_idx]
                if g=='inf': g = 1e8
                res[-1].append(g)

                minis[-1].append(func_vals.min())
            else:
                res[-1].append(func_vals)

    res = np.array(res)

    if minima:
        plt.boxplot(res.T)
        plt.ylim((-.5,5))
        plt.xlabel("Matrix No.")
        plt.ylabel("$\gamma$")

        return res, minis

    fig, axs = plt.subplots(1, len(res), figsize=(5*len(res), 6), sharex=True, sharey=sharey)
    for r, ax in zip(res, axs if len(res)>1 else [axs]):
        ax.set_yscale(yscale)
        ax.set_xlabel("$\gamma$")
        ax.set_ylabel("$\\frac{\sigma_i(\gamma)}{\sigma_1(\gamma)}$")

        ax.set_prop_cycle(color=[mpl.colormaps['tab20'](k) for k in np.linspace(0, 1, len(r))])

        if np.any([g=='inf' for g in func_gammas]):
            print(func_gammas)
            ax.plot(r.T)
            ax.set_xticks(np.arange(len(func_gammas)))
            ax.set_xticklabels(func_gammas)
        else:
            ax.plot(func_gammas, r.T, alpha=alpha)
            ax.set_xscale(xscale)

    ax.set_prop_cycle(None)  
    return res
    
def condition_number_for_point(pvals, gammas, percentile=0, derivative_kw={}):
    if type(percentile) in (int, float):
        l_percentile, u_percentile = percentile, 1 - percentile
    else:
        l_percentile, u_percentile = percentile
        u_percentile = 1 - u_percentile

    assert l_percentile < u_percentile
    l_idx = (pvals.shape[1] - 1) * l_percentile
    u_idx = (pvals.shape[1] - 1) * u_percentile

    l_idx = int(floor(l_idx))
    u_idx = int(ceil(u_idx))

    cond = pvals[:, l_idx] / pvals[:, u_idx]

    if derivative_kw: cond, gammas = calc_derivative(cond, gammas, along_axis=0, **derivative_kw)

    return cond, gammas

def condition_number(vals, percentile=0):
    vals, _ = prep_data(vals)
    return np.stack([np.stack([condition_number_for_point(pvals[:, np.any(pvals>0, axis=0)], percentile) for pvals in wvals]) for wvals in vals])

def plot_condition_number(*args, 
                          percentile=0, mode='lines minima dist', 
                          ticks=None, xlim=(1e-4, 1e3), ylim=(1, 1e5), 
                          kde_bw=.6, alpha=1.0,
                          cond_derivative=True,
                          **kwargs):
    upper_bound=1024.0
    
    kw = {} # kwargs for condition number derivative computation
    if not cond_derivative: print("warning: might be computing derivative of svals, before computing condition number.")
    if cond_derivative and 'derivative' in kwargs:
        kw['derivative'] = kwargs['derivative']
        if 'derivative_mode' in kwargs: kw['derivative_mode'] = kwargs['derivative_mode']
        if 'derivative_logspace' in kwargs:    kw['logspace'] = kwargs['derivative_logspace']

        kw['polyorder'] =     kwargs['approx_power'] if 'approx_power' in kwargs else 2
        kw['window_length'] = kwargs['approx_size']  if 'approx_size'  in kwargs else 9
        
        kwargs['derivative'] = 0 # disable derivative of svals

    sval_func = partial(condition_number_for_point, percentile=percentile, derivative_kw=kw)
    plot_func = partial(plot_sval_func, *args, sval_func=sval_func, alpha=alpha-.4, **kwargs)
    # if mode=='lines':
    #     return plot_func(minima=0)
    if mode=='minima':
        return plot_func(minima=1)
    if mode=='both':
        main = plot_func(minima=0)
        plt.show()
        mini = plot_func(minima=1)
        return main, mini
    
    print(mode)
    if 'lines' in mode:                     # modes: ['lines', 'lines minima', 'lines minima dist']
        gs, minis = plot_func(minima=1)
        plt.clf() # clear

        plot_func(minima=0)
        fig = plt.gcf()
        axs = fig.axes

        # labeling
        if ticks is None: ticks=ax.get_xticks()
        for ax in axs:
            ax.set_xlabel('$\gamma$')
            ax.set_ylabel('')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks)

        # minima distribution plots
        if 'minima' in mode:
            for i, (g, mini, ax) in enumerate(zip(gs, minis, axs)):
                colors = [mpl.colormaps['tab20'](k) for k in np.linspace(0, 1, len(mini))]
                ax.scatter(x=g, y=mini, marker='X', s=80, zorder=2, color=colors, alpha=alpha)
                # ax.violinplot(vert=False, dataset=g, showextrema=False, showmedians=False, widths=1)
                # ax.boxplot(vert=False, x=g, showfliers=False, showcaps=False)

                if 'print' in mode: 
                    mode_res = stats.mode(g, keepdims=False)
                    print(i, ':', mode_res.mode, ',\t# count:', mode_res.count)

                if 'dist' in mode:
                    height = 0.3
                    ax_histx = ax.inset_axes([0, -height, 1, height])

                    ax.set_xticks([])
                    ax_histx.set_xticks(np.log10(ticks))
                    ax_histx.set_xticklabels(ticks)

                    ax_histx.set_yticks([])
                    ax.set_xlabel('')
                    ax_histx.set_xlabel('$\gamma$')
                    
                    ax.set_ylim(ylim)
                    ax.set_xlim(xlim)
                    ax_histx.set_xlim(np.log10(xlim))
                    
                    if i==0:
                        lbl = "$\mathrm{argmin}_\gamma \kappa_q$"
                        ax_histx.set_ylabel(lbl, rotation=0, fontsize=20, ha='right')
                        lbl = "Dist. of\n minima"
                        ax_histx.set_ylabel(lbl, rotation=0, fontsize=12, ha='right')

                    # the kde-plot suggests that most condition numbers are maximised at gamma=1024.0
                    # in reality, they are maximised at gamma=inf, or a different larger value
                    # increase the value such that the hump is not visible in the kde-plot
                    n = (g == upper_bound).mean()
                    xy=(.98, 0.65)
                    xy=(.5, .08)
                    if n: ax_histx.annotate(f"*{n:0.0%} of lines are\nminimized by $\gamma=\infty$", xy=xy, xycoords='axes fraction', fontsize=10, ha='center')
                    
                    g[g == upper_bound] *= 1e14
                    with HiddenPrints(): sns.kdeplot(x=np.log10(g), bw_method=kde_bw, ax=ax_histx, fill=True)

        fig.set_tight_layout(True)
        fig.set_figheight(6)

        if type(percentile) in (int, float): u_percentile = 1 - percentile
        else:                                u_percentile = 1 - percentile[1]

        lbl = '$\kappa_{' + str(u_percentile) + '}(\gamma)$'
        lbl = '$\kappa_q$'
        axs[0].set_ylabel(lbl, rotation=0, fontsize=20, ha='right')

        return fig, axs

    

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


### Utility functions to read from disk and summarize (AUC mean etc) PifxeFlipping measuremens

import functools

# y_batch = target.detach().numpy()
# x_batch =   data.detach().numpy().reshape((100, 1, 28, 28))

# Decorate the function with lru_cache
@functools.lru_cache(maxsize=100)
def pf_main(baseline, mode='individual', gammas='gammas40', which='test'):
    print("Rerunning...")

    if which=='test first 100': which=''
    elif which=='test': which='__testset'
    else: assert 0

    key = f'PixFlipScores{which}__{baseline}__{mode}_gamma__{gammas}'
    # key = 'PixFlipScores_gamma_0_1_21_inf'
    batch_scores = load_data('d3', key)
    assert batch_scores, f"invalid key: {key}"
    
    if gammas=='gammas40': gammas = gammas40
    elif gammas=='gamma_0_1_21_inf': gammas = gammas_0_1_21_inf
    elif gammas=='gammas80': gammas = gammas80
    else: assert 0, "invalid gammas"

    print('shape:', np.array(batch_scores['LRP-0']['PixFlip']).shape)

    experiment_dict = {}

    for mode_str, mode_dict in batch_scores.items():
        experiment_dict[mode_str] = {}
        for method_str, method_scores in mode_dict.items():
            if 'AUC' in method_str: continue
            auc = batch_auc(method_scores)
            experiment_dict[mode_str][method_str + ' AUC (per sample)'] = auc
            experiment_dict[mode_str][method_str + ' AUC (batch mean)'] = np.mean(auc)
            experiment_dict[mode_str][method_str + ' AUC (batch median)'] = np.median(auc)

        # experiment_dict[mode_str]['Layerwise Relevancies'] = relevancies_per_mode[mode_str]
        
    # experiment_dict['x_batch'] = x_batch
    # experiment_dict['y_batch'] = y_batch


    ### prep plotting

    k = len(next(iter(experiment_dict.values()))['PixFlip AUC (per sample)'])

    aucs =        np.zeros((len(d3_after_conv_layer), k, len(gammas), 1))
    auc_means =   np.zeros((len(d3_after_conv_layer), 1, len(gammas), 1))
    auc_medians = np.zeros((len(d3_after_conv_layer), 1, len(gammas), 1))

    layer_prefixes = {}
    for full_key in experiment_dict.keys():
        if 'Gamma.' in full_key:
            layer_prefix = full_key.split('gamma=')[0]
            layer_int = int(layer_prefix.split('<')[1])
            if layer_int not in layer_prefixes:
                layer_prefixes[layer_int] = layer_prefix

    layer_prefixes = dict(sorted(layer_prefixes.items())).values() # sort by layer integer, then keep only the strings.
    print(layer_prefixes)

    for i, lpref in enumerate(layer_prefixes):
        modes_sorted = dict()
        if 0 in gammas: modes_sorted[0] = 'LRP-0'
        for mode_str in experiment_dict.keys():
            if lpref in mode_str:
                g = float(mode_str.split('gamma=')[1])
                modes_sorted[g] = mode_str

        # assert len(modes_sorted) == len(gammas), f"{len(modes_sorted)} != {len(gammas)}"
        if len(modes_sorted) != len(gammas):
            print(f"{len(modes_sorted)} != {len(gammas)}")

        modes_sorted = dict(sorted(modes_sorted.items())).values() # sort by gamma float, then keep only the strings.
        if i==0: print(modes_sorted)
        
        for j, mode_str in enumerate(modes_sorted):
            aucs       [i, :, j, 0] = experiment_dict[mode_str]['PixFlip AUC (per sample)']
            auc_means  [i, 0, j, 0] = experiment_dict[mode_str]['PixFlip AUC (batch mean)']
            auc_medians[i, 0, j, 0] = experiment_dict[mode_str]['PixFlip AUC (batch median)']

    return experiment_dict, aucs, auc_means, auc_medians

def pf_auc(baseline, **kwargs):
    return pf_main(baseline, **kwargs)[1]

def pf_auc_mean(baseline, **kwargs):
    return pf_main(baseline, **kwargs)[2]

def pf_auc_median(baseline, **kwargs):
    return pf_main(baseline, **kwargs)[3]

def pf_auc_quantile(baseline, quantiles, **kwargs):
    if quantiles is int or quantiles is float: quantiles = [quantiles]
    aucs = pf_auc(baseline, **kwargs)
    quantiles = np.quantile(aucs, quantiles, axis=1, keepdims=True).transpose((4,1,2,3,0))[0]
    return quantiles

### Plotting functions/wrappers for Condition numbers and PF-AUC
ticks = [5e-4, 5e-3, 5e-2, .25, 1, 4, 16, 64]
def plot_cond_casc(gammas_str="gammas40", **kwargs):
    kwargs.setdefault('dice', ((), (40,80)))
    kwargs.setdefault('xlim', (1e-4, 512))
    kwargs.setdefault('ylim', (1.4, 5e2))
    kwargs.setdefault('kde_bw', .4)
    kwargs.setdefault('ticks', [1e-4, .01, .25, 1, 4, 32, 512])
    return plot_cond(key='svals__m0_to_1__cascading_gamma__'+gammas_str, **kwargs)

def plot_cond_ind(gammas_str="gammas40", **kwargs):
    kwargs.setdefault('kde_bw', .4)
    kwargs.setdefault('ylim', (9, 2e6))
    if 'gammas400' in gammas_str:
        kwargs.setdefault('xlim', (0, 10))
        kwargs.setdefault('ticks', np.arange(11))
    else:
        kwargs.setdefault('xlim', (3e-6, 1e3))
    kwargs.setdefault('ticks', [1e-4, .01, .25, 1, 4, 32, 512])
    return plot_cond(key='svals__individual_layer__'+gammas_str, **kwargs)

def plot_cond_ind_gamma(gammas_str="gammas40", **kwargs):
    kwargs.setdefault('dice', (5,))
    kwargs.setdefault('ylim', (3, 3e2))
    kwargs.setdefault('xlim', (3e-6, 1e3))
    kwargs.setdefault('kde_bw', .4)
    kwargs.setdefault('ticks', [1e-4, .01, .25, 1, 4, 32, 512])
    return plot_cond(key='svals__m0_to_1__individual_gamma__'+gammas_str, **kwargs)

def plot_cond(key, hline=0., **kwargs):
    kwargs.setdefault('percentile', (0, .05))
    kwargs.setdefault('mode', 'lines')
    kwargs.setdefault('alpha', 1)

    der = 'derivative' in kwargs and kwargs['derivative'] > 0
    if der and 'derivative_mode' not in kwargs:
        print("Automatically using savgol smoothing for derivative.")
        kwargs['derivative_mode'] = 'savgol'

    if 'gammas400' in key:
        kwargs.setdefault('xscale', 'linear')
    if 'gammas400' in key or 'gammas80' in key:
        kwargs.setdefault('yscale', 'linear')
        
    fig, axs = plot_condition_number(('d3', key), **kwargs)

    if der and type(hline) in [float, int]: 
        for ax in axs: ax.axhline(hline)
    return fig, axs

def plot_pf_auc(agg, mode, baselines, fig, axes, norm=False, which='test', dice=(5,),
        gammas='gammas40', legend="upper left", **kwargs):
    kwargs.setdefault('sharey', True)
    kwargs.setdefault('plot_kwargs', {'ls':'--', 'lw':2})
    kwargs.setdefault('mark_minima', True)
    kwargs.setdefault('print_minima', False)
    ylabel = f"PixelFlipping AUC, {agg} over 8400 test set points"
    tag_line = [f"PF-AUC. (Replace pixels with baseline '{b}')" for b in baselines]
    if agg=='mean':   agg_func = pf_auc_mean
    if agg=='median': agg_func = pf_auc_median
    if type(agg)is list: # compute quantiles
        ylabel = f"PF-AUC, Quantile over 8400 test set points"
        agg_func = partial(pf_auc_quantile, quantiles=agg)
        tag_line = [f"PF-AUC, Q:{q} (Replace pixels with baseline '{b}')" for b in baselines for q in agg]

    if len(tag_line)==1: 
        ylabel += f"\n(Replace pixels with baseline '{baselines[0]}')"
        legend=False
    if legend==False: 
        tag_line=None

    # overwrite!
    ylable=None
    legend=False
    tag_line=None

    auc_agg = np.concatenate([agg_func(baseline, mode=mode, gammas=gammas, which=which) for baseline in baselines], axis=3)

    auto_ylim = None
    if norm:
        auc_agg -= auc_agg.min(axis=(1,2), keepdims=True)
        auc_agg /= auc_agg.max(axis=(1,2), keepdims=True)
        auto_ylim=(0,1)
    elif kwargs['sharey']:
        auto_ylim=(auc_agg.min(), auc_agg.max())
    
    kwargs.setdefault('ylim', auto_ylim or 'p100')
    # kwargs.setdefault('ax_func', lambda ax: ax.tick_params(axis='y', colors=purple, weight='bold'))
    kwargs.setdefault('ax_func', lambda ax: ax.tick_params(axis='y', colors=purple) or [label.set_fontweight('bold') for label in ax.get_yticklabels()])

    print(f"AUC. ylim: {kwargs['ylim']}, sharey {kwargs['sharey']}")
    plot_vals_lineplot(auc_agg, dice=dice, axes=axes,
                        xscale='log', yscale='linear', tag_line=tag_line,
                            ylabel=ylabel, legend=legend, **kwargs)
    
    fig.axes[-1].set_ylabel('PixelFlipping AUC', fontsize=SMALL_FS, color=purple)

def prettify_plot(fig, axs, mode_cond=False, mode_pf=False):
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.12)

    # the first plot weirdly has double x ticks, not sure where its coming from
    # we get to it by getting the first ax that is not already stored in axs.
    n = len(axs)
    fig.get_axes()[n].set_yticks([])

    if   mode_cond=='individual':       annotate_axs_individual(axs)
    elif mode_cond=='individual_gamma': annotate_axs_d3_individual_gamma(axs, n_expected=n, left=True)
    elif mode_cond=='cascading':        annotate_axs_d3_cascading(axs, left=True)
    else: 
        assert mode_cond==False, "Invalid mode_cond: " + mode_cond

    if   mode_pf=='individual_gamma': annotate_axs_d3_individual_gamma(axs, n_expected=n, pf=True)
    elif mode_pf=='cascading':        annotate_axs_d3_cascading(axs, pf=True)
    else: 
        assert mode_pf==False, "Invalid mode_pf: " + mode_pf




def plot_vals_lineplot(vals, gammas=None,
                mark_positive_slope=False, plot_only_non_zero=False, one_plot_per='weight',
                num_vals_largest=None, num_vals_total=None,
                ylabel="Singular values", title=None,
                ylim=4, xlim=None,
                yscale='linear', xscale='linear', sharey=False, xtick_mask=None,
                figsize=None, show=False, 
                green_line_at_x=None, tag_line=None, 
                colormap='viridis',
                # give location of legend
                legend=False, 
                # divide every spectra by its first singular value
                norm_s1=False, 
                # divide every (n-th singular) value by (the n-th singular value at gamma=0)
                norm_g0=False,
                # substract from every (n-th singular) value (the n-th singular value at gamma=inf)
                end_at_0=False,
                # spectra mode: one line represents one spectra. one spectra per gamma.
                spectra=False, 
                # dice: restrict which weights, points, gammas, and svals get plotted.
                dice=(),
                       
                # compute means to simplify plot
                mean=None, hmean=None,
                
                # put the matrices c value as text in the top right corner
                annotate_c=None, # pass c values in same form as data

                # option to pass axs (such that you can plot data on top of other plot)
                axes = None, twinx=True,

                plot_kwargs={},
                ax_func=(lambda ax: 0),

                print_minima=False,
                mark_minima=False,
                ):
    """
    Plots the evolution of Eigenvalues with increasing gammas in a lineplot.
    """

    if spectra:
        ylabel='$\sigma_i(\gamma)$'
        xlabel = '$i$'
    else:
        xlabel = '$\gamma$'

    vals, gammas = prep_data(vals, gammas, norm_g0=norm_g0, norm_s1=norm_s1, end_at_0=end_at_0, \
                             mean=mean, hmean=hmean, \
                             dice=dice)
    selectors = dice_to_selectors(dice)
    if end_at_0 and yscale=='log': vals += 1e-5

    if norm_s1:  ylabel='$\\frac{ \sigma_i(\gamma) }{ \sigma_1(\gamma) }$'
    if norm_g0:  ylabel='$\\frac{ \sigma_i(\gamma) }{ \sigma_i(0) }$'

    # plot one line per spectra
    if spectra:
        vals = np.transpose(vals, (0,1,3,2))
        vals = vals[:, :, np.any(vals, axis=(0,1,3)), :]

    # reduce number of eval lines to show to first n.
    assert not (num_vals_largest and num_vals_total)
    if num_vals_largest:
        vals = vals[:, :, :, :num_vals_largest]

    if type(xlim) == int:
        x_lim_lower = {'linear':0, 'log': max(1e-3, gammas[0])*.9}[xscale]
        xlim = [x_lim_lower, xlim]
    elif type(xlim) == tuple:
        xlim = list(xlim)
    elif type(xlim) != list and xlim is not None:
        print(f'Warn: Invalid xlim: {xlim}. Setting xlim to None.')
        xlim = None
    
    if spectra: pass
    elif xlim and 0:
        mask = np.logical_and(gammas >= xlim[0], gammas <= xlim[1])
        gammas = gammas[mask]
        vals = vals[:, :, mask]
        if xtick_mask: xtick_mask = xtick_mask[mask]

    percentile_to_plot = None
    if type(ylim) == str and ylim[0] == 'p': # we passed a percentile code like "p99" -> show at least 99 percentiles of every line
        percentile_to_plot = float(ylim[1:])
        ylim = None
    elif type(ylim) == int:
        y_lim_lower = {'linear':0, 'log': max(1e-3, vals.min())*.9}[yscale]
        ylim = [y_lim_lower, ylim]
    elif type(ylim) == tuple:
        ylim = list(ylim)
    elif type(ylim) != list and ylim is not None:
        print(f'Warn: Invalid ylim: {ylim}. Setting ylim to None.')
        ylim = None

    if one_plot_per=='in total': n_ax = (1,1)
    elif one_plot_per=='weight': n_ax = (1, vals.shape[0])
    elif one_plot_per=='point': 
        if vals.shape[0] == 1:   n_ax = (1, vals.shape[1])
        else:                    n_ax = (vals.shape[1], vals.shape[0])

    if figsize is None: 
        figsize = (20, 10) if n_ax==(1,1) else (5*n_ax[1], 3*n_ax[0])
    if axes:         
        if twinx:
            axs = np.vectorize(lambda ax: ax.twinx())(axes)
        else:
            axs = axes
    else:       
        fig, axs = plt.subplots(*n_ax, figsize=figsize, sharey=sharey)
    axs, ax_i, ax = np.array(axs).T.flatten(), -1, None
    assert len(axs) == np.prod(n_ax), "passed axs do not match passed data."

    if not axes:
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle(f'Evolution of {ylabel} with increasing $\gamma$' +
                    ('\nFat bar below indicates section of positive derivative' if mark_positive_slope else ''))

    ### helper functions ###
    def ax_init():
        nonlocal axs, ax, ax_i, xlabel, axes
        # iterate to next ax
        ax_i += 1
        ax = axs[ax_i]

        if axes and not twinx: return

        ax.set_yscale(yscale)

        if green_line_at_x is not None: ax.axvline(green_line_at_x, color="green")

        if axes: return
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
        
    def ax_show():
        nonlocal ax, xlim, ylim, legend, axes

        ax_func(ax)

        if (i==0 and not axes) or (i==len(axs)-1 and axes):
            ax.set_ylabel(ylabel)
        elif sharey==False:
            pass
        else:
            ax.set_ylabel('')
            ax.set_yticks([])

        if axes and not twinx: return

        if (not legend) and (tag_line is not None):
            legend='upper right'
        if legend:
            ax.legend(loc=legend)

        # set ylim
        ylim_u = ylim[1] * 1.1 + .1
        ylim_l = ylim[0] / 1.1
        if yscale=='linear': ylim_l -= .1
        ax.set_ylim((ylim_l, ylim_u))
        
        if percentile_to_plot: ylim = None

        if xlim is None: return
    
        # set xlim
        xlim_u = xlim[1] * 1.01 + .1
        xlim_l = xlim[0] / 1.01 - .1 if xscale=='linear' else xlim[0] / 1.1
        ax.set_xlim((xlim_l, xlim_u))

    ### preemptive checks ###
    if not spectra:
        assert np.all([[len(line) == len(gammas) for line in sub_list] for sub_list in vals]), "Shape doesn't match."

    if one_plot_per=='in total': ax_init()

    for i, per_points in enumerate(vals): # iterate matrices
        if one_plot_per=='weight': ax_init()

        for j, evals in enumerate(per_points): # iterate points
            if one_plot_per=='point': ax_init()
            if percentile_to_plot:
                pos_vals = evals[evals > 0] if (plot_only_non_zero or yscale=='log' or np.all(vals >= 0)) else evals
                pos_vals = pos_vals[np.logical_not(np.isnan(pos_vals))]
                
                # calculate lower and upper xlim, update if they are wider.
                l = np.percentile(pos_vals,                    (100-percentile_to_plot)/2)
                u = np.percentile(pos_vals, percentile_to_plot+(100-percentile_to_plot)/2)
                ylim = [l, u] if ylim is None else [min(ylim[0], l), max(ylim[1], u)]

            # reset color cycle
            ax.set_prop_cycle(None)
            # plot for this point
            if plot_only_non_zero:
                mask = np.any(np.abs(evals) > 1e-5, axis=0)
                evals = evals[:, mask]
                # print(f"W: {i}, p: {j}. {1-mask.mean():.0%} of lines are constantly zero and don't get plotted. Remaining:", mask.sum())
                ax.title.set_text(f"({i},{j}) {mask.sum()}/{np.prod(mask.shape)} lines are non-zero.")
            
            Y = evals  # + np.random.normal(0, .005, size=evals.shape[1])[None, :] # add some random noise, such that lines don't overlap.
            all_vals_nan = np.any((Y != 0)*1 - np.isnan(Y), axis=0)
            Y = Y[:, all_vals_nan]

            if num_vals_total:
                indices = np.linspace(0, Y.shape[1]-1, num_vals_total).round().astype(int)
                indices = np.unique(indices)
                # print(f"Reducing num of lines from: {Y.shape[1]} to {len(indices)}")
                Y = Y[:, indices]
            else:
                indices = np.arange(Y.shape[1])
            
            if colormap is not None:
                # count number of lines that have non-zero, non-nan elements in them
                num_colors = np.sum(np.any((Y != 0)*1 - np.isnan(Y), axis=0))
                ax.set_prop_cycle(mpl.cycler('color', [mpl.colormaps[colormap](k) for k in np.linspace(0, 1, num_colors)]))  

            if spectra: 
                Y = Y[np.any(Y, axis=1)]
                ax.plot(Y, label=gammas, **plot_kwargs)

            else:
                # labels for legend
                labels = [f'Exp. {i+1}, Point {j+1}, Sval {k+1}' for k in indices]
                labels = [f'Singular value {k+1}' for k in indices] # <- prettier, for Proposal plot
                if tag_line is not None:
                    tags = np.array(tag_line)
                    if tags.ndim == 0: 
                        assert 1 == len(labels)
                    else:
                        tags = tags[selectors[3]]
                        assert len(tags) == len(labels), "Invalid labels per line passed."
                    labels = tags

                # If gammas are numerical, use them to determine x position of lines. If they are strings, plot evals in equal spacing, and label them with the 'gammas'
                if np.any([type(g) is str for g in gammas]):
                    xtick = np.arange(len(gammas))
                    if xtick_mask is None: xtick_mask = np.full_like(xtick, True, dtype=bool)

                    ax.set_xticks(xtick[xtick_mask])
                    lbls = gammas
                    lbls = [pretty_num(lbl) for lbl in lbls]
                    lbls = np.array(lbls)[xtick_mask]
                    ax.set_xticklabels(lbls)
                else:
                    xtick=gammas

                ax.plot(xtick, Y, label=labels, **plot_kwargs)
                
                x_mini = xtick[Y.argmin(axis=0)]
                y_mini = Y.min(axis=0)
                if print_minima:
                    print(i, ':', x_mini[0], ',\t# value:', y_mini[0])
                if mark_minima:
                    ax.scatter(x=x_mini, y=y_mini, marker='X', s=150, zorder=2)


            if mark_positive_slope: # plot a scatter dot if the series values is increasing
                # calc sign of derivative
                is_positive = np.diff(evals, axis=0) > 0
                # reset color cycle
                ax.set_prop_cycle(None)

                for k, is_pos, label in zip(range(100), is_positive.T, [f'Point {j+1}, EV {k+1} (Increasing segment)' for k in range(evals.shape[1])]):
                    x = gammas[:-1][is_pos]
                    y = np.full_like(x, -.2 -.05*k - .15*j)
                    ax.scatter(x,y, s=5)

            if one_plot_per=='point': ax_show()
        if one_plot_per=='weight': ax_show()
    if one_plot_per=='in total': ax_show()
    
    if annotate_c is not None:
        cs = annotate_c[selectors[0], selectors[1]]
        for ax, c in zip(axs, cs.flatten()):
            annot = "$c=$" + f"{ c*100-99 :0.3f}".lstrip('0')
            ax.annotate(annot, xy=(0.75, 0.9), xycoords='axes fraction', fontsize=13)

    plt.subplots_adjust(hspace=0.3)
    if show:
        plt.show()
        
    if not axes:  
        return fig, axs


def plot_multiplicative_change(vals, gammas=None, hmean=False, end_at_0=False, **passed_kwargs):
    kwargs = {
        "ylabel": "Multiplicative change",
        "title": "Multiplicative change per Singular value. Shows which Singular values decrease fastest relative to their size. Blue are biggest Svals. Red smallest.",
        "yscale": "log",
        "ylim": 'p100',
        "sharey": False,
        "colormap": "seismic",
    }
    for k,v in passed_kwargs.items(): kwargs[k] = v
    # kwargs.update(passed_kwargs)
        
    with HiddenPrints():
        change, gammas = prep_data(vals, gammas, end_at_0=end_at_0, norm_g0=True, hmean=hmean)
    
    if end_at_0 and kwargs['yscale']=='log': change += 0.01
    
    return plot_vals_lineplot(change, gammas, **kwargs)
