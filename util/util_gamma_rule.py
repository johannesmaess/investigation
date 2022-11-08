from functools import partial
import numpy as np
import torch

from scipy.sparse import coo_array
from scipy.sparse.linalg import eigs

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


# TODO: maybe I should change this function to handle soarse matrices in Scipy, as it has more features than Sparse Pytorch.
def global_conv_matrix(conv, bias=None, img_shape=None, zero_padding=(0,0),
                        sparse_matrix=True):
    assert(img_shape and len(img_shape) == len(zero_padding))
    img_shape = np.array(img_shape)
    zero_padding = np.array(zero_padding)
    if bias: print("Warning: So far we currently do nothing with passed bias terms.")

    # this is the matrix that will represent the global convolution operation
    img_shape_padded = img_shape + 2*zero_padding
    img_padded_flattened_length = np.prod(img_shape_padded)

    img_flattened_length = np.prod(img_shape)

    res_shape = img_shape_padded - conv.shape + 1
    res_flattened_length = np.prod(res_shape)
    
    # this is gonna be the global convolution matrix
    if not sparse_matrix:
        trans = np.zeros((res_flattened_length, img_padded_flattened_length))

    # application positions of conv: this relates to the index where the top left corner of the conv sits in the padded input, at the application of the filter.
    img_positions = np.mgrid[0:res_shape[0], 0:res_shape[1]]
    img_positions = np.transpose(img_positions, (1,2,0)).reshape((-1,2))

    # distinct position in conv filter
    conv_positions = np.mgrid[0:conv.shape[0], 0:conv.shape[1]]
    conv_positions = np.transpose(conv_positions, (1,2,0)).reshape((-1,2))

    if sparse_matrix:
        x_indices_list = []
        y_indices_list = []
        vals_list = []

    # Write convolutional weights in many places of the global transition matrix
    for i, conv_pos in enumerate(conv_positions):
        # which weight of the conv to write?
        val = conv[conv_pos[0], conv_pos[1]]
        
        # calc all 576 places to write to
        x_indices = np.arange(res_flattened_length)
        y_indices = np.ravel_multi_index((img_positions + conv_pos).T, img_shape_padded)
        
        if sparse_matrix:
            x_indices_list.append(x_indices)
            y_indices_list.append(y_indices)
            vals_list.append(np.full(res_flattened_length, val))
        else:
            # write
            trans[x_indices, y_indices] = val

    # delete columns of trans matrix that are associated with padding. make trans square again.
    column_mask = np.ones(img_padded_flattened_length, dtype=bool)
    ara = np.arange(img_padded_flattened_length)
    # exclude all those img_positions that lie in the padded rows (the first two, and the last two).
    column_mask[ara < img_shape_padded[1] * zero_padding[0]] = False
    column_mask[ara >= len(column_mask) - img_shape_padded[1] * zero_padding[0]] = False 
    # exclude all those that are applied in the first rows, and last rows, that are just zero padding
    column_mask[np.mod(ara,                   img_shape_padded[1]) < zero_padding[1]] = False
    column_mask[np.mod(ara + zero_padding[1], img_shape_padded[1]) < zero_padding[1]] = False

    assert img_flattened_length == column_mask.sum(), "Number of columns that are to be preserved is unequal to the input length."

    if not sparse_matrix:
        ### filter ###
        trans = trans[:, column_mask]
    else:
        x_indices = np.array(x_indices_list).flatten()
        y_indices = np.array(y_indices_list).flatten()
        values = np.array(vals_list).flatten()

        ### filter ###
        indices_mask = column_mask[y_indices] # a lookup in a binary array -> binary array

        # drop those indices & values that are in columns that we want to delete.
        y_indices = y_indices[indices_mask]   # applying a binary array to an integer array -> smaller integer array
        x_indices = x_indices[indices_mask]   
        values = values[indices_mask]

        # reduce those y_indices that are AFTER columns that we want to delete.
        y_indices_translation = column_mask.cumsum() - 1
        y_indices = y_indices_translation[y_indices]

        trans = coo_array((values, (x_indices, y_indices)), shape=(res_flattened_length, img_flattened_length))

    return trans


# calculate surrogate model
def forw_surrogate_matrix(W, curr, gamma, checks=True, recover_activations=True, smart_gamma_func=None):

    # create unnormalized gamma forward matrix
    if not smart_gamma_func: 
        # apply one gamma parameter globally
        R_i_to_j = W + W * (W > 0) * gamma
    else: 
        assert gamma >= 0 and gamma < 1, f"If smart_gamma_func is given, gamma ({gamma}) should be a scale parameter between 0 and 1."

        # obtain maximum valid gamma per per row of matrix by function. Then use the 'gamma' parameter (range 0 to 1) to scale up to these row-wise maxima.
        gamma_per_row = smart_gamma_func(W, curr)
        assert len(gamma_per_row.shape) == 1 and gamma_per_row.shape[0] == W.shape[0], "The return of smart_gamma_func does not fit to the weight matrix W."

        R_i_to_j = W + W * (W > 0) * gamma * gamma_per_row[:, None]
    
    if checks:
        assert R_i_to_j.shape == W.shape

    # normalize it
    if recover_activations:
        # activation of following layer
        foll = W @ curr
        
        res_unnormalized = R_i_to_j @ curr
        forwards_ratio = foll / res_unnormalized
        forwards_ratio[np.logical_and(foll == 0, res_unnormalized == 0)] = 1 # rule: 0/0 = 1 (this is an edge case that does not matter much)

        R_i_to_j *= forwards_ratio[:, None]

        if checks==True:
            # check local equality of modified and original transtition matrix
            assert np.allclose(R_i_to_j @ curr,  W @ curr, atol=0.002), f"Too high difference in outputs (gamma = {gamma}). Maximum point-wise diff: {np.abs((R_i_to_j @ curr) - (W @ curr)).max()}"
        elif "warn" in checks.lower():
            # check local equality of modified and original transtition matrix
            if not np.allclose(R_i_to_j @ curr,  W @ curr, atol=0.002): 
                print(f"Warning: High difference in outputs (gamma = {gamma}). Maximum point-wise diff: {np.abs((R_i_to_j @ curr) - (W @ curr)).max()}")

    
    return R_i_to_j

def back_matrix(W, curr, gamma, smart_gamma_func=None, log=False):
    assert curr is not None, "Reference point needed for for backward matrix"

    R_j_given_i = forw_surrogate_matrix(W, curr, gamma, checks=True, recover_activations=True, smart_gamma_func=smart_gamma_func)

    R_j_and_i   = R_j_given_i * curr[None, :]
    R_i_and_j   = R_j_and_i.T
    R_j         = R_i_and_j.sum(axis=0)
    R_j.resize((1, R_j.shape[0]))
    R_i_given_j = R_i_and_j * (1 / R_j) # we multiply by the reciprocal, as direct division would turn coo_arrays into dense matrices

    if log:
        print('R_j_given_i\n', R_j_given_i)
        print('R_j_and_i\n', R_j_and_i)
        print('R_i_and_j\n', R_i_and_j)
        print('R_j\n', R_j)
        print('R_i_given_j\n', R_i_given_j)

    return R_i_given_j

def calc_evals(W, point, gammas, num_evals=None, return_evecs=False, mode="forw recover activations", smart_gamma_func=None, abs_evals=True):
    
    if num_evals is None: num_evals = W.shape[0]

    matrix_func_dict = {
        "forw recover activations": partial(forw_surrogate_matrix, recover_activations=True), 
        "forw":                     partial(forw_surrogate_matrix, recover_activations=False), 
        "back":                     back_matrix
    }
    assert mode in matrix_func_dict, f'Mode "{mode}" not available'
    matrix_func = matrix_func_dict[mode]

    forwards = [matrix_func(W, point, gamma, smart_gamma_func=smart_gamma_func) for gamma in gammas]

    evals, evecs = [], []
    for M in tqdm(forwards):
        if type(M) is np.ndarray:
            vals, vecs = np.linalg.eig(M)
        elif type(M) is coo_array:
            if num_evals < M.shape[0]:
                vals, vecs = eigs(M, k=num_evals)
            else:
                # falls back to dense matrix eigendecomposition
                vals, vecs = np.linalg.eig(M.toarray())
        else:
            assert False, f"matrix of type {type(M)} not supported."

        evals.append(vals)
        if return_evecs:
            evecs.append(vecs.T)

    evals = np.array(evals)
    if return_evecs:
        evecs = np.array(evecs)

    # sort by ascending abs(eigenvalues)
    if abs_evals: evals = np.abs(evals)
    order = np.argsort(-np.abs(evals), axis=1)

    if num_evals and num_evals < evals.shape[1]:
        order = order[:, :num_evals]

    x_index = np.ones(order.shape[1]) * np.arange(order.shape[0])[:, None]
    x_index = x_index.astype(int)
    evals = evals[x_index, order]

    if not return_evecs: 
        return evals

    evecs = evecs[x_index, order]
    return evals, evecs

def calc_evals_batch(weights_list, points_list, gammas=np.linspace(0,1,201)[:-1], num_evals=None, **kwargs): # kwargs can include 'mode', 'return_evecs' and 'smart_gamma_func'
    return_evecs = kwargs['return_evecs'] if 'return_evecs' in kwargs else False

    ### preemptive checks ###
    if not num_evals: # lower bound
        num_evals = min([min(W.shape) for W in weights_list])
    else: # upper bound
        maxi = max([max(W.shape) for W in weights_list])
        if maxi < num_evals:
            print(f"Warn: too high number of evals requested ({num_evals}). We instead return {maxi}.")
            num_evals = maxi

    if (kwargs['mode'] in ["forw"]):
        if (points_list is not None and type(points_list) is list and len(points_list) > 1):
            print("Note: For mode \"{mode}\" the matrix is not reference point dependent.\nFunction will only return one Eval set per weight matrix.")
        points_list = [None]
    
    computed_evals = np.zeros((len(weights_list), len(points_list), len(gammas), num_evals))
    if return_evecs:
        computed_evecs = np.zeros((len(weights_list), len(points_list), len(gammas), num_evals, weights_list[0].shape[0]))

    for i, W in enumerate(weights_list):
        for j, point in enumerate(points_list):
            ret = calc_evals(W, point, gammas, num_evals, **kwargs)
            if return_evecs: 
                computed_evals[i][j], computed_evecs[i][j] = ret
            else: 
                computed_evals[i][j] = ret

    if return_evecs: 
        return computed_evals, computed_evecs
    return computed_evals


def plot_evals_lineplot(precomputed_evals, gammas=np.linspace(0,1,201)[:-1], 
                num_evals=None, mark_positive_slope=False, percentile_to_plot=None, ylim=4, one_plot_per='weight',
                yscale='linear', green_line_at_x=None):
    """
    Plots the evolution of Eigenvalues with increasing gammas in a lineplot.
    """

    # reduce number of eval lines to show
    if num_evals:
        precomputed_evals = precomputed_evals[:, :, :, :num_evals]

    n_ax_dict = {
        'in total': 1,
        'weight': precomputed_evals.shape[0],
        'point': precomputed_evals.shape[0] * precomputed_evals.shape[1]
    }
    assert one_plot_per in ['point', 'weight', 'in total']
    n_ax = n_ax_dict[one_plot_per]

    y_lim_lower = {'linear':0, 'log': .1}[yscale]

    figsize = (5*n_ax, 3) if n_ax>1 else (20, 10)
    fig, axs = plt.subplots(1, n_ax, figsize=figsize)
    axs, ax_i, ax = np.array(axs).flatten(), -1, None

    fig.suptitle('Evolution of abs(complex eigenvalues) with increasing $\gamma$' +
            ('\nFat bar below indicates section of positive derivative' if mark_positive_slope else ''))

    ### helper functions ###
    def ax_init():
        nonlocal axs, ax, ax_i
        # iterate to next ax
        ax_i += 1
        ax = axs[ax_i]

        # plt.figure(figsize=(20,10))
        ax.set_xlabel('$\gamma$')
        ax.set_ylabel('Eigenvalue')
        ax.set_ylim((y_lim_lower, ylim))

        if green_line_at_x is not None: ax.axvline(green_line_at_x, color="green")
        
    def ax_show():
        nonlocal ax
        ax.set_yscale(yscale)
        # ax.legend(loc='upper right')

    ### preemptive checks ###
    assert np.all([[len(line) == len(gammas) for line in sub_list] for sub_list in precomputed_evals]), "Shape doesn't match."

    if one_plot_per=='in total': ax_init()

    for i, data_for_one_matrix_for_all_points in enumerate(precomputed_evals): # iterate matrices
        if one_plot_per=='weight': ax_init()

        for j, evals in enumerate(data_for_one_matrix_for_all_points): # iterate points
            if one_plot_per=='point': ax_init()
            if percentile_to_plot:
                if np.all(precomputed_evals >= 0):
                    y_lim_upper = np.percentile(evals, percentile_to_plot) + .2
                    ax.set_ylim(y_lim_lower, y_lim_upper)
                else: # we didi not take the absolute of evals
                    y_lim_lower = np.percentile(evals,                    (100-percentile_to_plot)/2) - .2
                    y_lim_upper = np.percentile(evals, percentile_to_plot+(100-percentile_to_plot)/2) + .2
                    range_lim = y_lim_upper - y_lim_lower
                    y_lim_lower -= range_lim * 0.02
                    y_lim_upper += range_lim * 0.02
                    ax.set_ylim((y_lim_lower, y_lim_upper))

            # reset color cycle
            ax.set_prop_cycle(None)
            # plot for this point
            Y = evals + np.random.normal(0, .005, size=evals.shape[1])[None, :] # add some random noise, such that lines don't overlap.
            labels = [f'Exp. {i+1}, Point {j+1}, EV {k+1}' for k in range(evals.shape[1])]
            ax.plot(gammas, Y, label=labels)

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


def eval_peak_distribution_plot(computed_evals, gammas, weights_lbls=None):
    """
    Calculates and plots the distribution of "peak": At which Gamma does the line of evals have its maximum? This is often near to where the Eval would explode to infinity.
    NOT RECOMMENDED to use, as the "Peaks" can also simply be found by looking at the Row entries while Gamma increases. No need to cals Evals. Also this method might be inconsistent, as the ordering of Evals is not consistent across rising Gamma.
    Instead, use the sign_flip_distribution_plot below.
    """

    peak_indices = np.argmax(computed_evals[:,:,:,0], axis=2) # where does the largest eval peak?
    peak_gammas = gammas[peak_indices]

    n = len(computed_evals)
    fig, axs = plt.subplots(1, n, sharey=True, figsize=(n*1.1, 5))

    for peaks_for_one_weight, ax, lbl in zip(peak_gammas, axs, weights_lbls):
        is_below_bound = peaks_for_one_weight == 0
        is_above_bound = peaks_for_one_weight == gammas.max()
        mask = np.logical_and(1-is_below_bound, 1-is_above_bound)

        if mask.sum()!=0:
            sns.violinplot(ax=ax, data=peaks_for_one_weight[mask])
        ax.axhline(.25, color="green")
        
        # info = f"@min: {is_below_bound.sum()}\nGOOD: {mask.sum()}\n@max: {is_above_bound.sum()}"
        # info = f"{is_below_bound.sum()}-{mask.sum()}-{is_above_bound.sum()}"
        info = f"{is_below_bound.sum()}\n{mask.sum()}\n{is_above_bound.sum()}"
        ax.set_title(lbl + "\n" + info)

    return peak_gammas

def calc_gamma_for_sign_flip(W, curr, log=False):

    # Multiply weights * earlier layer activations.
    # Then mask them out by if they are positive
    # Then sum axis 1 to obtain row-wise sum at negative/positive entry positions.

    sum_at_pos_weight_entries = (W * curr[None, :] * (W > 0)).sum(axis=1)
    sum_at_neg_weight_entries = (W * curr[None, :] * (W < 0)).sum(axis=1)
    gamma_at_sign_flip = - sum_at_neg_weight_entries / sum_at_pos_weight_entries - 1
    
    if log:
        print('W\n', W)
        print('W * curr[None, :]\n', W * curr[None, :])
        print('sum_at_pos_weight_entries\n', sum_at_pos_weight_entries)
        print('sum_at_neg_weight_entries\n', sum_at_neg_weight_entries)
        print('gamma_at_sign_flip\n', gamma_at_sign_flip)

    return gamma_at_sign_flip
    

def sign_flip_distribution_plot(weights_list, points_list, weights_lbls, mode='back', gammaRange=(0, 3)):
    """
    Calculates and plots the distribution of "peak": At which Gamma do the sign flip? This is a critical point for the functionality of the Gamma rule.
    """

    assert mode == 'back', "Only 'back' mode supported so far."
    peaks_per_weight = []

    n = len(weights_list)
    fig, axs = plt.subplots(1, n, sharey=True, figsize=(n*1.1, 5))

    for W, ax, lbl in zip(weights_list, axs, weights_lbls):
        peaks_for_one_weight = np.array([calc_gamma_for_sign_flip(W, p) for p in points_list]) # <-- untested change
        peaks_per_weight.append(peaks_for_one_weight)

        is_below_bound = peaks_for_one_weight <= gammaRange[0]
        is_above_bound = peaks_for_one_weight >= gammaRange[1]

        # plot
        mask = np.logical_and(1-is_below_bound, 1-is_above_bound)
        if mask.sum(): sns.violinplot(ax=ax, data=peaks_for_one_weight[mask])
        ax.axhline(.25, color="green")

        # label
        info = f"{is_below_bound.sum()}\n{mask.sum()}\n{is_above_bound.sum()}"
        ax.set_title(lbl + "\n" + info)

    return peaks_per_weight


def smart_gamma_func(W, curr, lower_bound=0, lower_bound_handling=0., upper_bound=100000, upper_bound_handling='clip', log=0):
    """
    Returns a maximum suggested gamma parameter per row of the transition matrix W, given the current data point gamma. 
    """
    gammas = calc_gamma_for_sign_flip(W, curr, log=False)

    if log: print(gammas, end=' ')

    if upper_bound_handling=='clip':
        if upper_bound is not None:
            gammas = gammas.clip(None, upper_bound)
    else:
        assert False, f"upper_bound_handling '{upper_bound_handling}' not implemented."

    if lower_bound_handling=='min within range':
        gammas[gammas <= lower_bound] = gammas[gammas > lower_bound].min()
    elif type(lower_bound_handling) in [int, float]:
        gammas[gammas <= lower_bound] = lower_bound_handling
    else:
        assert False, f"lower_bound_handling '{lower_bound_handling}' not implemented."

    if log: print('->', gammas)

    return gammas