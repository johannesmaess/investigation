from functools import partial
from tqdm.notebook import tqdm

import numpy as np
from numpy.linalg import eig, svd

from util.util_data_summary import pretty_num
from util.common import HiddenPrints

import torch

from scipy import stats
from scipy.sparse import coo_array

import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns

# import os
# os.environ["SCIPY_USE_PROPACK"] = "1"
from scipy.sparse.linalg import eigs, svds


def global_conv_matrix(conv, bias=None, img_shape=None, zero_padding=(0,0),
                        sparse_matrix=True, verbose=False):
    """
    Creates matrix representation of a 2D->2D convolution.
    Note that this only handles one color-channel/feature as input, and one feature as output.
    Assumes stride == 1.
    """
    assert(img_shape is not None and len(img_shape) == len(zero_padding))
    img_shape = np.array(img_shape)
    zero_padding = np.array(zero_padding)
    if bias and verbose: print("Warning: So far we currently do nothing with passed bias terms.")

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


def conv_matrix_from_pytorch_layer(layer, img_shape, in_feat_no, out_feat_no):
    """
    Helper fucntion to conv_matrix.
    Creates matrix representation of 2D->2D convolution.
    Only considers one input feature and one output feature.
    """
    assert layer.stride == (1,1), "stride != 1 is not implemented."

    conv = layer.weight.detach()[out_feat_no, in_feat_no]
    bias = layer.bias.detach().numpy()[0] if layer.bias is not None else None                  # TODO: we don't factor in biases so far.
    padd = layer.padding

    trans = global_conv_matrix(conv, bias,
                                img_shape=img_shape,
                                zero_padding=padd,
                                sparse_matrix=True) # using a sparse matrix reduces the funciton runtime by many orders of magnitude

    return trans


def global_conv_matrix_from_pytorch_layer(layer, inp_shape, out_shape, inp_feats=None, out_feats=None, force_square_matrix=False, load_bar=False):
    """
    Helper fucntion to global_conv_matrix.

    """
    assert len(inp_shape) == 3, "Invalid input shape."
    assert len(out_shape) == 3, "Invalid output shape."

    n_out_feats = layer.weight.shape[0]
    n_inp_feats = layer.weight.shape[1]
    assert n_inp_feats == inp_shape[0], "Input shape does not match passed layer."
    
    inp_img_shape = inp_shape[1:]
    out_img_shape = out_shape[1:]

    # if not one specific output feature / filter is selected, create matrix for all of them.
    if inp_feats is None: inp_feats = np.arange(n_inp_feats)
    if out_feats is None: out_feats = np.arange(n_out_feats)


    block_shape = (np.prod(out_img_shape), np.prod(inp_img_shape))
    global_shape = (block_shape[0] * len(out_feats), block_shape[1] * len(inp_feats))

    load_func = lambda x: x
    if load_bar==True:
        load_func = tqdm 
        print(len(out_feats), 'feats to iterate')
    for i, out_feat in load_func(enumerate(out_feats)):
        x_pivot = block_shape[0] * i
        for j, inp_feat in enumerate(inp_feats):
            y_pivot = block_shape[1] * j

            # (x_pivot, y_pivot) is the positiion in the global conv matrix that this submatrix starts at (top left corner).
            trans = conv_matrix_from_pytorch_layer(layer, inp_img_shape, inp_feat, out_feat)
            assert trans.shape == block_shape, f"Costructed conv matrix has unexpected shape. {trans.shape} != {block_shape}"

            # write entries to global matrix stores
            if (i,j) == (0,0):
                entries_per_submatrix = len(trans.data)
                entries_total = entries_per_submatrix * len(out_feats) * len(inp_feats)
                values, x_indices, y_indices = np.zeros(entries_total), np.zeros(entries_total), np.zeros(entries_total)
            
            store_pivot = (i*len(inp_feats) + j) * entries_per_submatrix
            for inp, target in zip((trans.data, trans.row + x_pivot, trans.col + y_pivot), (values, x_indices, y_indices)):
                assert len(inp) == entries_per_submatrix 
                target[store_pivot:store_pivot+entries_per_submatrix] = inp


    if force_square_matrix==True:
        # pad the transition matrix with zero columns on the right, or zero rows on the bottom, to make it square
        n = max(*global_shape)
        global_shape = (n, n)

    global_trans = coo_array((values, (x_indices, y_indices)), shape=global_shape)

    return global_trans




# calculate surrogate model
def forw_surrogate_matrix(W, curr, gamma, checks=True, recover_activations=True, smart_gamma_func=None):

    # create unnormalized gamma forward matrix
    if not smart_gamma_func: 
        # apply one gamma parameter globally
        R_i_to_j = W + W * (W > 0) * gamma
    else: 
        if smart_gamma_func==smart_gamma_max_before_sign_flip:
            assert gamma >= 0 and gamma < 1, f"If smart_gamma_func is smart_gamma_max_before_sign_flip, gamma ({gamma}) should be a scale parameter between 0 and 1."

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

def back_matrix(W, curr, gamma, smart_gamma_func=None, delete_unactivated_subnetwork=False, log=False):
    """
    Note: R_j_forwards is the activation of the forward surrogate model, which is equivalent to the normal NN's pre-ReLU activation z.

    Params:
    delete_unactivated_subnetwork: If the output Relevancy R_j_forward < 0, then a_j = 0, then R_j_backwards = 0.
    In using the back_matrix later, R_i_given_j will in these case always be multiplied by 0.
    Setting delete_unactivated_subnetwork='mask', recognizes thes rows in the matrix and sets their entries R_i_given_j=0, to simplify the matrix.
    """

    assert curr is not None, "Reference point needed for backward matrix"

    R_j_given_i = forw_surrogate_matrix(W, curr, gamma, checks=True, recover_activations=False, smart_gamma_func=smart_gamma_func)

    R_j_and_i   = R_j_given_i * curr[None, :]
    R_i_and_j   = R_j_and_i.T
    R_j         = R_i_and_j.sum(axis=0) # + 1e-9

    if delete_unactivated_subnetwork: 
        R_j[R_j <= 0] = np.inf
        R_j[(W@curr) <= 0] = np.inf

    R_j.resize((1, R_j.shape[0]))
    R_i_given_j = R_i_and_j * (1 / R_j) # we multiply by the reciprocal, as direct division would turn coo_arrays into dense matrices


    if log:
        print('R_j_given_i\n', R_j_given_i)
        print('R_j_and_i\n', R_j_and_i)
        print('R_i_and_j\n', R_i_and_j)
        print('R_j\n', R_j)
        print('R_i_given_j\n', R_i_given_j)

    R_i_given_j = prune_coo_array(R_i_given_j)

    return R_i_given_j

def back_joint_matrix(W, curr, gamma, R_j_backwards, **kwargs):
    """
    Calculates the "joint relevancy" between node j of the later layer and node i of the earlier layer.
    Analogous to the joint probability, we achieve it by multiplying the conditional R_i_given_j with the prior R_j_backwards.
    Note, that in our theroy we distinguish between R_j_forwards and R_j_backwards;
    to emulate a computation step of the LRP method, this function required R_j_backwards as the prior.
    """
    R_i_given_j = back_matrix(W, curr, gamma, **kwargs)
    
    assert R_j_backwards.shape == (1, R_i_given_j.shape[1])
    # R_j_backwards.resize((1, R_j_backwards.shape[0]))
    
    R_i_and_j_backwards = R_i_given_j * R_j_backwards
    return R_i_and_j_backwards

def calc_mats(M, point, gammas, mode, output_layer_relevancies=None, smart_gamma_func=None):
    """
    For one forward transition matrix (representing a transition in a NN),
    and a set of gammas, compute an associated LRP matrix.
    """
    if type(gammas) is list: gammas = [1e8 if g=='inf' else g for g in gammas]
    assert (mode=="back joint") != (output_layer_relevancies is None), "output_layer_relevancies should be given iff. mode=='back joint'."

    matrix_func_dict = {
        "forw recover activations": partial(forw_surrogate_matrix, recover_activations=True), 
        "forw":                     partial(forw_surrogate_matrix, recover_activations=False), 
        "back":                     partial(back_matrix,           delete_unactivated_subnetwork=False),
        "back clip":                partial(back_matrix,           delete_unactivated_subnetwork='mask'),
        "back joint":               partial(back_joint_matrix,     R_j_backwards=output_layer_relevancies),
    }
    assert mode in matrix_func_dict, f'Mode "{mode}" not available'
    matrix_func = matrix_func_dict[mode]

    matrices = [matrix_func(M, point, gamma, smart_gamma_func=smart_gamma_func) for gamma in gammas]
    return matrices


def calc_mats_batch(weights_list, points_list, **kwargs):
    """
    Wrapper for calc_mats, to execute it quickly on multiple weights and reference points.
    """
    return np.array([[calc_mats(W, point, **kwargs) for point in points_list] for W in weights_list])

def calc_vals(M, num_vals, return_vecs=False, svd_mode=True, abs_vals=False):
    """
    For a given matrix, calculate its eigen or singular-value decomposition and return it sorted by the values magnitudes.
    """
    if type(M) is coo_array:
        if num_vals==min(M.shape): 
            # the svds solver wants num_vals > min(M.shape).
            # pad coo_array, with one imagined columns full of zero.
            larger_shape = np.maximum(M.shape, min(M.shape)+1)
            M = coo_array((M.data, (M.row, M.col)), shape=larger_shape)

        if not svd_mode:     vals, vecs    = eigs(M, k=num_vals, which="LM")
        else:            
            if return_vecs: vecs, vals, _ = svds(M, k=num_vals, which="LM", return_singular_vectors='u'  )
            else:                  vals   = svds(M, k=num_vals, which="LM", return_singular_vectors=False)
    elif type(M) is np.ndarray:
        if np.any(np.isnan(M)):
            print(M)
            return
        if not svd_mode: vals, vecs    = eig(M)
        else:            vecs, vals, _ = svd(M, full_matrices=False)
    else:
        raise Exception(f"Invalid type {type(M)}")

    vals = np.array(vals)
    if abs_vals: vals = np.abs(vals)

    # determine order by magnitude, for the num_vals largest vals
    order = np.argsort(-np.abs(vals), axis=0)[:num_vals]

    # return sorted vals and vecs
    return (vals[order], np.array(vecs).T[order] if return_vecs else None)

def calc_vals_batch(matrices, num_vals='auto', return_vecs=False, svd_mode=True, abs_vals=False, tqdm_for='matrix'):
    """
    Wraps around calc_evals to make calls for multiple weights, and multiple reference points.
    Mostly useful for determining an efficient number of vals to compute per matrix, putting the results into uniform arrays, and its checks.
    """
    # display progress util
    itg, itp, itm = [lambda x: x]*3
    if tqdm_for=='matrix': itm = tqdm
    if tqdm_for=='point':  itp = tqdm
    if tqdm_for=='gamma':  itg = tqdm

    n_weights, n_points, n_gammas = matrices.shape[:3]
    assert len(matrices[0, 0, 0].shape) == 2, "'matrices' should contain 2D arrays (np.ndarray or scipy.coo_array), nested in a 2D structure"

    if abs_vals or svd_mode: dtype=np.float
    else:                    dtype=np.cfloat

    # if return_evcs==True, we want all matrices to be of the same size. also extrace vec_len.
    if return_vecs: 
        vec_len = matrices[0, 0, 0].shape[0]
        for i, matrices_per_weight in enumerate(matrices):
            for j, matrices_per_point in enumerate(matrices_per_weight):
                for k, matrix_per_gamma in enumerate(matrices_per_point):
                    assert matrix_per_gamma.shape == matrices[0,0,0].shape, "Pass only matrices of same shape"

    # calculate upper bound for rank of matrix
    max_rank_per_matrix = np.zeros((n_weights, n_points), dtype=int)
    for i, matrices_per_weight in enumerate(matrices):
        for j, matrices_per_point in enumerate(matrices_per_weight):
            W = matrices_per_point[0]
            if isinstance(W, coo_array):
                max_rank_per_matrix[i,j] = min([len(np.unique(W.row)), len(np.unique(W.col))])
            else:
                max_rank_per_matrix[i,j] = min([np.sum(np.any(W, axis=0)), np.sum(np.any(W, axis=1))])

    # valculate the number of vals to be requested per matrix
    vals_per_matrix = max_rank_per_matrix if num_vals=='auto' else np.clip(max_rank_per_matrix, a_min=None, a_max=num_vals)

    # initialize stores
    computed_evals =                 np.zeros((n_weights, n_points, n_gammas, vals_per_matrix.max())         , dtype=dtype)
    if return_vecs: computed_evecs = np.zeros((n_weights, n_points, n_gammas, vals_per_matrix.max(), vec_len), dtype=dtype)

    # calculate decomposition
    for i, matrices_per_weight in itm(enumerate(matrices)):
        for j, matrices_per_point in itp(enumerate(matrices_per_weight)):
            for k, matrix_per_gamma in itg(enumerate(matrices_per_point)):
                evals, evecs = calc_vals(matrix_per_gamma, num_vals=vals_per_matrix[i,j], return_vecs=return_vecs)
                computed_evals[i, j, k, :len(evals)] = evals
                if return_vecs: computed_evecs[i, j, k, :len(evecs)] = evecs

    if dtype==np.cfloat:
        # if none of the calculations returned imaginary parts, change dtype to real.
        if not np.any(np.imag(computed_evals)):
            computed_evals = np.real(computed_evals)
        if return_vecs and not np.any(np.imag(computed_evecs)):
            computed_evecs = np.real(computed_evecs)

    return (computed_evals, computed_evecs if return_vecs else None)

def calc_evals_batch(weights_list, points_list, gammas=np.linspace(0,1,201)[:-1], mode="forw recover activations", smart_gamma_func=None, output_layer_relevancies=None, return_matrices=False, num_vals_largest=None, return_evecs=False, abs_evals=False, svd_mode=False):
    """
    Wrapper function around calc_mats_batch & calc_vals_batch, mostly for backwards compatibility.
    """
    matrices = calc_mats_batch(weights_list=weights_list, points_list=points_list, gammas=gammas, mode=mode, output_layer_relevancies=output_layer_relevancies, smart_gamma_func=smart_gamma_func)
    evals, evecs = calc_vals_batch(matrices=matrices, num_vals=num_vals_largest, return_vecs=return_evecs, svd_mode=svd_mode, abs_vals=abs_evals)
    
    return (evals, evecs, np.array(matrices) if return_matrices else None)

def col_norms_for_matrices(comp_mats, ord=1):
    """
    Computes the column norms for every (spars/dense) matrix in a 2D list.
    """
    col_norms = np.zeros((len(comp_mats), len(comp_mats[0]), len(comp_mats[0][0]), comp_mats[0][0][0].shape[1]))

    for i_weight, mats_for_weight in enumerate(comp_mats):
        for i_point, mats_for_point in enumerate(mats_for_weight):
            for i_gamma, mat in tqdm(enumerate(mats_for_point)):
                if ord==1:
                    # compute one norm of every column of the matrix
                    norm = np.ones(mat.shape[0]) @ np.abs(mat)
                elif ord==2:
                    mat_squared = mat.copy()
                    mat_squared.data = mat_squared.data**2
                    norm = np.ones(mat.shape[0]) @ mat_squared
                    norm = np.sqrt(norm)
                else:
                    assert False, "Unsupported ord."

                col_norms[i_weight, i_point, i_gamma] = norm

    return col_norms


def plot_vals_lineplot(vals, gammas=np.linspace(0,1,201)[:-1], 
                mark_positive_slope=False, plot_only_non_zero=False, one_plot_per='weight',
                num_vals_largest=None, num_vals_total=None,
                ylabel="Singular values", title=None,
                ylim=4, xlim=None,
                yscale='linear', xscale='linear', sharey=False, xtick_mask=None,
                figsize=None, show=True, 
                green_line_at_x=None, tag_line=None, 
                colormap='viridis'):
    """
    Plots the evolution of Eigenvalues with increasing gammas in a lineplot.
    """

    # reduce number of eval lines to show to first n.
    if num_vals_largest:
        vals = vals[:, :, :, :num_vals_largest]
    assert not (num_vals_largest and num_vals_total)

    n_ax_dict = {
        'in total': (1, 1),
        'weight': (1, vals.shape[0]),
        'point': vals.shape[:2]
    }
    assert one_plot_per in ['point', 'weight', 'in total']
    n_ax = n_ax_dict[one_plot_per]

    # 
    if type(xlim) == int:
        x_lim_lower = {'linear':0, 'log': max(1e-3, gammas.min())*.9}[xscale]
        xlim = [x_lim_lower, xlim]
    elif type(xlim) == tuple:
        xlim = list(xlim)
    elif type(xlim) != list and xlim is not None:
        print(f'Warn: Invalid xlim: {xlim}. Setting xlim to None.')
        xlim = None
    
    if xlim:
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
        print('Warn: Invalid ylim: {ylim}. Setting ylim to None.')
        ylim = None

    if figsize is None: 
        figsize = (20, 10) if n_ax==(1,1) else (5*n_ax[1], 3*n_ax[0])
    fig, axs = plt.subplots(*n_ax, figsize=figsize, sharey=sharey)
    axs, ax_i, ax = np.array(axs).flatten(), -1, None

    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle(f'Evolution of {ylabel} with increasing $\gamma$' +
                ('\nFat bar below indicates section of positive derivative' if mark_positive_slope else ''))

    ### helper functions ###
    def ax_init():
        nonlocal axs, ax, ax_i
        # iterate to next ax
        ax_i += 1
        ax = axs[ax_i]

        # plt.figure(figsize=(20,10))
        ax.set_xlabel('$\gamma$')
        if i==0 or sharey==False: 
            ax.set_ylabel(ylabel)

        if green_line_at_x is not None: ax.axvline(green_line_at_x, color="green")

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        
    def ax_show():
        nonlocal ax, xlim, ylim

        if tag_line is not None:
            ax.legend(loc='upper right')

        # set ylim
        ylim_u = ylim[1] * 1.01 + .1
        ylim_l = ylim[0] / 1.01 - .1 if yscale=='linear' else ylim[0] / 1.1
        ax.set_ylim((ylim_l, ylim_u))
        
        if percentile_to_plot: ylim = None

        if xlim is None: return
    
        # set xlim
        xlim_u = xlim[1] * 1.01 + .1
        xlim_l = xlim[0] / 1.01 - .1 if xscale=='linear' else xlim[0] / 1.1
        ax.set_xlim((xlim_l, xlim_u))

    ### preemptive checks ###
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

            labels = [f'Exp. {i+1}, Point {j+1}, Sval {k+1}' for k in range(Y.shape[1])]
            labels = [f'Singular value {k+1}' for k in range(Y.shape[1])] # <- prettier, for Proposal plot
            if tag_line is not None:
                assert len(tag_line) == len(labels), "Invalid labels per line passed."
                labels = tag_line
            
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

            if colormap is not None:
                # count number of lines that have non-zero, non-nan elements in them
                num_colors = np.sum(np.any((Y != 0)*1 - np.isnan(Y), axis=0))
                ax.set_prop_cycle(mpl.cycler('color', [mpl.colormaps[colormap](k) for k in np.linspace(0, 1, num_colors)]))  
            ax.plot(xtick, Y, label=labels)


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

    plt.subplots_adjust(hspace=0.3)
    if show:
        plt.show()
    else:
        return fig, axs


def plot_multiplicative_change(vals, gammas, hmean=False, normalize=False, **passed_kwargs):
    kwargs = {
        "ylabel": "Multiplicative change",
        "title": "Multiplicative change per Singular value. Shows which Singular values decrease fastest relative to their size. Blue are biggest Svals. Red smallest.",
        "yscale": "log",
        "ylim": 'p100',
        "sharey": False,
        "colormap": "seismic",
    }
    for k,v in passed_kwargs.items(): kwargs[k] = v
        
    if normalize:
        vals = vals - vals[:, :, -1:, :]
        
    with HiddenPrints():
        change = vals / vals[:, :, :1, :] # potential divide by 0
    
    if normalize and kwargs['yscale']=='log': change += 0.01
    
    # calculate harmonic mean over...
    if hmean=='points': 
        if normalize: 
            change[:, :, -1] = 1 # the entries for gamma=inf are normalized to 0. The hmean can not be calculated for 0 entries.
            change = np.clip(change, a_min=1e-3, a_max=None)
        change = stats.hmean(change, axis=1, keepdims=True, nan_policy='omit')
        if normalize: 
            change[:, :, -1] = 0 # We thus adopt the policy hmean(..., 0, ...) = 0.
    
    return plot_vals_lineplot(change, gammas, **kwargs)

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


def smart_gamma_max_before_sign_flip(W, curr, lower_bound=0, lower_bound_handling=0., upper_bound=100000, upper_bound_handling='clip', log=0):
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


def smart_gamma_wo_sign_flips(W, curr):
    """
    Three possible cases:
    1. The normal result is negative. In this case the following ReLU makes the output neurons activation and thus relevancy 0.
    2. The normal result = "positive" result. This means all incoming weights are positive.
    3. 0 < normal_result < positive_result. This means some weights are negative. 

    3. is the only case that interests us, and where the gamma rule will have any effect.
    """
    normal_res =    W          @ curr
    positive_res = (W * (W>0)) @ curr
    
    mask = np.zeros_like(normal_res)
    mask[np.logical_and(0 < normal_res, normal_res < positive_res)] = 1

    return mask

def prune_coo_array(mat, atol=0):
    """
    Remove 0 entries from coo_array.
    """
    if not isinstance(mat, coo_array): return mat

    mask = (np.abs(mat.data) > atol)
    return coo_array((mat.data[mask], (mat.row[mask], mat.col[mask])), shape=mat.shape)