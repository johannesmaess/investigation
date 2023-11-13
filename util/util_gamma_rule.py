from functools import partial
from tqdm import tqdm

import numpy as np
from numpy.linalg import eig, svd

from util.common import parse_partition
from util.naming import *
from util.util_pickle import load_data, save_data

import torch
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
            # computation of smallest svals by passing negative k:
            # issue with argpack: doesnt converge sometimes
            # issue with propack: how many? sometimes last one is 0.
            if num_vals < 0: which="SM"; num_vals = -num_vals
            else:            which="LM"
            if return_vecs: lvecs, vals, rvecs = svds(M, k=num_vals, which=which, return_singular_vectors=True)
            else:                  vals        = svds(M, k=num_vals, which=which, return_singular_vectors=False)
    elif type(M) is np.ndarray:
        if np.any(np.isnan(M)):
            print(M)
            return
        if not svd_mode: vals, lvecs        = eig(M)
        else:            lvecs, vals, rvecs = svd(M, full_matrices=False)
    else:
        raise Exception(f"Invalid type {type(M)}")

    vals = np.array(vals)
    if abs_vals: vals = np.abs(vals)

    # determine order by magnitude, for the num_vals largest vals
    order = np.argsort(-np.abs(vals), axis=0)[:num_vals]

    # return sorted vals and vecs
    vals = vals[order]
    if return_vecs:
        lvecs = np.array(lvecs).T[order]
        if svd_mode:
            rvecs = np.array(rvecs)[order]
            return vals, lvecs, rvecs
        return vals, lvecs
    return vals


def calc_vals_batch(matrices=None, num_vals='auto', return_vecs=False, svd_mode=True, abs_vals=False, tqdm_for='point', pickle_key=None, overwrite=False, partition=None, matrices_shape=None):
    """
    Wraps around calc_evals to make calls for multiple weights, and multiple reference points.
    Mostly useful for determining an efficient number of vals to compute per matrix, putting the results into uniform arrays, and its checks.

    Tries to load the previously computed result first.

    overwrite - Don't load previous result. Compute and overwrite it.
    partition = (i, j) - only the i'th weight and j'th poitn are computes, and saved with appendix __wi__pj .

    """
    # display progress util
    itg, itp, itm = [lambda x: x]*3
    if tqdm_for=='matrix': itm = tqdm
    if tqdm_for=='point':  itp = tqdm
    if tqdm_for=='gamma':  itg = tqdm

    if pickle_key is not None:
        mkey, dkey = pickle_key
        ind = dkey.find('__')

        # key for loading LRP matrices: "LRP__..."
        dkey_lrp = 'LRP' + dkey[ind:]

        # if passed key is of form "__..." default to "svals__..."
        if num_vals == 'auto':
            dkey = 'svals' + dkey[ind:]
        elif type(num_vals) is int:
            dkey = f'svals{num_vals}' + dkey[ind:]
        else:
            assert 0, f"Invalid num_vals {num_vals}"

        # try loading the data
        if not overwrite:
            vals = load_data(mkey, dkey)
            if vals is not False:
                if matrices is not None: assert vals.shape[:3] == matrices.shape[:3], "Found svals in storage, but they do not match the passed matrices."
                if partition is not None: print("Found unpartitioned, full result. Returning.")
                return vals

        # if matrices are not passed, try to load them
        if matrices is None:
            matrices = load_data(mkey, dkey_lrp)
            if matrices is False: 
                assert  partition is not False, "matrices are not passed, and can not be loaded from storage."
            else:
                marices_shape = len(matrices), len(matrices[0])
            

        # for saving the svals
        def save_func(x): 
            print("Saving vals under key:", (mkey, dkey))
            save_data(mkey, dkey, x, partition=partition)
    else:
        save_func = lambda x: 0
        assert matrices is not None, "Neither the matrices, nor a key for loading them from storage is passed."
        
    if partition is not None:
        partition = parse_partition(*matrices_shape, partition)
        w, p = partition
        if matrices not in [False, None]:
            matrices = matrices[w:w+1, p:p+1]
        else:
            matrices = load_data(mkey, dkey_lrp, partition=partition)
        
    
    assert matrices is not False, "matrices are not passed, and can not be loaded from storage."
            
    # print('type(matrices):', type(matrices))
    n_weights, n_points, n_gammas = len(matrices), len(matrices[0]), len(matrices[0][0])
    m0 = matrices[0][0][0]
    assert len(m0.shape) == 2, "'matrices' should contain 2D arrays (np.ndarray or scipy.coo_array), nested in a 3D structure"

    if abs_vals or svd_mode: dtype=np.float
    else:                    dtype=np.cfloat

    # if return_evcs==True, we want all matrices to be of the same size. also extrace vec_len.
    if return_vecs: 
        vec_len = m0.shape[0]
        for i, matrices_per_weight in enumerate(matrices):
            for j, matrices_per_point in enumerate(matrices_per_weight):
                for k, matrix_per_gamma in enumerate(matrices_per_point):
                    assert matrix_per_gamma.shape == m0.shape, "Pass only matrices of same shape"

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
    vals_per_matrix = max_rank_per_matrix if num_vals=='auto' else np.clip(max_rank_per_matrix, a_min=None, a_max=np.abs(num_vals))

    # initialize stores
    if partition: n_weights, n_points = 1, 1
    computed_vals =     np.zeros((n_weights, n_points, n_gammas, vals_per_matrix.max())             , dtype=dtype)
    if return_vecs: 
        computed_lvecs = np.zeros((n_weights, n_points, n_gammas, vals_per_matrix.max(), m0.shape[0]), dtype=dtype)
    if return_vecs and svd_mode: 
        computed_rvecs = np.zeros((n_weights, n_points, n_gammas, vals_per_matrix.max(), m0.shape[1]), dtype=dtype)

    # calculate decomposition
    try:
        for i, matrices_per_weight in itm(enumerate(matrices)):
            for j, matrices_per_point in itp(enumerate(matrices_per_weight)):
                for k, matrix_per_gamma in itg(enumerate(matrices_per_point)):
                    res = calc_vals(matrix_per_gamma, num_vals=vals_per_matrix[i,j], return_vecs=return_vecs)
                    vals = res[0] if return_vecs else res
                    computed_vals[i, j, k, :len(vals)] = vals
                    if return_vecs:
                        lvecs = res[1]
                        computed_lvecs[i, j, k, :len(lvecs)] = lvecs
                        if svd_mode:
                            rvecs = res[2]
                            computed_rvecs[i, j, k, :len(rvecs)] = rvecs

                save_func(computed_vals)

    except KeyboardInterrupt:
        print("Received Interrupt. Stop computation, return incomplete result.")

    if dtype==np.cfloat:
        # if none of the calculations returned imaginary parts, change dtype to real.
        if not np.any(np.imag(computed_vals)):
            computed_vals = np.real(computed_vals)
        if return_vecs and not np.any(np.imag(computed_lvecs)):
            computed_lvecs = np.real(computed_lvecs)
        if return_vecs and svd_mode and not np.any(np.imag(computed_rvecs)):
            computed_rvecs = np.real(computed_rvecs)

    save_func(computed_vals)

    if return_vecs:
        if svd_mode:
            return computed_vals, computed_lvecs, computed_rvecs
        return computed_vals, computed_lvecs
    return computed_vals

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