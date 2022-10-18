import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.sparse import coo_array
from scipy.sparse.linalg import eigs

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
        y_indices = y_indices[indices_mask]   # applying a binary array to an integer array -> integer array
        x_indices = x_indices[indices_mask]   
        values = values[indices_mask]

        # reduce those y_indices that are AFTER columns that we want to delete.
        y_indices_translation = column_mask.cumsum() - 1
        y_indices = y_indices_translation[y_indices]

        indices = np.vstack([x_indices, y_indices])
        trans = torch.sparse_coo_tensor(indices = indices,
                                        values = values,
                                        size=(res_flattened_length, img_flattened_length))

    return trans


# calculate surrogate model
def forw_surrogate_matrix(W, curr, gamma, checks=True, normalization=True):

    assert ((curr is None) != normalization), "A reference point is needed iff. normalization==True (and the original models activations are supposed to be recovered)."

    # create unnormalized gamma forward matrix
    R_i_to_j = W + gamma * np.clip(W, 0, None)
    
    if checks:
        assert R_i_to_j.shape == W.shape

    # normalize it
    if normalization:
        # activation of following layer
        foll = W @ curr
        
        res_unnormalized = R_i_to_j @ curr
        forwards_ratio = foll / res_unnormalized
        forwards_ratio[np.logical_and(foll == 0, res_unnormalized == 0)] = 1 # rule: 0/0 = 1 (this is an edge case that does not matter much)

        R_i_to_j *= forwards_ratio[:, None]

        if checks:
            # check local equality of modified and original transtition matrix
            curr = np.random.normal(1, .2, size=len(R_i_to_j))
            assert np.allclose(R_i_to_j @ curr,  W @ curr, atol=0.0001), f"Too high difference in outputs. Maximum point-wise diff: {np.abs((R_i_to_j @ curr) - (W @ curr)).max()}"
    
    return R_i_to_j

# calculate surrogate model - sparse
def forw_surrogate_matrix_sparse(W, curr, gamma, checks='Warn only', normalization=True):
    # create unnormalized gamma forward matrix
    clipped = torch.sparse_coo_tensor(indices = W.coalesce().indices(),
                                        values = W.coalesce().values().clip(0,None),
                                        size = W.shape)
    R_i_to_j = W + gamma * clipped
    
    if checks:
        assert R_i_to_j.shape == W.shape
          
    # normalize it
    if normalization:
        # activation of following layer
        foll = W @ curr

        res_unnormalized = R_i_to_j @ curr
        forwards_ratio = foll / res_unnormalized
        forwards_ratio[torch.logical_and(foll == 0, res_unnormalized == 0)] = 1 # rule: 0/0 = 1 (this is an edge case that does not matter much)

        indices = R_i_to_j.coalesce().indices()
        x_indices = indices[0]
        forwards_ratio_per_value = forwards_ratio[x_indices]

        R_i_to_j = torch.sparse_coo_tensor(indices = indices,
                                values = R_i_to_j.coalesce().values() * forwards_ratio_per_value,
                                size = R_i_to_j.shape)

        if "warn" in checks.lower():
            # check local equality of modified and original transtition matrix
            if not np.allclose(R_i_to_j @ curr,  W @ curr, atol=0.002):
                print(f"Warning: High difference in outputs (gamma = {gamma}). Maximum point-wise diff: {np.abs((R_i_to_j @ curr) - (W @ curr)).max()}")

        elif checks:
            # check local equality of modified and original transtition matrix
            assert np.allclose(R_i_to_j @ curr,  W @ curr, atol=0.002), f"Too high difference in outputs (gamma = {gamma}). Maximum point-wise diff: {np.abs((R_i_to_j @ curr) - (W @ curr)).max()}"

    if checks == 2: # level for major checks
        print("Running checks of lvl 2. Dense matrix computation can take a while...")

        R_i_to_j_dense = forw_surrogate_matrix(W.to_dense(), curr, gamma, checks=True, normalization=normalization)
        diff = (R_i_to_j_dense - R_i_to_j)

        print("Warning: We allow a high point-wise error of 0.2 in the forward matrix.") # We hope that this gets cancelled out well though, as the error is high absolute, but very low in relative terms.
        assert np.allclose(diff, 0, atol=0.2), f"Too high difference in forward matrices (gamma = {gamma}). Maximum point-wise diff: {np.abs(R_i_to_j_dense - R_i_to_j).max()}"

        print("The five largest error are:")
        for i in range(5):
            ind = (np.unravel_index(diff.abs().argmax(), shape=R_i_to_j_dense.shape))
            print(ind, diff[ind], R_i_to_j_dense[ind], R_i_to_j[ind])
            diff[ind] = 0

    return R_i_to_j

def calc_evals(W, point, gammas, num_evals=None, return_evecs=False, normalization=True, return_only_non_zero=True):
    forwards = [forw_surrogate_matrix(W, point, gamma, normalization=normalization) for gamma in gammas]
    # backwards = [back_surrogate_matrix(W, point, gamma) for gamma in gammas]

    evals, evecs = list(zip(*[np.linalg.eig(forward) for forward in forwards]))

    evals = np.array(evals)
    evecs = np.array(evecs)

    if return_only_non_zero:
        # remove evals and evecs where eval is 0:
        is_non_zero = evals[0] != 0
        evals = evals[:, is_non_zero]
        evecs = evecs[:, is_non_zero, :]

    # sort by ascending abs(eigenvalues)
    evals = np.abs(evals)
    order = np.argsort(-evals, axis=1)

    if num_evals and num_evals < evals.shape[1]:
        order = order[:, :num_evals]

    # print(f"Matrix {i+1}, Point {j+1}: {is_non_zero.sum()} of {evals.shape[1]} Eigenvalues are non-zero. {order.shape[1]} get plotted.")

    x_index = np.ones(order.shape[1]) * np.arange(order.shape[0])[:, None]
    x_index = x_index.astype(int)
    evals = evals[x_index, order]
    evecs = evecs[x_index, order] # todo

    if not return_evecs: 
        return evals
    return evals, evecs

def calc_evals_sparse(W, point, gammas, num_evals, normalization=True):
    # compute sparse matrices in Pytorch COO format
    forwards = [forw_surrogate_matrix_sparse(W, point, gamma, normalization=normalization) for gamma in gammas]
    # change from Pytorch COO sparse to scipy COO sparse
    forwards = [coo_array((forw.coalesce().values(), forw.coalesce().indices()), forw.shape) for forw in forwards]

    evals, evecs = list(zip(*[eigs(forward, k=num_evals) for forward in forwards]))

    evals = np.array(evals)
    del evecs
    evals = np.abs(evals) # NOTE: we only analyse abs(EV) so far

    return evals

def calc_evals_batch(weights_list, points_list, gammas, num_evals=None, return_evecs=False, normalization=True):
    ### preemptive checks ###
    if not num_evals: # lower bound
        num_evals = min([min(W.shape) for W in weights_list])
    else: # upper bound
        maxi = max([max(W.shape) for W in weights_list])
        if maxi < num_evals:
            print(f"Warn: too high number of evals requested ({num_evals}). We instead return {maxi}.")
            num_evals = maxi

    if (not normalization):
        if (points_list is not None) or (type(points_list) is list and len(points_list) > 1):
            print("Note: Without normlization, the forward matrix is not reference point dependent.\nFunction will only return one Eval set per weight matrix.")
        points_list = [None]
    
    computed_evals = np.zeros((len(weights_list), len(points_list), len(gammas), num_evals))
    if return_evecs:
        computed_evecs = np.zeros((len(weights_list), len(points_list), len(gammas), num_evals, len(weights_list[0])))

    for i, W in enumerate(weights_list):
        for j, point in enumerate(points_list):
            # print(i,j)
            if type(W) is torch.Tensor and W.is_sparse: # sparse
                if not num_evals or num_evals >= len(W): # sparse, but all EVs requested
                    # compute in dense pipeline
                    ret = calc_evals(W.to_dense().numpy(), point, gammas, num_evals=len(W), normalization=normalization, return_evecs=return_evecs, return_only_non_zero=False)    
                else:
                    ret = calc_evals_sparse(W, point, gammas, num_evals, normalization=normalization) # TODO ret evecs
            else:
                ret = calc_evals(W, point, gammas, num_evals, normalization=normalization, return_evecs=return_evecs, return_only_non_zero=False) # warning: assignment might fail if func returns less than num_evals

            if return_evecs: 
                computed_evals[i][j], computed_evecs[i][j] = ret
            else: 
                computed_evals[i][j] = ret

    if not return_evecs: 
        return computed_evals
    return computed_evals, computed_evecs


def plot_evals_lineplot(precomputed_evals, gammas, 
                num_evals=None, mark_positive_slope=True, percentile_to_plot=None, ylim=4, one_plot_per='weight',
                yscale='linear'):

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

    ylim_lower = {'linear':0, 'log': .1}[yscale]

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
        ax.set_ylim((ylim_lower, ylim))
        
    def ax_show():
        nonlocal ax
        ax.set_yscale(yscale)
        ax.legend(loc='upper right')

    ### preemptive checks ###
    assert np.all([[len(line) == len(gammas) for line in sub_list] for sub_list in precomputed_evals]), "Shape doesn't match."

    if one_plot_per=='in total': ax_init()

    for i, data_for_one_matrix_for_all_points in enumerate(precomputed_evals): # iterate matrices
        if one_plot_per=='weight': ax_init()

        for j, evals in enumerate(data_for_one_matrix_for_all_points): # iterate points
            if one_plot_per=='point': ax_init()
            if percentile_to_plot:
                y_lim = np.percentile(evals, percentile_to_plot) + .2
                plt.ylim((ylim_lower, y_lim))

            # reset color cycle
            plt.gca().set_prop_cycle(None)
            # plot for this point
            Y = evals + np.random.normal(0, .005, size=evals.shape[1])[None, :] # add some random noise, such that lines don't overlap.
            labels = [f'Exp. {i+1}, Point {j+1}, EV {k+1}' for k in range(evals.shape[1])]
            ax.plot(gammas, Y, label=labels)

            if mark_positive_slope: # plot a scatter dot if the series values is increasing
                # calc sign of derivative
                is_positive = np.diff(evals, axis=0) > 0
                # reset color cycle
                plt.gca().set_prop_cycle(None)

                for k, is_pos, label in zip(range(100), is_positive.T, [f'Point {j+1}, EV {k+1} (Increasing segment)' for k in range(evals.shape[1])]):
                    x = gammas[:-1][is_pos]
                    y = np.full_like(x, -.2 -.05*k - .15*j)
                    ax.scatter(x,y, s=5)

            if one_plot_per=='point': ax_show()
        if one_plot_per=='weight': ax_show()
    if one_plot_per=='in total': ax_show()