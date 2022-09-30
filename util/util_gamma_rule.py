import numpy as np
import torch
import matplotlib.pyplot as plt

def global_conv_matrix(conv, bias=None, img_shape=None, zero_padding=(0,0),
                        sparse_matrix=False):
    assert(img_shape and len(img_shape) == len(zero_padding))
    img_shape = np.array(img_shape)
    zero_padding = np.array(zero_padding)
    if bias: print("Warning: So far we currently do nothing with passed bias terms.")

    # this is the matrix that will represent the global convolution operation
    img_shape_padded = img_shape + 2*zero_padding
    img_flattened_length = np.prod(img_shape_padded)

    res_shape = img_shape_padded - conv.shape + 1
    res_flattened_length = np.prod(res_shape)
    
    # this is gonna be the global convolution matrix
    if not sparse_matrix:
        trans = np.zeros((res_flattened_length, img_flattened_length))
    else:
        trans = torch.sparse.Tensor(size=(res_flattened_length, img_flattened_length))

    # application positions of conv: this relates to the index where the top left corner of the conv sits in the padded input, at the application of the filter.
    img_positions = np.mgrid[0:res_shape[0], 0:res_shape[1]]
    img_positions = np.transpose(img_positions, (1,2,0)).reshape((-1,2))

    # distinct position in conv filter
    conv_positions = np.mgrid[0:conv.shape[0], 0:conv.shape[1]]
    conv_positions = np.transpose(conv_positions, (1,2,0)).reshape((-1,2))

    # Write convolutional weights in many places of the global transition matrix
    for i, conv_pos in enumerate(conv_positions):
        # which weight of the conv to write?
        val = conv[conv_pos[0], conv_pos[1]]
        
        # calc all 576 places to write to
        x_indices = np.arange(res_flattened_length)
        y_indices = np.ravel_multi_index((img_positions + conv_pos).T, img_shape_padded)
        
        # write
        trans[x_indices, y_indices] = val

    # delete columns of trans matrix that are associated with padding. make trans square again.
    mask = np.ones(img_flattened_length)
    ara = np.arange(img_flattened_length)
    # exclude all those img_positions that lie in the padded rows (the first two, and the last two).
    mask[ara < img_shape_padded[1] * zero_padding[0]] = False
    mask[ara >= len(mask) - img_shape_padded[1] * zero_padding[0]] = False 
    # exclude all those that are applied in the first rows, and last rows, that are just zero padding
    mask[np.mod(ara,                   img_shape_padded[1]) < zero_padding[1]] = False
    mask[np.mod(ara + zero_padding[1], img_shape_padded[1]) < zero_padding[1]] = False

    trans = trans[:, mask != 0]
    return trans


# calculate surrogate model
def forw_surrogate_matrix(W, curr, gamma):
    # activation of following layer
    foll = W @ curr
    
    # create unnormalized gamma forward matrix
    R_i_to_j = W + gamma * np.clip(W, 0, None)
    
    assert R_i_to_j.shape == W.shape
    assert (R_i_to_j @ curr).shape == (W.shape[0],)
          
    # normalize it
    forwards_ratio = foll / (R_i_to_j @ curr)
    forwards_ratio[np.logical_and(foll == 0, (R_i_to_j @ curr) == 0)] = 1 # rule: 0/0 = 1 (this is an edge case that does not matter much)


    R_i_to_j *= forwards_ratio[:, None]
    
    assert R_i_to_j.shape == W.shape

    # check local equality of modified and original transtition matrix
    assert np.allclose(R_i_to_j @ curr,  W @ curr), f"Too high difference in outputs: {(R_i_to_j @ curr) - (W @ curr)}"

    # print(np.percentile(np.absolute(R_i_to_j), (0,1,99,100)))
    
    return R_i_to_j

def run_and_plot(weights_list, points_list, gammas,
                num_evals=None, mark_positive_slope=True, percentile_to_plot=None, ylim=4,
                one_plot_per_point=False):
    # helper functions
    def plt_init():
        plt.figure(figsize=(20,10))
        plt.title('Evolution of abs(complex eigenvalues) with increasing $\gamma$' +
                ('\nFat bar below indicates section of positive derivative' if mark_positive_slope else ''))
        plt.xlabel('$\gamma$')
        plt.ylabel('Eigenvalue')
        plt.ylim((-1, ylim))
        
    def plt_show():
        plt.legend(loc='upper right')
        plt.show()
        
    for i, W in enumerate(weights_list):
        if not one_plot_per_point: plt_init()

        for j, point in enumerate(points_list):
            if one_plot_per_point: plt_init()

            forwards = [forw_surrogate_matrix(W, point, gamma) for gamma in gammas]
            # backwards = [back_surrogate_matrix(W, point, gamma) for gamma in gammas]

            evals, evecs = list(zip(*[np.linalg.eig(forward) for forward in forwards]))
            del evecs

            evals = np.array(evals)
            # evecs = np.array(evecs)

            # remove evals and evecs where eval is 0:
            is_non_zero = evals[0] != 0
            evals = evals[:, is_non_zero]
            # evecs = evecs[:, is_non_zero, :]

            # sort by ascending abs(eigenvalues)
            evals = np.abs(evals)
            order = np.argsort(-evals, axis=1)

            if num_evals and num_evals < evals.shape[1]:
                order = order[:, :num_evals]

            print(f"Matrix {i+1}, Point {j+1}: {is_non_zero.sum()} of {evals.shape[1]} Eigenvalues are non-zero. {order.shape[1]} get plotted.")

            x_index = np.ones(order.shape[1]) * np.arange(order.shape[0])[:, None]
            x_index = x_index.astype(int)
            evals = evals[x_index, order]
            # evecs = evecs[x_index, order] todo

            if percentile_to_plot:
                y_lim = np.percentile(evals, percentile_to_plot)
                plt.ylim((-1, y_lim))

            # reset color cycle
            plt.gca().set_prop_cycle(None)
            # plot for this gamma
            plt.plot(gammas, evals + np.random.normal(0, .005, size=evals.shape[1])[None, :], label=[f'Exp. {i+1}, Point {j+1}, EV {k+1}' for k in range(evals.shape[1])])

            if mark_positive_slope: # plot a scatter dot if the series values is increasing
                # calc sign of derivative
                is_positive = np.diff(evals, axis=0) > 0

                # reset color cycle
                plt.gca().set_prop_cycle(None)

                for k, is_pos, label in zip(range(100), is_positive.T, [f'Point {j+1}, EV {k+1} (Increasing segment)' for k in range(evals.shape[1])]):
                    x = gammas[:-1][is_pos]
                    y = np.full_like(x, -.2 -.05*k - .15*j)
                    plt.scatter(x,y, s=5)


            if one_plot_per_point: plt_show()
        if not one_plot_per_point: plt_show()