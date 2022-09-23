import numpy as np
import matplotlib.pyplot as plt


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