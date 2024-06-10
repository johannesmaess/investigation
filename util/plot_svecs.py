from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from util.common import match_gammas
from util.util_data_summary import pretty_num
import util.util_tutorial as tut_utils


def plot_svecs(svals, lvecs, rvecs, iw, ip, method_selection = None, val_selection = None, methods = None, scale_rvec = 2, save_func = None, decrease = .3):
    if methods is None:
        try: 
            methods = match_gammas(svals)
            methods = ['γ='+pretty_num(g) for g in methods]
        except: pass
    
    if method_selection is None:
        method_selection = np.arange(svals.shape[2])

    # for one point
    svals_p = svals[iw, ip]
    lvecs_p = lvecs[iw, ip]

    n_vals = np.any(svals_p > 0, axis=0).sum()
    svals_p = svals_p[:, :n_vals]
    lvecs_p = lvecs_p[:, :n_vals]

    if val_selection is None:
        val_indices = np.arange(n_vals)
        val_selection = []
    else:
        # make indices positive such that we can print them later.
        val_indices = sorted(list(set([v if v>=0 else n_vals+v for v in val_selection])))

    svals_p = svals_p[:, val_indices]
    lvecs_p = lvecs_p[:, val_indices]
    
    if rvecs is not None:
        rvecs_p = rvecs[iw, ip]
        rvecs_p = rvecs_p[:, :n_vals]
        rvecs_p = rvecs_p[:, val_indices]

    # for one gamma
    for i_parameter in method_selection:
        svals_g, lvecs_g = svals_p[i_parameter], lvecs_p[i_parameter]
        
        n_vals, dimensionality = lvecs_g.shape
        # shap = shapes[dimensionality]
        shap = (1, 28, 28)
        
        # decrease strength of heatmaps a bit based on Singular values, reduce to one channel
        vec = lvecs_g * (svals_g**decrease)[:, None]
        vec = vec.reshape((n_vals, *shap))
        # vec = np.abs(vec)
        vec = vec.sum(axis=1)
        
        # svecs are agnostic to sign. make all heatmaps predominantly positive
        is_positive = 0 < vec.sum(axis=(1,2), keepdims=True)
        vec *= (is_positive * 2 - 1)
        
        # reshape and plot batch
        vec = vec.transpose(1,0,2)
        vec = vec.reshape((shap[1],n_vals*shap[2]))
        move_left = 2
        vec = np.hstack((vec[:, move_left:], np.zeros((vec.shape[0], move_left))))
        ax = tut_utils.heatmap(vec, sx=200, sy=4)
        
        for i, _ in enumerate(lvecs_g):
            idx = str(val_indices[i]+1)
            y, x = 1.9, shap[2] * (i+.1)
            val_txt = '$\sigma_{' + idx + '}=$' + pretty_num(svals_g[i])
            val_txt = '$\sigma_{' + idx + '}/\sigma_1=$' + pretty_num(svals_g[i]/svals_g[0])
            val_txt = '$\sigma_{' + idx + '}=$' + pretty_num(svals_g[i]) \
                + ',\n$\sigma_{' + idx + '}/\sigma_1=$' + pretty_num(svals_g[i]/svals_g[0])
            ax.text(x, y, val_txt + ', $u_{' + idx + '}$:')
                
            x = shap[2] * (i+1)
            line = Line2D([x, x], [0, shap[1] - 1], color='black', linewidth=1)
            ax.add_line(line)

        if rvecs is not None:
            rvecs_g = rvecs_p[i_parameter]
            for i, rvec in enumerate(rvecs_g):
                
                rvec = rvec[:, None]
                if not is_positive[i,0,0]: rvec *= -1
                
                # plot rvec
                w, h = scale_rvec * np.array(rvec.shape)
                y, x = 5, shap[2] * (i+1) - scale_rvec - 1
                inset_ax = ax.inset_axes([x/vec.shape[1], y/vec.shape[0], h/vec.shape[1], w/vec.shape[0]])
                tut_utils.heatmap(rvec, ax=inset_ax)
                
                for d in ['top', 'bottom', 'left', 'right']:
                    inset_ax.spines[d].set_color('black')
                    inset_ax.spines[d].set_linewidth(.3)
                inset_ax.xaxis.set_ticks_position('top')
                inset_ax.set_xticks([0])
                inset_ax.set_xticklabels(['$v_{' + idx + '}$'])
                inset_ax.set_yticks(np.arange(10))

        # Label with method
        if methods is not None: 
            m = pretty_num(methods[i_parameter])
            # print('method:', m)
            ax.text(1, shap[2]-2, m, fontsize=16)
            
        if save_func is not None:
            assert methods is not None, "Pass descriptive 'methods' to save figs."
            
            fn = f'svecs_w{iw}_p{ip}_'
            for v in val_selection: fn += f'{v}_'
            fn += m
            
            save_func(fig=plt.gcf(), fn=fn)
            print('Saved:', fn)
        
        plt.show()