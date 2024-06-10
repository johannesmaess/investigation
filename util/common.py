import os, sys
from util.naming import *

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
### check column sum of all sparse arrays within a multidimensional np array
def vectorize(arr, func=lambda x: x.sum(axis=0)):
    # Assuming arr is an array of sparse arrays
    result = np.array([func(x) for x in arr.flatten()])
    shape = list(arr.shape) + [-1]
    result = result.reshape(shape)
    return result
    
### convenience function to get the right gamma array, based on svals shape
        
def match_gammas(vals):
    num = len(vals[0][0])
    
    if num==3: return gammas3
    if num==5: return gammas5
    if num==22: return gammas_0_1_21_inf
    if num==40: return gammas40
    if num==80: return gammas80
    if num==400: return gammas400

    raise Exception("Provide gammas manually, if they are not any of {gammas3, gammas5, gammas_0_1_21_inf, gammas40, gammas80}.")

    
def parse_partition(n_weights, n_points, partition):
    """
    partition int should be zero indexed
    output partition 2-tuple will be zero indexed
    """
    if type(partition) == int:
        partition = (partition % n_weights, int(partition / n_weights))
    
    if type(partition) is tuple:
        assert len(partition) == 2 \
            and 0 <= partition[0] < n_weights \
            and 0 <= partition[1] < n_points, \
            f"Invalid partition {partition}."
    else:
        assert partition is None, f"Pass integer or tuple as partition. {partition}"
        
    return partition
    
LARGE_FS = 24
SMALL_FS = 15
TINY_FS = 12
purple = (min(67/255, 1), min(1/255, 1), min(84/255, 1))

def annotate_common(axs, n_expected=None, xlabel='$\gamma$', ylabel=None, yvertical=True):
    axs = np.array(axs).flatten()
    if n_expected is not None: assert len(axs) == n_expected
    
    for ax in axs: 
        ax.set_title('')
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=SMALL_FS)

    if ylabel is not None:
        if yvertical:
            axs[0].set_ylabel(ylabel, fontsize=SMALL_FS)
        else:
            axs[0].set_ylabel(ylabel, fontsize=SMALL_FS, rotation=0, ha='right')

    return axs

def annotate_axs_d3_cascading(axs, n_expected=6, pf=False, left=False, **kwargs):
    axs = annotate_common(axs, n_expected=n_expected, **kwargs)

    Rstr = "$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$"
    if not left and not pf:
        for ax in axs:
            ax.set_title(p, loc='left', fontsize=TINY_FS)
        Rstr=""
    elif pf:
        Rstr=""
    else:
        Rstr += ", "

    oneline = True
    kw = { 'fontsize': SMALL_FS }
    if pf:      kw['loc'] = 'right'; kw['color'] = purple
    elif left:  kw['loc'] = 'left'
    else:       kw['loc'] = 'center'

    axs[0].set_title(Rstr + "$\gamma^{(1)} = \gamma$" + "" if oneline else "\n$\gamma^{(2-6)} = 0$",    **kw)
    if n_expected==1: return
    axs[1].set_title(Rstr + "$\gamma^{(1,2)} = \gamma$" + "" if oneline else "\n$\gamma^{(3-6)} = 0$",  **kw)
    if n_expected==2: return
    axs[2].set_title(Rstr + "$\gamma^{(1-3)} = \gamma$" + "" if oneline else "\n$\gamma^{(4-6)} = 0$",  **kw)
    if n_expected==3: return
    axs[3].set_title(Rstr + "$\gamma^{(1-4)} = \gamma$" + "" if oneline else "\n$\gamma^{(5,6)} = 0$",  **kw)
    if n_expected==4: return
    axs[4].set_title(Rstr + "$\gamma^{(1-5)} = \gamma$" + "" if oneline else "\n$\gamma^{(6)} = 0$",    **kw)
    if n_expected==5: return
    axs[5].set_title(Rstr + "$\gamma^{(1-6)} = \gamma$",                                                **kw)

def annotate_axs_d3_individual_gamma(axs, pf=False, left=False, **kwargs):
    kwargs.setdefault('n_expected', 6)
    axs = annotate_common(axs, **kwargs)
    for i, ax in enumerate(axs):
        Rstr = "$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$"
        gstr = "$\gamma^{(" + str(i+1) + ")} = \gamma$"
        if pf:
            # annotate for Pixelflipping: only gamma, in purple, on the right.
            ax.set_title(gstr, loc='right', fontsize=SMALL_FS, color=purple)
            continue

        if not left:
            # print on left and in the middle
            ax.set_title(Rstr, loc='left', fontsize=TINY_FS)
            ax.set_title(gstr, loc='center', fontsize=SMALL_FS)
        else:
            # print on left and in the middle
            ax.set_title(Rstr + ", "+ gstr, loc='left', fontsize=SMALL_FS)

def annotate_axs_all(axs, **kwargs):
    axs = annotate_common(axs, n_expected=1, **kwargs)

    axs[0].set_title("$A^{(0 \\leftarrow T)}_{LRP}$, $\gamma^{(0-T)} = \gamma$", loc='left',  fontsize=SMALL_FS)
    
def annotate_axs_individual(axs, ts=None, **kwargs):
    axs = annotate_common(axs, **kwargs)
    if ts is None:
        ts = np.arange(len(axs)) + 1
    else:
        ts = np.array(ts).flatten()
        assert len(ts) == len(axs)

    def annotate_one(ax, t):
        title="$A^{(" + str(t) + " \\leftarrow " + str(t+1) + ")}_{LRP}(\gamma)$"
        ax.set_title(title, loc='left', fontsize=SMALL_FS)

    for ax, t in zip(axs, ts):
        annotate_one(ax, t)