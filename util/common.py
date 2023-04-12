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
        
      
    
### convenience function to get the right gamma array, based on svals shape
        
def match_gammas(vals):
    num = vals.shape[2]
    
    if num==3: return gammas3
    if num==5: return gammas5
    if num==22: return gammas_0_1_21_inf
    if num==40: return gammas40

    print("Provide gammas manually, if they are not any of {gammas3, gammas5, gammas_0_1_21_inf, gammas40}.")
    assert 0
    
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

def annotate_common(axs, n_expected=None, xlabel='$\gamma$', ylabel=None):
    axs = np.array(axs).flatten()
    if n_expected is not None: assert len(axs) == n_expected
    
    for ax in axs: 
        ax.set_title('')
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=SMALL_FS)

    if ylabel is not None:
        axs[0].set_ylabel(ylabel, fontsize=SMALL_FS)

    return axs

def annotate_axs_d3_cascading(axs, **kwargs):
    axs = annotate_common(axs, n_expected=6, **kwargs)

    axs[0].set_title("$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$", loc='left', fontsize=LARGE_FS)
    axs[1].set_title("$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$", loc='left', fontsize=LARGE_FS)
    axs[2].set_title("$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$", loc='left', fontsize=LARGE_FS)
    axs[3].set_title("$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$", loc='left', fontsize=LARGE_FS)
    axs[4].set_title("$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$", loc='left', fontsize=LARGE_FS)
    axs[5].set_title("$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$", loc='left', fontsize=LARGE_FS)

    axs[0].set_title("$\gamma^{(1)} = \gamma$\n$\gamma_{(2-6)} = 0$",     loc='right', fontsize=SMALL_FS)
    axs[1].set_title("$\gamma^{(1,2)} = \gamma$\n$\gamma_{(3-6)} = 0$", loc='right', fontsize=SMALL_FS)
    axs[2].set_title("$\gamma^{(1-3)} = \gamma$\n$\gamma_{(4-6)} = 0$", loc='right', fontsize=SMALL_FS)
    axs[3].set_title("$\gamma^{(1-4)} = \gamma$\n$\gamma_{(5,6)} = 0$", loc='right', fontsize=SMALL_FS)
    axs[4].set_title("$\gamma^{(1-5)} = \gamma$\n$\gamma_{(6)} = 0$",   loc='right', fontsize=SMALL_FS)
    axs[5].set_title("$\gamma^{(1-6)} = \gamma$",                     loc='right', fontsize=SMALL_FS)

def annotate_axs_d3_all(axs, **kwargs):
    axs = annotate_common(axs, n_expected=1, **kwargs)

    axs[0].set_title("$R^{(1 \\leftarrow T)}_{\cdot | \cdot}$", loc='left',  fontsize=LARGE_FS)
    axs[0].set_title("$\gamma^{(1-6)} = \gamma$",                 loc='right', fontsize=LARGE_FS)
    
def annotate_axs_individual(axs, ts=None, **kwargs):
    axs = annotate_common(axs, **kwargs)
    if ts is None:
        ts = np.arange(len(axs))
    else:
        ts = np.array(ts).flatten()
        assert len(ts) == len(axs)

    def annotate_one(ax, t):
        title="$R^{(" + str(t) + " \\leftarrow " + str(t+1) + ")}_{\cdot | \cdot}(\gamma)$"
        ax.set_title(title, loc='left', fontsize=LARGE_FS)

    for ax, t in zip(axs, ts):
        annotate_one(ax, t)