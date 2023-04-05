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
    
