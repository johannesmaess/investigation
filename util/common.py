import os, sys
from util.naming import *
from util.util_lrp import LRP_global_mat

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
    
    
    
### convenience functions for LRP matrix creation

## d3 model
def funcs_cascading__d3__m1_to_1(model): # m1 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-3, l_ub=l_ub, delete_unactivated_subnetwork=True) for l_ub in d3_after_conv_layer[:-1]]

def funcs_inv_cascading__d3__m1_to_1(model): # m1 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-3, l_lb=l_ub-2, delete_unactivated_subnetwork=True) for l_ub in d3_after_conv_layer[:-1][::-1]]

## s4 models
def funcs_individual__s4(model):
    return [partial(LRP_global_mat, model=model, l_inp=l_out-1, l_out=l_out, delete_unactivated_subnetwork=True) for l_out in s4_after_conv_layer]
def funcs_cascading__s4__m1_to_1(model): # m1 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-3, l_ub=l_ub, delete_unactivated_subnetwork=True) for l_ub in s4_after_conv_layer]
def funcs_inv_cascading__s4__m1_to_1(model): # m1 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-3, l_lb=l_ub-2, delete_unactivated_subnetwork=True) for l_ub in s4_after_conv_layer[::-1]]
