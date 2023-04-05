#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-5
n_tasks = 5

### Configure on local / Qsub system ###
import os
import sys

if 'SGE_TASK_ID' in os.environ:
    sys.path.append("/home/johannes/Masterarbeit")
    i_task = int(os.environ['SGE_TASK_ID'])
    print(f"Task {i_task} of {n_tasks}.")
    assert i_task <= n_tasks
else:
    print("Running in non-parallel mode.")
    i_task, n_tasks = 1, 1

### Imports ###
from util.util_gamma_rule import calc_vals_batch
from util.util_lrp import LRP_global_mat, calc_mats_batch_functional, funcs_cascading__s4__m1_to_1, funcs_inv_cascading__s4__m1_to_1
from util.util_cnn import load_mnist_v4_models, first_mnist_batch
# from util.util_data_summary import *
from util.naming import *
from util.common import *

### Config ###

data, target = first_mnist_batch()
model_dict = load_mnist_v4_models()

LRP =  1
SVALS= 1

## cascading 
if i_task == 1:
    model=model_dict[s4f3_tag]
    pickle_key =   ('s4f3', 'svals__m1_to_1___inverted_cascading_gamma__gammas40')
    
    if LRP:
        mat_funcs = funcs_inv_cascading__s4__m1_to_1(model)
        mask = np.any(model.forward(data).detach().numpy(), axis=1)
        d = data[mask][:20]


if i_task == 2:
    model=model_dict[s4f5_tag]
    pickle_key =   ('s4f5', 'svals__m1_to_1___inverted_cascading_gamma__gammas40')
    
    if LRP:
        mat_funcs = funcs_inv_cascading__s4__m1_to_1(model)
        mask = np.any(model.forward(data).detach().numpy(), axis=1)
        d = data[mask][:20]
        
if i_task == 3:
    model=model_dict[s4f7_tag]
    pickle_key =   ('s4f7', 'svals__m1_to_1___inverted_cascading_gamma__gammas40')
    
    if LRP:
        mat_funcs = funcs_inv_cascading__s4__m1_to_1(model)
        mask = np.any(model.forward(data).detach().numpy(), axis=1)
        d = data[mask][:20]

if i_task == 4:
    model=model_dict[s4f9_tag]
    pickle_key =   ('s4f9', 'svals__m1_to_1___inverted_cascading_gamma__gammas40')
    
    if LRP:
        mat_funcs = funcs_inv_cascading__s4__m1_to_1(model)
        mask = np.any(model.forward(data).detach().numpy(), axis=1)
        d = data[mask][:20]

if i_task == 5:
    model=model_dict[s4f11_tag]
    pickle_key =   ('s4f11', 'svals__m1_to_1___inverted_cascading_gamma__gammas40')
    
    if LRP:
        mat_funcs = funcs_inv_cascading__s4__m1_to_1(model)
        mask = np.any(model.forward(data).detach().numpy(), axis=1)
        d = data[mask][:20]

    
print(pickle_key, len(mat_funcs), len(d), i_task)

if LRP:
    print("Computing mats...")
    calc_mats_batch_functional(mat_funcs, gammas40, d, pickle_key=pickle_key, overwrite=True)
    print("Done with mats.")

if SVALS:
    print("Computing Svals...")
    calc_vals_batch(pickle_key=pickle_key, overwrite=True)#, partition=partition)
    print("Done with Svals.")