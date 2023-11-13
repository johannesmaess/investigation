#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-100
n_tasks = 100

### Configure on local / Qsub system ###
import os
import sys

if 'SGE_TASK_ID' in os.environ:
    sys.path.append("/home/johannes/Masterarbeit")
    i_task = int(os.environ['SGE_TASK_ID'])
    print(f"Task {i_task} of {n_tasks}.")
    assert 1 <= i_task <= n_tasks
else:
    print("Running in non-parallel mode.")
    i_task, n_tasks = 1,1

### Imports ###
from util.util_gamma_rule import calc_vals_batch
from util.util_lrp import *
from util.util_cnn import load_mnist_v4_models, first_mnist_batch
# from util.util_data_summary import *
from util.naming import *
from util.common import *

### Config ###

data, target = first_mnist_batch(1000)
model_dict = load_mnist_v4_models()

LRP =  1
SVALS= 1

n_weights, n_points = 5, 20

n_partitions = n_weights * n_points
num = int((i_task-1) / n_partitions) + 1
partition = (i_task-1) % n_partitions


# overwrite
num = 0
partition = i_task-1

if num == 0:
    model=model_dict[d3_tag]
    model_key =     'd3'

if num == 1:
    model=model_dict[s4f3_tag]
    model_key =     's4f3'
    
if num == 2:
    model=model_dict[s4f5_tag]
    model_key =     's4f5'

if num == 3:
    model=model_dict[s4f7_tag]
    model_key =     's4f7'
    
if num == 4:
    model=model_dict[s4f9_tag]
    model_key =     's4f9'
        
if num == 5:
    model=model_dict[s4f11_tag]
    model_key =     's4f11'
    
    
    
pickle_key =   (model_key, 'svals__individual_layer__gammas400')

print(pickle_key, 'i_task', i_task, 'num', num, 'partition', partition)


if LRP:
    mask = np.any(model.forward(data).detach().numpy(), axis=1)
    d = data[mask][:n_points]
    
    mat_funcs = funcs_individual__d3(model)
    
    print("Computing mats...")
    calc_mats_batch_functional(mat_funcs, gammas400, d, pickle_key=pickle_key, overwrite=False, tqdm_for='gamma', partition=partition)
    print("Done with mats.")

if SVALS:
    print("Computing Svals...")
    calc_vals_batch(pickle_key=pickle_key, overwrite=False, tqdm_for='gamma', partition=partition, matrices_shape=(n_weights, n_points))
    print("Done with Svals.")
