#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-400
n_tasks = 400

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
# from tqdm import tqdm
# import copy
# from functools import partial

# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# import quantus

from util.util_gamma_rule import calc_vals_batch

# from util.util_lrp import LRP_global_mat, calc_mats_batch_functional
from util.util_cnn import load_mnist_v4_models, first_mnist_batch
from util.util_data_summary import *
from util.util_pickle import *
from util.naming import *

### Config ###

data, target = first_mnist_batch()
model_dict = load_mnist_v4_models()
model_d3 = model_dict[d3_tag]

print(i_task)

num = i_task % 100
partition = num % 5, int(num / 5)

if 0 < i_task <= 100:
    pickle_key = ('d3', 'svals__individual_layer__gammas40')
    # mat_funcs = [partial(LRP_global_mat, model=model_d3, l_inp=l_inp, l_out=l_out, delete_unactivated_subnetwork=True) for l_inp, l_out in [(2, 3), (4, 5), (7,8), (9, 10), (11, 12)]]
if 100 < i_task <= 200:
    pickle_key = ('d3', 'svals__m1_to_1___cascading_gamma__gammas40')
    # mat_funcs = [partial(LRP_global_mat, model=model_d3, l_ub=l_ub, l_inp=1, l_out=-2, delete_unactivated_subnetwork=True) for l_ub in d3_after_conv_layer[:-1]]

# eps 0
if 200 < i_task <= 300:
    pickle_key = ('d3', 'svals__individual_layer__gammas40__eps0')
    # mat_funcs = [partial(LRP_global_mat, eps=0, model=model_d3, l_inp=l_inp, l_out=l_out, delete_unactivated_subnetwork=True) for l_inp, l_out in [(2, 3), (4, 5), (7,8), (9, 10), (11, 12)]]
if 300 < i_task <= 400:
    pickle_key = ('d3', 'svals__m1_to_1___cascading_gamma__gammas40__eps0')
    # mat_funcs = [partial(LRP_global_mat, eps=0, model=model_d3, l_ub=l_ub, l_inp=1, l_out=-2, delete_unactivated_subnetwork=True) for l_ub in d3_after_conv_layer[:-1]]

print(pickle_key, partition)

# print("Computing mats...")
# calc_mats_batch_functional(mat_funcs, gammas40, data[:20], pickle_key=pickle_key)
print("Done with mats. Computing Svals...")
calc_vals_batch(pickle_key=pickle_key, overwrite=True, partition=partition)
print("Done with Svals.")