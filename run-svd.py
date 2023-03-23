#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1
n_tasks = 1

### Configure on local / Qsub system ###
import os
import sys


if 'SGE_TASK_ID' in os.environ:
    sys.path.append("/home/johannes/Masterarbeit")
    i_task = int(os.environ['SGE_TASK_ID']) - 1
    print(f"Task {i_task} of {n_tasks}.")
    assert i_task < n_tasks
else:
    print("Running in non-parallel mode.")
    i_task, n_tasks = 0, 1

### Imports ###
%load_ext autoreload
%autoreload 2

import os
from tqdm import tqdm
import copy
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

# import quantus

from util.util_gamma_rule import calc_vals_batch

from util.util_lrp import LRP_global_mat, calc_mats_batch_functional
from util.util_cnn import load_mnist_v4_models, first_mnist_batch
from util.util_data_summary import *
from util.util_pickle import *
from util.naming import *

### Config ###

data, target = first_mnist_batch()
model_dict = load_mnist_v4_models()
model_d3 = model_dict[d3_tag]


# todo run on cluster
pickle_key = ('d3', 'svals__individual_layer__gammas5')
mat_funcs = [partial(LRP_global_mat, model=model_d3, l_inp=l_inp, l_out=l_out, delete_unactivated_subnetwork=True) for l_inp, l_out in [(2, 3), (4, 5), (7,8), (9, 10), (11, 12)]]
calc_mats_batch_functional(mat_funcs, gammas40, data[:20], pickle_key=pickle_key)
calc_vals_batch(pickle_key=pickle_key, overwrite=True)