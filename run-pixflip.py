#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-235
n_tasks = 235

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

import quantus

from util.quantus import *
from util.naming import *
from util.common import *
from util.util_pickle import *
from util.util_cnn import first_mnist_batch, load_mnist_v4_models

#### define LRP params
# gamma_mode = 'cascading_gamma'
gamma_mode = 'individual_gamma'

# gammas = gammas_0_1_21_inf
gammas = gammas40

if   gammas is gammas40:          g_str = 'gammas40'
elif gammas is gammas_0_1_21_inf: g_str = 'gammas_0_1_21_inf'
else: assert 0

modes = {0: 'LRP-0'}
for i, l_ub in enumerate(d3_after_conv_layer):
    for j, g in enumerate(gammas):
        if g=='inf': g = 1e8
        if g!=0:
            g = np.round(g, 8)
            if gamma_mode=='cascading_gamma':  modes[i*1000+j] = f'Gamma.            l<{l_ub} gamma={g}'
            if gamma_mode=='individual_gamma': modes[i*1000+j] = f'Gamma. l>{l_ub-2} l<{l_ub} gamma={g}'





k = None # num points = all

data, target = first_mnist_batch()

y_batch = target.detach().numpy()
x_batch =   data.detach().numpy().reshape((100, 1, 28, 28))

pixFlipMetric = quantus.PixelFlipping(disable_warnings = False, perturb_baseline='black')

def flipScores(a_batch, k=None):
    if k==None: k=len(a_batch)
    return pixFlipMetric(
        model=model,
        x_batch=x_batch[:k],
        y_batch=y_batch[:k],
        a_batch=a_batch[:k],
        device=device
    )

# load v4 models
model_dict = load_mnist_v4_models()
model = model_dict[d3_tag]
model.eval()


### load relevancies

assert (relevancies_per_mode := load_data('d3', f'Rels__m0_to_0__{gamma_mode}__{g_str}'))

### partitioned computation
if n_tasks != 1: assert n_tasks == len(modes.values()), f"{n_tasks} != {len(modes.values())}"
for i, mode_str in enumerate(modes.values()):
    i += 1
    if i == i_task or n_tasks == 1:
        print(i, mode_str)
        batch_scores = { mode_str: { 'PixFlip': flipScores(relevancies_per_mode[mode_str][0].numpy(), k) } }
        save_data('d3', 'PixFlipScores__black__individual_gamma__gammas40', batch_scores, partition=(0, i))