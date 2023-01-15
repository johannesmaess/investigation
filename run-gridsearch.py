#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-81
n_tasks = 81

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

import torch
from explanations_can_be_manipulated.src.nn.networks import ExplainableNet
from explanations_can_be_manipulated.src.nn.enums import LRPRule

from util.naming import *
from util.util_pickle import *
from util.util_cnn import data_loaders, load_mnist_v4_models
from learning_lrp import shap_error

metric_tags = all_metric_tags
batch_size = 100
background_size = 100 # SHAP background size
shap_config = f'shap__background_size-{background_size}__batch_size-10__model-cb1-8-8-8_cb2-16-16-16_seed-0'

model_dict = load_mnist_v4_models()
model_d3 = model_dict[d3_tag]
model = ExplainableNet(model_d3).eval().to(device)

test_loader_shap = data_loaders(shapley_values_for=('d3', shap_config), shuffle=False, batch_size=batch_size)

# parameters
# n_steps = 32
# mini, maxi = 0.1, 20

# exp = (maxi/mini)**(1/(n_steps-1))
# v = mini * exp**np.arange(n_steps)

# v = np.concatenate((np.linspace(0, mini, 9)[:-1], v))

## linear only
maxi=1
v = np.linspace(0, 1, 81)
n_steps = len(v)

gammas = np.transpose([np.tile(v, len(v)), np.repeat(v, len(v))])

if n_tasks > 1 and len(gammas) % n_tasks == 0: # multiple qsub tasks
    gammas = gammas.reshape((n_tasks, -1, 2))
    gammas = gammas[i_task]
    print(f"Running {len(gammas)} gammas.")
else:
    print(f"Cant run multiple threads with {len(gammas)} n_gammas, {n_tasks} n_tasks.")
    if i_task > 0: 
        print('Exit.')
        exit()

errs = []

for gamma_early, gamma_late in tqdm(gammas):
    with torch.no_grad():
        model.change_lrp_rules(gamma=gamma_early, lrp_rule_nl=LRPRule.gamma, start_l=0, end_l=3)
        model.change_lrp_rules(gamma=gamma_late , lrp_rule_nl=LRPRule.gamma, start_l=4, end_l=8)
        errs.append(shap_error(model, test_loader_shap, metric_tags=metric_tags))

tag = f"gridlrp__V4__model={d3_tag}__backs={background_size}__n_{n_steps}_r_{0}_{maxi}__task_{i_task:03d}_{n_tasks:03d}"
data = (gammas, errs, metric_tags)
save_data('d3', tag, data)
print("Saved under tag=\n" + tag)


"""
Version Info

V4
Add many metric/loss functions.

"""