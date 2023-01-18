#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-64
n_tasks = 64

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
from learning_lrp import train_lrp

batch_size = 100
T = 5e5
lr = 0.02
momentum = 0. # SGD optim only.

loss_tag = "self--corr--\
pred-perturbation-insensitivity=3,\
pred-class-sensitivity=1,\
perturbed-class-sensitivity=1,\
"
loss_tag_short="self-corr-3-1-1"
print(loss_tag, loss_tag_short)

metric_tags = []
shuffle = True

### PARALLELES ###

# loss_tags = [tag for tag in all_shap_metric_tags if 'corr' in tag or 'abs-norm' in tag]
# lr_T_list =[
#     (.001, 1e6),
#     (.002, 1e6),
#     (.005, 1e6),
#     (.01 , 1e5),
#     (.02 , 1e5),
#     (.05 , 1e5),
#     (.1  , 1e5),
#     (.2  , 1e5),
#     (.5  , 1e5),
#     ]

# if n_tasks==len(loss_tags)*len(lr_T_list):
#     i, j = int(i_task % len(loss_tags)), int(i_task / len(loss_tags))
#     loss_tag = loss_tags[i]
#     lr, T = lr_T_list[j]
    
#     print(i_task, loss_tag, lr, T)
# else:
#     print("Can not run multi thread.")

shap_config = 'shap__background_size-100__batch_size-10__model-cb1-8-8-8_cb2-16-16-16_seed-0'
test_loader_shap = data_loaders(shapley_values_for=('d3', shap_config), shuffle=shuffle, batch_size=batch_size)

model_dict = load_mnist_v4_models()
model_d3 = model_dict[d3_tag]

if n_tasks > 1:
    v = 17**np.linspace(0,1,8) - 1
    gammas = np.transpose([np.tile(v, len(v)), np.repeat(v, len(v))])
    print(gammas.shape)
else:
    gammas = [[1, 1]]

assert n_tasks == len(gammas)
ge, gl = gammas[i_task]

# parameters and optimizer
gamma_early = torch.Tensor([ge]).requires_grad_(True)
gamma_late  = torch.Tensor([gl]).requires_grad_(True)
param_string = f"[ge={round(float(gamma_early.data), 3)},gl={round(float(gamma_late.data), 3)}]"
parameters = [gamma_early, gamma_late]

if 0:
    optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer_string = f"opt=Adam,lr={lr}"
else:
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    optimizer_string = f"opt=SGD,lr={lr},mom={momentum}"

    

model = ExplainableNet(model_d3).eval().to(device)
model.change_lrp_rules(gamma=gamma_early, lrp_rule_nl=LRPRule.gamma, start_l=0, end_l=3)
model.change_lrp_rules(gamma=gamma_late , lrp_rule_nl=LRPRule.gamma, start_l=4, end_l=8)

model, gammas, gammas_t, errs, errs_t = train_lrp(
            model, test_loader_shap, optimizer, parameters,
            loss_tag = loss_tag, 
            metric_tags = metric_tags,
            T = T)

# shap notation: 
# tag = f"llrp__V5__model={d3_tag}__loss={loss_tag}__params={param_string}__lr={lr}__T={T:.0e}__backs={100}__batchs={batch_size}__s{int(shuffle)}"

# self notation:
tag = f"llrp__self__V6__model=d3s0__loss={loss_tag_short}__{optimizer_string}__params={param_string}__T={T:.0e}__batchs={batch_size}__s{int(shuffle)}"

data = (gammas, gammas_t, errs, errs_t, metric_tags)
save_data('d3', tag, data)
print("Saved under tag=\n" + tag)

"""
Version Info

V4
Add many metric/loss functions.

V5
Specify in tag which parameters are to be learned, and what their initial value is.
(before, all experiments where done with (gamma=0.051, start_l=0, end_l=3) and (gamma=0.05, start_l=4, end_l=8))

V6
Big bug: When trying to compute relevancies wrt an perturbed input, the relevancies where still computed wrt the unperturbed sample.

V7 
Clip gamma. Min=1e-7.
Specify which Optimizer is used. (Before Adam only.)
"""