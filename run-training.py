#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-9
n_tasks = 9

import sys
sys.path.append("/home/johannes/Masterarbeit")
import os
i_task = int(os.environ['SGE_TASK_ID']) - 1

import torch
from explanations_can_be_manipulated.src.nn.networks import ExplainableNet
from explanations_can_be_manipulated.src.nn.enums import LRPRule

from util.naming import *
from util.util_pickle import *
from util.util_cnn import data_loaders, load_mnist_v4_models
from learning_lrp import train_lrp

batch_size = 100
lr = 0.02
T = 1e6
loss_func = 'shap--pred--mse'
err_tags = ['shap--pred--mse', 
            'shap--pred--corr']
shuffle = True

### PARALLELES ###

# lr_list = [.001, .002, .005, .01, .02, .05, .1, .2, .5]
# if n_tasks==9:
#     lr = lr_list[i_task]
#     if lr < 0.02: T *= 10

# lr_T_list =[
#     (.001, 1e4),
#     (.01 , 1e4),
#     (.1  , 1e4),
#     (.001, 1e5),
#     (.01 , 1e5),
#     (.1  , 1e5),
#     (.001, 1e6),
#     (.01 , 1e6),
#     (.1  , 1e6),
# ]
# if n_tasks==9:
#     lr, T = lr_T_list[i_task]

lr_T_list =[
    (.001, 1e7),
    (.002, 1e7),
    (.005, 1e7),
    (.01 , 1e7),
    (.02 , 1e7),
    (.05 , 1e7),
    (.1  , 1e6),
    (.2  , 1e6),
    (.5  , 1e6),
]
if n_tasks==9:
    lr, T = lr_T_list[i_task]

shap_config = 'shap__background_size-100__batch_size-10__model-cb1-8-8-8_cb2-16-16-16_seed-0'

model_dict = load_mnist_v4_models()
model_d3 = model_dict[d3_tag]

test_loader_shap = data_loaders(shapley_values_for=('d3', shap_config), shuffle=shuffle, batch_size=batch_size)

# parameters and optimizer
gamma_early = torch.Tensor([0.051]).requires_grad_(True)
gamma_late  = torch.Tensor([0.050]).requires_grad_(True)
parameters = [gamma_early, gamma_late]
optimizer = torch.optim.Adam(parameters, lr=lr)

model = ExplainableNet(model_d3).eval().to(device)
model.change_lrp_rules(gamma=gamma_early, lrp_rule_nl=LRPRule.gamma, start_l=0, end_l=3)
model.change_lrp_rules(gamma=gamma_late , lrp_rule_nl=LRPRule.gamma, start_l=4, end_l=8)

model, gammas, gammas_t, errs, errs_t = train_lrp(
            model, test_loader_shap, optimizer, parameters,
            loss_func = loss_func, 
            err_tags = err_tags,
            T = T)

tag = f"llrp__V3__model={d3_tag}__loss={loss_func}__lr={lr}__T={T:.0e}__backs={100}__batchs={batch_size}__s{int(shuffle)}"
data = (gammas, gammas_t, errs, errs_t, err_tags)
save_data('d3', tag, data)
print("Saved under tag=\n" + tag)