from functools import partial
import os

from util.util_lrp import *
from util.util_gamma_rule import calc_vals_batch
from util.util_pickle import *
from util.naming import *

from util.util_cnn import CNNModel, params_from_filename, params_to_filename, data_loaders

from tqdm import tqdm

# load data
train_loader, test_loader = data_loaders()
for i, (data, target) in enumerate(test_loader):
    break

# load v4 models
model_dict = {}
for fn in os.listdir(MNIST_CNN_PATH):
    if 'mnist_cnn_v4' in fn:
        params = params_from_filename(fn)
        cnn_model = CNNModel(*params).to(device)
        cnn_model.load_state_dict(torch.load(os.path.join(MNIST_CNN_PATH, fn), map_location=device))
        model_dict[fn[13:-6]] = cnn_model

### Deep model ###
model_d3 = model_dict['cb1-8-8-8_cb2-16-16-16_seed-0']

# svd from 2nd to 2nd to last layer.
mat_funcs = [partial(LRP_global_mat, model=model_d3, l_leq=l_leq, l_inp=1, l_out=-2) for l_leq in d3_after_conv_layer[:-1]]
print("mat_funcs done")
LRP__m1_to_1___cascading_gamma__gammas_0_1_21_inf = calc_mats_batch_functional(mat_funcs, gammas_0_1_21_inf, data[:20])
print(LRP__m1_to_1___cascading_gamma__gammas_0_1_21_inf.shape, LRP__m1_to_1___cascading_gamma__gammas_0_1_21_inf[:, :1, :1])

svals__m1_to_1___cascading_gamma__gammas_0_1_21_inf, _ = calc_vals_batch(LRP__m1_to_1___cascading_gamma__gammas_0_1_21_inf, num_vals='auto', tqdm_for='point')
print(svals__m1_to_1___cascading_gamma__gammas_0_1_21_inf.shape)
save_data('svals__m1_to_1___cascading_gamma__gammas_0_1_21_inf', svals__m1_to_1___cascading_gamma__gammas_0_1_21_inf)
print('saved.')

