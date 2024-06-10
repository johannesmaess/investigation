from util.util_cnn import *
from util.util_lrp import *
from util.naming import *

from util.util_gamma_rule import calc_vals_batch
from util.util_data_summary import *
from util.vgg import *

model_dict = load_mnist_v4_models()
model_d3 = model_dict['cb1-8-8-8_cb2-16-16-16_seed-0']
data, target = first_mnist_batch(batch_size=100)

key = '__15_to_0__testset100__all_gamma__gammas80'
mats = load_data('d3', 'LRP'+key)

first_p = None
first_n = 20
if first_p: new_key = key + f'__p{first_p:03}'
if first_n: new_key = key + f'__n{first_n:03}'

As, Ls = layerwise_forward_pass(model_d3, data)
A = [a.detach().flatten() for a in As[15]]

new_mats = batch_transform_xai_mat(mats, A, first_p=first_p, first_n=first_n)


calc_vals_batch(new_mats, pickle_key=('d3', 'svals'+new_key), overwrite=True)