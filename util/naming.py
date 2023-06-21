import numpy as np
from functools import partial

device = "cpu"
MNIST_CNN_PATH = './models/'
PICKLE_PATH = './pickle/'
data_dir = "./dataset"
cat16_dir = data_dir + "/cat16"


gammas3 = [0, .25, 'inf']
gammas5 = [0, .125, .25, .5, 'inf']
gammas_0_1_21_inf = list(np.linspace(0,1,21).round(3)) + ['inf']
# from 0 to 1000, roughly log distributed:
gammas40 = np.concatenate(([0, 1e-5, 3e-5, 1e-4, 3e-4], np.linspace(0.001, .01, 7)[:-1], np.linspace(0.01, .05, 7)[:-1], np.linspace(.05, .2, 7)[:-1], np.linspace(.2, 1, 7), 2**(np.arange(10)+1)))
gammas80 = np.logspace(-5, 4, 80)
gammas400 = np.linspace(0,10,400)

### shap

all_shap_metric_tags = []
for d in ['mae', 'mse', 'corr']:
    for n in ['', 'norm', 'abs-norm']:
        for which in ['pred', 'other', 'both'][:]:
            all_shap_metric_tags.append(f"shap--{d}--{n}--{which}")

### commonly used model identifiers and their conv layer indices

## v4 models
d3_tag = 'cb1-8-8-8_cb2-16-16-16_seed-0'

d3_after_conv_layer = [3, 5, 8, 10, 12, 16]

## V5 models
s4f3_tag = 'Fs-3-3-3-3_seed-0'
s4f5_tag = 'Fs-5-5-5-5_seed-0'
s4f7_tag = 'Fs-7-7-7-7_seed-0'
s4f9_tag = 'Fs-9-9-9-9_seed-1'
s4f11_tag = 'Fs-11-11-11-11_seed-1'

s6f3_tag = 'Fs-3-3-3-3-3-3_seed-0'
s6f5_tag = 'Fs-5-5-5-5-5-5_seed-1'

s6f7_tag = 'Fs-7-7-7-7-7-7_seed-0'
s8f7_tag = 'Fs-7-7-7-7-7-7-7-7_seed-0'

s4_after_conv_layer = [3, 5, 7]
s6_after_conv_layer = [3, 5, 7, 9, 11]
s8_after_conv_layer = [3, 5, 7, 9, 11, 13, 15]

