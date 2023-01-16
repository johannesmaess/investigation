import numpy as np

device = "cpu"
MNIST_CNN_PATH = './models/'
PICKLE_PATH = './pickle/'
data_dir = "./dataset"
cat16_dir = data_dir + "/cat16"

# commonly used model identifiers
d3_tag = 'cb1-8-8-8_cb2-16-16-16_seed-0'

gammas3 = [0, .25, 'inf']
gammas5 = [0, .125, .25, .5, 'inf']
gammas_0_1_21_inf = list(np.linspace(0,1,21).round(3)) + ['inf']

d3_after_conv_layer = [3, 5, 8, 10, 12, 16]

all_shap_metric_tags = []
for d in ['mae', 'mse', 'corr']:
    for n in ['', 'norm', 'abs-norm']:
        for which in ['pred', 'other', 'both'][:]:
            all_shap_metric_tags.append(f"shap--{d}--{n}--{which}")