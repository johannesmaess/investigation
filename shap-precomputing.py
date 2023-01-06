import numpy as np
import shap 
from tqdm import tqdm
from util.util_pickle import save_data, load_data
from util.util_cnn import data_loaders, first_mnist_batch, load_mnist_v4_models
from util.common import HiddenPrints

from util.naming import d3_tag

# config
config = {
    'background_size': 400,
    'batch_size': 10,
    'model': d3_tag
}
config = dict(sorted(config.items()))

# load model
model_dict = load_mnist_v4_models()
model_d3 = model_dict[d3_tag]

# load training data as shap rbackground
background, background_target = first_mnist_batch(batch_size=100, test=False)
background = background.reshape((-1, 1, 28, 28))
e = shap.DeepExplainer(model_d3, background)

# prepare test loader
_, test_loader = data_loaders(batch_size=10)

# file name
fn = "shap__"
for k,v in config.items(): fn += k + '-' + str(v) + '__'
L = len(test_loader)

# shap_per_batch = []'''‘±'

for i, (x, target) in tqdm(enumerate(test_loader)):
    x = x.reshape((-1,1,28,28)).data
    with HiddenPrints(): vals = e.shap_values(x)      # dimensions: [num_classes, [datapoints, channels, x, y]]
    vals = np.stack(vals).transpose((1, 0, 2, 3, 4))  # dimensions: [datapoints, num_classes, channels, x, y]
    
    # save results
    # shap_per_batch.append(vals)

    save_data('d3', fn+f'batch-{i+1}-{L}', vals)
    