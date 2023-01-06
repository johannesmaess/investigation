#!/home/johannes/msc/bin/python
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-20
n_tasks = 20

# remove this line when no GPU is needed!  #$ -l cuda=1
# do not fill the qlogin queue
# start processes in current working directory # provide environment variables to processes
# start 100 instances: from 1 to 100

import sys
sys.path.append("/home/johannes/Masterarbeit")
import os
i_task = int(os.environ['SGE_TASK_ID']) 

import numpy as np
import shap 
from tqdm import tqdm

from util.util_pickle import save_data, load_data
from util.util_cnn import data_loaders, first_mnist_batch, load_mnist_v4_models
from util.common import HiddenPrints

from util.naming import d3_tag, PICKLE_PATH

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
_, test_loader = data_loaders(batch_size=10, shuffle=False)

# file name
fn_partial = "shap__"
for k,v in config.items(): fn_partial += k + '-' + str(v) + '__'
L = len(test_loader)

n_batches = len(test_loader)

for i_batch, (x, target) in tqdm(enumerate(test_loader)):
    if not (i_batch % n_tasks == i_task-1): 
        continue # this batch gets handled by another worker.

    fn = fn_partial + f'batch-{i_batch+1:04d}-{L:04d}'
    if os.path.exists(os.path.join(PICKLE_PATH, 'd3', fn+'.pickle')):
        continue

    x = x.reshape((-1,1,28,28)).data
    with HiddenPrints(): vals = e.shap_values(x)      # dimensions: [num_classes, [datapoints, channels, x, y]]
    vals = np.stack(vals).transpose((1, 0, 2, 3, 4))  # dimensions: [datapoints, num_classes, channels, x, y]
    
    # save results
    save_data('d3', fn, vals)
    