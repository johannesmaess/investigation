import pickle
import numpy as np
import os
from tqdm import tqdm

from util.naming import PICKLE_PATH


def save_data(model_tag, data_tag, data):
    path = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    assert(np.all(data == load_data(model_tag, data_tag)))

def load_data(model_tag, data_tag):
    path = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}.pickle')
    with open(path, 'rb') as handle:
        data_loaded = pickle.load(handle)
    return data_loaded

def get_shap_configs(model_tag):
    return np.unique([fn.split('__batch-')[0] for fn in tqdm(os.listdir(os.path.join(PICKLE_PATH, model_tag))) if 'shap' in fn])

def load_shaps(model_tag, config):
    shaps = []
    for fn in tqdm(sorted(os.listdir(PICKLE_PATH + model_tag))):
        if config in fn:
            data_tag = fn[:-7]
            data = load_data(model_tag, data_tag)
            shaps.append(data)        
    return np.vstack(shaps)
