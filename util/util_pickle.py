import pickle
import numpy as np
import os
from tqdm import tqdm

from util.naming import PICKLE_PATH

def save_data(model_tag, data_tag, data):
    path = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for d, l in zip(data, load_data(model_tag, data_tag)):
        assert(np.all(d == l))

def load_data(model_tag, data_tag):
    path = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}.pickle')
    with open(path, 'rb') as handle:
        data_loaded = pickle.load(handle)
    return data_loaded

def get_shap_configs(model_tag):
    return np.unique([fn.split('__batch-')[0] for fn in tqdm(os.listdir(os.path.join(PICKLE_PATH, model_tag))) if 'shap__background' in fn])

def get_matching_tags(model_tag, data_tags_partial):
    if type(data_tags_partial) is not list: data_tags_partial = [data_tags_partial]
    matches = lambda fn: np.all([data_tag_partial in fn for data_tag_partial in data_tags_partial])
    fns = sorted(os.listdir(os.path.join(PICKLE_PATH, model_tag)))
    return np.unique([fn[:-7] for fn in tqdm(fns) if matches(fn)])

def load_shaps(model_tag, config):
    shaps = []
    for fn in tqdm(sorted(os.listdir(PICKLE_PATH + model_tag))):
        if config in fn:
            data_tag = fn[:-7]
            data = load_data(model_tag, data_tag)
            shaps.append(data)        
    return np.vstack(shaps)
