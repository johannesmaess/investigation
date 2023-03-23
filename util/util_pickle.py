import pickle
import numpy as np
import os
import glob
from tqdm import tqdm

from util.naming import PICKLE_PATH

def save_data(model_tag, data_tag, data):
    path = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(model_tag, data_tag, partitioned=False):
    if partitioned:
        path_regex = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}__partition*')    
        matches = sorted(list(glob.glob(path_regex)))
        ws = sorted(list(set([m[:-11] for m in matches])))
        num_ps = len(set([m[-10:-7] for m in matches]))

        res = []
        for w in ws:
            matches_w = sorted(list(glob.glob(f'{w}*')))
            assert len(matches_w) == num_ps, f"Not same number of points for all ws: {len(matches_w)} != {num_ps}"

            r = []
            for m in matches_w:
                with open(m, 'rb') as handle:
                    r.append(pickle.load(handle))

            r = np.concatenate(r, axis=1)
            print(r.shape)
            res.append(r)
        
        return np.concatenate(res, axis=0)
    
    path = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}.pickle')
    if not os.path.exists(path):
        return False
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
