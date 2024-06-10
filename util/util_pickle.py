import pickle
import numpy as np
import os
import glob
from tqdm import tqdm

from util.naming import PICKLE_PATH

def save_data(model_tag, data_tag, data, partition=None):
    if partition is not None:
        assert type(partition) is tuple, "Partition has to be a tuple ( weight, point )."
        w,p = partition
        data_tag += f"__partition_{w:03d}_{p:03d}"
        
        print("Saving:", model_tag, data_tag)
    
    path = os.path.join(PICKLE_PATH, model_tag)
    if not os.path.exists(path): os.mkdir(path)
    path = os.path.join(path, f'{data_tag}.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Saved:', path)

def load_data(model_tag, data_tag, partition=None, partitioned=False, verbose=True, warn=True):
    if verbose:
        print("Attempt loading:", model_tag, data_tag)
        # assert data_tag != 'svals__m0_to_1__cascading_gamma__gammas40'
    if partitioned:
        path_regex = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}__partition*')    
        matches = sorted(list(glob.glob(path_regex)))
        ws = sorted(list(set([m[:-11] for m in matches])))
        num_ps = len(set([m[-10:-7] for m in matches]))

        if len(ws) == 1 and ws[0][-3:] == '000': # we are loading Pixel Flipping scores
            res = {}
            for m in matches:
                with open(m, 'rb') as handle:
                    d = pickle.load(handle)
                    assert type(d) is dict, "Expected pix flip scores in dictionary form!"
                    res.update(d)
            return res
        
        # if we are reading in Svals into one unified array, make sure to pad them with zeros to a unified length.
        # if we are reading LRP matrices, num_vals will stay 0.
        num_vals = 0
        for m in matches:
            with open(m, 'rb') as handle:
                d = pickle.load(handle)
                if d.ndim < 4: break
                
                num_vals = max(num_vals, d.shape[3])
                del d

        res = []
        expected_shape = None
        for w in ws[:]:
            print(w)
            matches_w = sorted(list(glob.glob(f'{w}*')))
            assert len(matches_w) == num_ps, f"Not same number of points for all ws: {len(matches_w)} != {num_ps}. matches: {[(m[-14:-11], m[-10:-7]) for m in matches_w]}"
            
            r = []
            for m in matches_w:
                with open(m, 'rb') as handle:
                    d = pickle.load(handle)
                    if num_vals != 0: 
                        d = np.pad(d, [(0,0), (0,0), (0,0), (0, num_vals-d.shape[3])])
                    if expected_shape is not None:
                        assert expected_shape == d.shape, f"Last loaded shape is not equal to other shapes: {expected_shape}, {d.shape}"
                    else: 
                        expected_shape = d.shape
                    r.append(d)

            r = np.concatenate(r, axis=1)
            res.append(r)
        
        return np.concatenate(res, axis=0)
    
    if partition is not None:
        assert type(partition) is tuple, "Partition has to be a tuple ( weight, point )."
        w,p = partition
        data_tag += f"__partition_{w:03d}_{p:03d}"
        
        print("Loading:", model_tag, data_tag)
    
    path = os.path.join(PICKLE_PATH, model_tag, f'{data_tag}.pickle')
    if not os.path.exists(path):
        if warn: print('Missing file:', path)
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
