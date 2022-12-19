import pickle
import numpy as np
import os

def save_data(model_tag, data_tag, data):
    path = os.path.join('./pickle', model_tag, f'{data_tag}.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    assert(np.all(data == load_data(model_tag, data_tag)))

def load_data(model_tag, data_tag):
    path = os.path.join('./pickle', model_tag, f'{data_tag}.pickle')
    with open(path, 'rb') as handle:
        data_loaded = pickle.load(handle)
    return data_loaded