from quantus.helpers import utils
import numpy as np

def batch_auc(batch):
    return [utils.calculate_auc(np.array(curve)) for curve in batch]
def batch_mean_auc(batch):
    return np.mean(batch_auc(batch))