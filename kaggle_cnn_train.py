from util.util_cnn import init_and_train_and_store
from multiprocessing import Pool

# parameters
seeds = [0, 1]
channel_configs = [([2,2], [3,3]), ([2,2], [2,3])]

# cartesian product
configs = [(s, *cc) for s in seeds for cc in channel_configs]

with Pool(3) as p:
    p.map(init_and_train_and_store, *configs)

