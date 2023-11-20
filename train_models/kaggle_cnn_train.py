from util.util_cnn import init_and_train_and_store
from multiprocessing import Pool
from functools import partial

# append parent directory to path
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # parameters
    seeds = [0, 1]
    channel_configs = [([2,2], [3,3]), ([2,2], [2,3])]

    # cartesian product
    configs = [(s, *cc) for s in seeds for cc in channel_configs]

    ### V5
    configs = []
    for seed in [0, 1]:
        for depth in [2,6,8]: #[2,4,6,8]:
            for F in [9, 11]: #[3,5,7]:
                configs.append({
                    'v': 5,
                    'seed': seed,
                    'Fs': [F]*depth
                })

    print(configs)
    with Pool(4) as p:
        p.map(partial(init_and_train_and_store, n_iters=50000, device='mps'), configs)

if __name__ == "__main__":
    main()