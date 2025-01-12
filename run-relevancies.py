from tqdm import tqdm

from util.util_cnn import first_mnist_batch, load_mnist_v4_models
from util.util_lrp import layerwise_forward_pass, compute_relevancies

from util.naming import *
from util.util_pickle import *

from multiprocessing import Manager, Pool

# load v4 models
model_dict = load_mnist_v4_models()
model_d3 = model_dict['cb1-8-8-8_cb2-16-16-16_seed-0']

### RUN ###

gamma_mode = 'cascading_gamma'
# gamma_mode = 'individual_gamma'

# gammas = gammas_0_1_21_inf
# gammas = gammas40
gammas = gammas80

if   gammas is gammas40:          g_str = 'gammas40'
if   gammas is gammas80:          g_str = 'gammas80'
elif gammas is gammas_0_1_21_inf: g_str = 'gammas_0_1_21_inf'
else: assert 0

modes = {0: 'LRP-0'}
for i, l_ub in enumerate(d3_after_conv_layer):
    for j, g in enumerate(gammas):
        if g=='inf': g = 1e8
        if g!=0:
            g = np.round(g, 8)
            if gamma_mode=='cascading_gamma':  modes[i*1000+j] = f'Gamma.            l<{l_ub} gamma={g}'
            if gamma_mode=='individual_gamma': modes[i*1000+j] = f'Gamma. l>{l_ub-2} l<{l_ub} gamma={g}'

data, target = first_mnist_batch(batch_size=100)

A, layers = layerwise_forward_pass(model_d3, data)
L = len(layers)

def func(mode, shared_dict):
    rels = compute_relevancies(mode=mode, layers=layers, A=A, output_rels='correct class', target=target, return_only_l=0)
    if mode!="info": shared_dict[mode] = rels
    print("Done with mode:", mode)

if __name__ == '__main__':
    with Manager() as manager:
        shared_dict = manager.dict()

        num_processes = 20  # Set the desired number of parallel threads
        pool = Pool(processes=num_processes)

        args = [(mode, shared_dict) for mode in list(modes.values())]
        with Pool(processes=num_processes) as pool:
            pool.starmap(func, args)

        relevancies_per_mode = dict(shared_dict)

    print(relevancies_per_mode.keys())
    print(len(relevancies_per_mode.keys()))

    save_data('d3', f'Rel0__m0_to_0__{gamma_mode}__{g_str}', relevancies_per_mode)
    print("Done saving.")
