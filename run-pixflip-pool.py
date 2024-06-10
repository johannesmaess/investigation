import quantus

import torch

from util.quantus import *
from util.naming import *
from util.common import *
from util.util_pickle import *
from util.util_cnn import first_mnist_batch, load_mnist_v4_models

from multiprocessing import Manager, Pool

#### define PixFlip params

# perturb_baseline='black'
# perturb_baseline='max_diff'
# perturb_baseline='mean'
# perturb_baseline='interpolation'
perturb_baseline='expanding_perturbation_interpolation'
inpaint_radius = 3

#### define XAI params

exp_mode = "ig"

if exp_mode == 'lrp':
    # gamma_mode = 'individual_gamma'
    gamma_mode = 'cascading_gamma'
    # gamma_mode = 'all_gamma' # restrict cascading_gamma to where the whole network uses gamma

    # gammas = gammas40
    # gammas = gammas_0_1_21_inf
    gammas = gammas80

    if   gammas is gammas40:          g_str = 'gammas40'
    if   gammas is gammas80:          g_str = 'gammas80'
    elif gammas is gammas_0_1_21_inf: g_str = 'gammas_0_1_21_inf'
    else: assert 0
elif exp_mode == 'ig':
    ig_steps = 50

perturb_func = None

metric = quantus.PixelFlipping
if 'expanding_perturbation' in perturb_baseline:
    metric = PixelFlippingExpandingPerturbationSet

if perturb_baseline=='max_diff':
    perturb_func = max_diff_replacement_by_indices
elif 'interpolation' in perturb_baseline:
    perturb_func = interpolation_replacement_by_indices

num_points = 100
data, target = first_mnist_batch(batch_size=num_points)
y_batch = target.detach().numpy()
x_batch =   data.detach().numpy().reshape((-1, 1, 28, 28))


if exp_mode == 'lrp':
    load_mode = (gamma_mode if gamma_mode != "all_gamma" else "cascading_gamma")
    load_params = f"{load_mode}__{g_str}"
    save_params = f"{gamma_mode}__{g_str}"
if exp_mode == 'ig':
    load_params = f"testset100__ig{ig_steps}"
    save_params = f"ig{ig_steps}"
if exp_mode == 'shap':
    load_params = f"testset__shap__background_size-100__batch_size-10"
    save_params = f"shap__background_size-100__batch_size-10"
    
perturb_params = ''
if 'interpolation' in perturb_baseline: 
    perturb_params = f'_{inpaint_radius}'
save_key = f'PixFlipScores__testset{num_points}__{perturb_baseline}{perturb_params}__{save_params}'

def save_func(shared_dict_scores, postfix=""):
    save_data('d3', save_key + postfix, dict(shared_dict_scores))


pixFlipMetric = metric(disable_warnings = True, perturb_func=perturb_func, perturb_baseline=perturb_baseline,
                                      perturb_func_kwargs = {
                                          'inpaint_radius': inpaint_radius
                                      })

def flipScores(model, x_batch, y_batch, a_batch):
    return pixFlipMetric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        device=device
    )


def func(model, mode_str, dict_rels, shared_dict_scores, num_points, save_func=lambda *args: 0):
    n=10
    np.random.seed(42)
    scores = []
    a_batch = dict_rels[mode_str]
    if isinstance(a_batch, torch.Tensor): a_batch = a_batch.numpy()
    a_batch = a_batch[:num_points]
    assert len(x_batch) == len(y_batch)
    assert num_points <= len(y_batch), f"Requesting more inputs than available: {x_batch.shape}, {y_batch.shape}, {num_points}"
    assert num_points <= len(a_batch), 'Requesting more evaluations than available'

    if isinstance(a_batch, torch.Tensor):
        a_batch = a_batch.numpy()

    print("Starting mode:", mode_str)
    for i in tqdm(range(0, num_points, n)):
        # create minibatches to pass to pixflip system
        xb = x_batch[i:i+n]
        yb = y_batch[i:i+n]
        ab = a_batch[i:i+n]
        print(f'{xb.shape}, {yb.shape}, {ab.shape}')
        assert len(xb) == len(yb) == len(ab), f"Not matching shape: {xb.shape}, {yb.shape}, {ab.shape}"

        minibatch_scores = flipScores(model, xb, yb, ab)
        scores.append(minibatch_scores)

    shared_dict_scores[mode_str] = { 'PixFlip': np.concatenate(scores, axis=0) }
    save_func(shared_dict_scores, '__intermediate')
    print("Done with mode:", mode_str)
    

if __name__ == '__main__':
    # load v4 models
    model_dict = load_mnist_v4_models()
    model = model_dict[d3_tag]
    del model_dict
    model.eval()

    ### load relevancies
        
    key = f'Rel0__m0_to_0__{load_params}'
    relevancies_per_mode = load_data('d3', key)
    if relevancies_per_mode is False:
        print('Falling back on unnormalized Relevances. This should not matter, because PF does not care about the sum of the heatmap.')
        relevancies_per_mode = load_data('d3', key+'__unnormalized')
    
    assert relevancies_per_mode is not False, f'Failed to load: {key}'
    print('Loaded: ', key)
    
    if exp_mode == 'lrp' and gamma_mode == 'all_gamma': relevancies_per_mode = {mode_str: v for mode_str, v in relevancies_per_mode.items() if ("l<16" in mode_str)}
    
    print(f"Running with {len(relevancies_per_mode.keys())} modes.")

    with Manager() as manager:
        shared_dict_scores = manager.dict()

        num_processes = 14  # Set the desired number of parallel threads
        # 70 threads, bs 10 -> 250 sec pro batch
        # 50 threads, bs 10 -> 190 sec pro batch

        args = [(model, mode_str, relevancies_per_mode, shared_dict_scores, num_points, save_func) for mode_str in relevancies_per_mode.keys()]
        with Pool(processes=num_processes) as pool:
            pool.starmap(func, args)
        
        save_func(shared_dict_scores)
        
    print('Done:', save_key)