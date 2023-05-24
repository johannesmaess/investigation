import quantus

from util.quantus import *
from util.naming import *
from util.common import *
from util.util_pickle import *
from util.util_cnn import first_mnist_batch, load_mnist_v4_models

from multiprocessing import Manager, Pool

#### define LRP & PixFlip params

perturb_baseline='black'
# perturb_baseline='max_diff'

gamma_mode = 'cascading_gamma'
# gamma_mode = 'individual_gamma'

gammas = gammas40
# gammas = gammas_0_1_21_inf

if   gammas is gammas40:          g_str = 'gammas40'
elif gammas is gammas_0_1_21_inf: g_str = 'gammas_0_1_21_inf'
else: assert 0

perturb_func = None
if perturb_baseline=='max_diff':
    perturb_func = max_diff_replacement_by_indices


k = 1000000 # num points = all
data, target = first_mnist_batch(batch_size=k)

y_batch = target.detach().numpy()
x_batch =   data.detach().numpy().reshape((-1, 1, 28, 28))

pixFlipMetric = quantus.PixelFlipping(disable_warnings = True, perturb_func=perturb_func, perturb_baseline=perturb_baseline)

def flipScores(x_batch, y_batch, a_batch):
    return pixFlipMetric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        device=device
    )


def func(mode_str, dict_rels, shared_dict_scores):
    n = 100
    np.random.seed(42)
    scores = []
    a_batch = dict_rels[mode_str].numpy()
    assert len(x_batch) == len(y_batch) == len(a_batch), f"Not matching shape: {x_batch.shape}, {y_batch.shape}, {a_batch.shape}"

    print("Starting mode:", mode_str)
    for i in tqdm(range(0, len(a_batch), n)):
        # create minibatches to pass to pixflip system
        xb = x_batch[i:i+n]
        yb = y_batch[i:i+n]
        ab = a_batch[i:i+n]
        assert len(xb) == len(yb) == len(ab), f"Not matching shape: {xb.shape}, {yb.shape}, {ab.shape}"

        minibatch_scores = flipScores(xb, yb, ab)
        scores.append(minibatch_scores)

    shared_dict_scores[mode_str] = { 'PixFlip': np.concatenate(scores, axis=0) }
    print("Done with mode:", mode_str)

if __name__ == '__main__':
    # load v4 models
    model_dict = load_mnist_v4_models()
    model = model_dict[d3_tag]
    model.eval()

    ### load relevancies

    assert (relevancies_per_mode := load_data('d3', f'Rel0__m0_to_0__{gamma_mode}__{g_str}'))

    print(f"Running with {len(relevancies_per_mode.keys())} modes.")
    print("Relevancies shape:", relevancies_per_mode['LRP-0'].shape)

    assert len(relevancies_per_mode['LRP-0']) ==len(x_batch) == len(y_batch), \
        f"{relevancies_per_mode['LRP-0'].shape}, {x_batch.shape}, {y_batch.shape}"
    

    with Manager() as manager:
        shared_dict_scores = manager.dict()

        num_processes = 4  # Set the desired number of parallel threads

        args = [(mode_str, relevancies_per_mode, shared_dict_scores) for mode_str in relevancies_per_mode.keys()]
        with Pool(processes=num_processes) as pool:
            pool.starmap(func, args)

        dict_scores = dict(shared_dict_scores)

    save_data('d3', f'PixFlipScores__{perturb_baseline}__{gamma_mode}__{g_str}', dict_scores)
    print('Saved.')