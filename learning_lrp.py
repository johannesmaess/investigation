import torch
import numpy as np

from tqdm import tqdm
from functools import partial

from explanations_can_be_manipulated.src.nn.enums import ExplainingMethod
from explanations_can_be_manipulated.src.nn.utils import get_expl

method = ExplainingMethod.lrp

def shap_error(model, test_loader_shap, distance_tags, other=False):
    """
    Compute mean error over test set.
    """
    distance_funcs = [distance_func_from_tag(tag) for tag in distance_tags]

    distances_per_batch = []
    with torch.no_grad():
        for x, _, shap_per_class in test_loader_shap:
            # lrp heatmaps for predicted classes
            expl_pred, output, class_idx = get_expl(model, x, method, full=True)
            # shapley value heatmaps for predicted classes
            shap_pred = torch.Tensor(shap_per_class[np.arange(len(class_idx)), class_idx][:, 0])

            if not other:
                expl, shap = expl_pred, shap_pred
            else:
                expl_other, _, _, other_idx = get_expl(model, x, method, desired_index='other', forward_result=(output, class_idx), full=True)
                shap_other = torch.Tensor(shap_per_class[np.arange(len(other_idx)), other_idx][:, 0])

                expl = torch.vstack((expl_pred, expl_other))
                shap = torch.vstack((shap_pred, shap_other))

            # calculate distance, take mean over batch
            distances_for_batch = [f(shap, expl).mean() for f in distance_funcs]
            distances_per_batch.append(distances_for_batch)
    return np.mean(distances_per_batch, axis=0)

def corr_coef_differentiable(x, y):
    """
    If one dimensional vectors are passed, we simply calculate correlation.
    If Multi-dimensional vectors are passed, we treat the first dim as a batch dim.
    We calculate the Correlation per point-pair in the batches, and return their Mean.
    """
    
    if len(x.shape) >= 2: x = x.view((len(x), -1))
    if len(y.shape) >= 2: y = y.view((len(y), -1))
    assert x.shape == y.shape
    batched = int(len(x.shape) == 2)
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy, axis=batched) / (torch.sqrt(torch.sum(vx ** 2, axis=batched)) * torch.sqrt(torch.sum(vy ** 2, axis=batched)))

    if batched: return corr.mean()
    else:       return corr

def proj(gamma):
    with torch.no_grad():
        if gamma.data < 0: gamma.zero_()

def distance_func_from_tag(tag):
    if 'mae' in tag:
        return lambda x, y: (x-y).abs()
    if 'mse' in tag:
        return lambda x, y: ((x-y)**2).mean()
    if 'corr' in tag: 
        return lambda x, y: corr_coef_differentiable(x, y)

    raise 'Invalid distance tag: ' + tag
    
def train_lrp(model, test_loader_shap, optimizer, parameters,
            loss_func = 'shap--pred--mse', 
            T = 8e4,
            err_tags = ['shap--pred--mae', 
                        'shap--pred--mse', 
                        'shap--pred--corr'],
            ):

    # prework
    batch_size = len(next(iter(test_loader_shap))[0]) # look how long first batch is
    n_batches = len(test_loader_shap)
    n_epochs = int(T / n_batches / batch_size)
    if T < n_epochs * n_batches * batch_size: n_epochs += 1
    print('n_epochs', n_epochs)
    err_tags_shap = [tag for tag in err_tags if 'shap--' in tag]
    err_func = partial(shap_error, distances=err_tags_shap)

    # to save state
    gammas, gammas_t = [], []
    errs, errs_t = [], []
    
    def log_err():
        nonlocal model, test_loader_shap, errs, parameters
        print(f"{c_samples}/{T}. Params: {[round(float(param.detach()), 4) for param in parameters]}.")
        if not err_tags: return

        # evaluate for all passed error functions
        errors = err_func(model, test_loader_shap)
        errs.append(errors)
        errs_t.append(c_samples)
        print(f"Metrics: {np.array(errors).round(8)}")

    # initial state
    c_samples = 0
    log_err()

    for i in tqdm(range(n_epochs)):
        for j, (x, target, shap_per_class) in (enumerate(test_loader_shap)):
            x = x.reshape((batch_size, 1, 28, 28)).data

            optimizer.zero_grad()

            # calculate entropy loss
            expl, output, class_idx = get_expl(model, x, method, full=True)

            # construct 'probability distribution'
            if loss_func=='entropy':
                print(expl.view(-1)[:10])
                dist = expl.abs()
                dist /= dist.sum((1,2), keepdim=True)
                ent = torch.special.entr(dist)
                print(ent)
                loss = ent.sum()
                
            elif 'shap--' in loss_func:
                assert 'shap--pred' in loss_func, 'Invalid loss_func: ' + loss_func

                # find shapley value heatmaps for predicted classes
                ground_truth = shap_per_class[np.arange(batch_size), class_idx][:, 0]
                ground_truth = torch.Tensor(ground_truth)

                if 'shap--pred-contrastive' in loss_func: # TODO finish
                    # compute lrp heatmaps for *other* classes
                    expl_other, _, _, class_idx_other = get_expl(model, x, method, desired_index='other', forward_result=(output, class_idx), full=True)
                    # find shapley value heatmaps for *other* classes
                    shap_other = shap_per_class[np.arange(batch_size), class_idx_other][:, 0]
                    
                    # extract float that follows '-contrastive'
                    weight = loss_func.split('-contrastive')[1].split('-')[0]
                
                
                # normalize per heatmap
                if 'normalize' in loss_func:
                    expl = expl / expl.abs().sum(axis=(1,2), keepdim=True)
                    ground_truth = ground_truth / ground_truth.abs().sum(axis=(1,2), keepdim=True)
                
                if 'mse' in loss_func:
                    loss = ((ground_truth - expl)**2).sum()  
                elif 'mae' in loss_func:
                    loss = (ground_truth - expl).abs().sum()
                elif 'corr' in loss_func:
                    loss = corr_coef_differentiable(ground_truth.view(-1), expl.view(-1))
                else:
                    raise 'Invalid loss_func: ' + loss_func
                
            # update gammas
            loss.backward()
            optimizer.step()
            # project into valid range
            for gamma in parameters: proj(gamma)
            
            gammas.append([float(param.detach()) for param in parameters])
            gammas_t.append(c_samples)
                
            if 0:
                clear_output()
                fig, ax = plt.subplots()
                ax.plot(gammas)
                ax.twinx().plot(losses, c='r')
                plt.plot()
                
            c_samples += batch_size
            if c_samples > T: break
            
        # measure and log performance after end of ever epoch
        log_err()

    return model, gammas, gammas_t, errs, errs_t