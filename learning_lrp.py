import torch
import numpy as np

from tqdm import tqdm
from functools import partial

from explanations_can_be_manipulated.src.nn.enums import ExplainingMethod
from explanations_can_be_manipulated.src.nn.utils import get_expl

method = ExplainingMethod.lrp

def shap_error(model, test_loader_shap, metric_tags):
    """
    Compute mean error over test set.
    """
    metric_funcs = [metric_func_from_tag(tag) for tag in metric_tags]
    compute_other = np.any([('other' in tag or 'both' in tag) for tag in metric_tags])

    metrics_per_batch = []
    with torch.no_grad():
        for x, _, shap_per_class in test_loader_shap:
            # lrp heatmaps for predicted classes
            expl_pred, output, class_idx = get_expl(model, x, method, full=True)
            # shapley value heatmaps for predicted classes
            shap_pred = torch.Tensor(shap_per_class[np.arange(len(class_idx)), class_idx][:, 0])

            if compute_other:
                expl_other, _, _, other_idx = get_expl(model, x, method, desired_index='other', forward_result=(output, class_idx), full=True)
                shap_other = torch.Tensor(shap_per_class[np.arange(len(other_idx)), other_idx][:, 0])

                expl_both = torch.vstack((expl_pred, expl_other))
                shap_both = torch.vstack((shap_pred, shap_other))

            # calculate metric, take mean over batch
            metrics_for_batch = []
            for f, tag in zip(metric_funcs, metric_tags):
                if 'pred'    in tag: m = f(shap_pred,  expl_pred )
                elif 'other' in tag: m = f(shap_other, expl_other)
                elif 'both'  in tag: m = f(shap_both,  expl_both )
                metrics_for_batch.append(m.mean())
            metrics_per_batch.append(metrics_for_batch)

            break
    return np.mean(metrics_per_batch, axis=0)

def corr_coef_differentiable(x, y):
    """
    If one dimensional vectors are passed, we simply calculate correlation.
    If Multi-dimensional vectors are passed, we treat the first dim as a batch dim.
    We calculate the Correlation per point-pair in the batches, and return their Mean.
    """
    
    if x.ndim > 2: x = x.view((len(x), -1))
    if y.ndim > 2: y = y.view((len(y), -1))
    assert x.shape == y.shape
    batched = int(x.ndim == 2)
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy, axis=batched) / (torch.sqrt(torch.sum(vx ** 2, axis=batched)) * torch.sqrt(torch.sum(vy ** 2, axis=batched)))

    if batched: return corr.mean()
    else:       return corr

def metric_func_from_tag(tag):
    """
    'tag' is a string that identifies the desired metric function.
    Ex: "mse abs-norm"
    """

    # set normalization function
    if 'abs-norm' in tag:
        def n(z):
            z = z.view((len(z), -1)).abs()
            return z / z.sum(axis=1, keepdim=True)
    elif 'norm' in tag:
        def n(z):
            z = z.view((len(z), -1))
            return z / z.sum(axis=1, keepdim=True)
    else:
        # no normalization
        n = lambda z: z

    # set metric function (composed with normalization function)
    if 'mae' in tag:
        return lambda x, y: (n(x)-n(y)).abs()
    if 'mse' in tag:
        return lambda x, y: ((n(x)-n(y))**2).mean()
    if 'corr' in tag: 
        return lambda x, y: 1 - corr_coef_differentiable(n(x), n(y))

    raise 'Invalid metric tag: ' + tag

def proj(gamma):
    with torch.no_grad():
        if gamma.data < 0: gamma.zero_()
    
def train_lrp(model, test_loader_shap, optimizer, parameters,
            loss_tag = 'shap--pred--mse', 
            T = 8e4,
            metric_tags = ['shap--pred--mae', 
                           'shap--pred--mse', 
                           'shap--pred--corr'],
            ):

    # prework
    batch_size = len(next(iter(test_loader_shap))[0]) # look how long first batch is
    n_batches = len(test_loader_shap)
    n_epochs = int(T / n_batches / batch_size)
    if T > n_epochs * n_batches * batch_size: n_epochs += 1
    print('n_epochs', n_epochs)
    metric_tags_shap = [tag for tag in metric_tags if 'shap--' in tag]
    err_func = partial(shap_error, metric_tags=metric_tags_shap)

    # to save state
    gammas, gammas_t = [], []
    errs, errs_t = [], []
    
    def log_err():
        nonlocal model, test_loader_shap, errs, parameters
        print(f"{c_samples}/{T}. Params: {[round(float(param.detach()), 4) for param in parameters]}.")
        if not metric_tags: return

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
            optimizer.zero_grad()

            # calculate explanation of predicted class
            x = x.reshape((batch_size, 1, 28, 28)).data
            expl_pred, output, class_idx = get_expl(model, x, method, full=True)

            if loss_tag=='entropy': # TODO haven't used this in a long time, not sure if it works.
                # construct 'probability distribution'
                dist = expl_pred.abs()
                dist /= dist.sum((1,2), keepdim=True)
                ent = torch.special.entr(dist)
                print(ent)
                loss = ent.sum()
                
            elif 'shap' in loss_tag:
                # find shapley value heatmaps for predicted classes
                shap_pred = torch.Tensor(shap_per_class[np.arange(batch_size), class_idx][:, 0])

                if 'other' in loss_tag or 'both' in loss_tag:
                    # compute lrp heatmaps for *other* classes
                    expl_other, _, _, class_idx_other = get_expl(model, x, method, desired_index='other', forward_result=(output, class_idx), full=True)
                    # find shapley value heatmaps for *other* classes
                    shap_other = torch.Tensor(shap_per_class[np.arange(batch_size), class_idx_other][:, 0])
                
                    expl_both = torch.vstack((expl_pred, expl_other))
                    shap_both = torch.vstack((shap_pred, shap_other))
                
                f = metric_func_from_tag(loss_tag)
                if 'pred'    in loss_tag: loss = f(shap_pred,  expl_pred ).sum()
                elif 'other' in loss_tag: loss = f(shap_other, expl_other).sum()
                elif 'both'  in loss_tag: loss = f(shap_both,  expl_both ).sum()

            elif 'self' in loss_tag:
                # Self-supervised learning of LRP parameters. 
                # For the loss function, only the qualities of (a related set of) heatmaps is used to improve the heatmaps.

                f = metric_func_from_tag(loss_tag)
                x_noisy = 
                
                
                
            # update gammas
            loss.backward()
            optimizer.step()
            # project into valid range
            for gamma in parameters: proj(gamma)
            
            gammas.append([float(param.detach()) for param in parameters])
            gammas_t.append(c_samples)
                
            c_samples += batch_size
            if c_samples > T: break
            
        # measure and log performance after end of ever epoch
        log_err()

    return model, gammas, gammas_t, errs, errs_t