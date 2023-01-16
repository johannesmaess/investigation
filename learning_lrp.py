import torch
import numpy as np

from tqdm import tqdm
from functools import partial

from explanations_can_be_manipulated.src.nn.enums import ExplainingMethod
from explanations_can_be_manipulated.src.nn.utils import get_expl

method = ExplainingMethod.lrp

def perturb_point(point, k=1, mode='normal', var=.5, clip=[0,1], activated_subnetwork=True, seed=1):
    with torch.no_grad():
        torch.manual_seed(seed)
        
        if 'normal' in mode:
            perturbation = torch.normal(0, var, size=(k, *point.shape))

        if activated_subnetwork:
            perturbation *= point[None] > 0
        
        point_perturbed = point[None] + perturbation
        
        if clip is not None:
            if clip[0] is not None:   point_perturbed = point_perturbed.clip(min=clip[0])
            if clip[1] is not None:   point_perturbed = point_perturbed.clip(max=clip[1])
        
        if k==1: point_perturbed = point_perturbed[0]
        
    point_perturbed.requires_grad_(point.requires_grad)
    return point_perturbed

def metrics_shap(model, test_loader_shap, metric_tags):
    """
    Compute mean over test set, for a set of shapley value metric functions.
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
                expl_other, _, _, other_idx = get_expl(model, x, method, desired_index='other', forward_result=output, full=True)
                shap_other = torch.Tensor(shap_per_class[np.arange(len(other_idx)), other_idx][:, 0])

                expl_both = torch.vstack((expl_pred, expl_other))
                shap_both = torch.vstack((shap_pred, shap_other))

            # calculate metric, take mean over batch
            metrics_for_batch = []
            for f, tag in zip(metric_funcs, metric_tags):
                if 'pred'    in tag: m = f(shap_pred,  expl_pred )
                elif 'other' in tag: m = f(shap_other, expl_other)
                elif 'both'  in tag: m = f(shap_both,  expl_both )
                
                if 'corr' in tag: m *= -1 # we want optimization=minimization
                metrics_for_batch.append(m.mean())
            metrics_per_batch.append(metrics_for_batch)
    return np.mean(metrics_per_batch, axis=0)

def metrics_self(model, test_loader, metric_tags):
    """
    Compute mean over test set, for a set of self-supervised metrics.
    """
    metric_funcs = [metric_func_from_tag(tag) for tag in metric_tags]

    compute_other = np.any([('-sensitivity' in tag) for tag in metric_tags])
    compute_perturbation = np.any([('perturb' in tag) for tag in metric_tags])

    metrics_per_batch = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0]
            # lrp heatmaps for predicted classes
            expl_pred, output, class_idx = get_expl(model, x, method, full=True, R='one')

            if compute_other:
                expl_other, _, _, other_idx = get_expl(model, x, method, full=True, R='one', desired_index='other', forward_result=output)

            if compute_perturbation:
                x_perturbed = perturb_point(x)
                expl_perturbed_pred, output_perturbed, _, _ = get_expl(model, x_perturbed, method, full=True, R='one', desired_index=class_idx)
                if compute_other:
                    expl_perturbed_other, _, _, _ =        get_expl(model, x_perturbed, method, full=True, R='one', desired_index=other_idx, forward_result=output_perturbed)

            # print(class_idx)
            # print(other_idx)

            # calculate metric, take mean over batch
            metrics_for_batch = []
            for f, tag in zip(metric_funcs, metric_tags):
                if 'pred-perturbation-insensitivity' in tag: m = f(expl_pred, expl_perturbed_pred).sum()
                elif        'pred-class-sensitivity' in tag: m = f(expl_pred, expl_other).sum()
                elif   'perturbed-class-sensitivity' in tag: m = f(expl_perturbed_pred, expl_perturbed_other).sum()

                metrics_for_batch.append(m.mean())
            metrics_per_batch.append(metrics_for_batch)

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
        return lambda x, y: (n(x)-n(y)).abs().mean()
    if 'mse' in tag:
        return lambda x, y: ((n(x)-n(y))**2).mean()
    if 'corr' in tag: 
        return lambda x, y: corr_coef_differentiable(n(x), n(y))

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
    err_func = partial(metrics_shap, metric_tags=metric_tags_shap)

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
                    expl_other, _, _, class_idx_other = get_expl(model, x, method, desired_index='other', forward_result=output, full=True)
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
                loss = 0

                if 'perturb' in loss_tag: # precompute for loss functions that use perturbed point
                    x_perturbed = perturb_point(x)
                    expl_perturbed_pred, output_perturbed, _ = get_expl(model, x_perturbed, method, full=True, desired_index=class_idx)
                
                get_float_after = lambda s: float(loss_tag.split(s)[1].split(',')[0])

                if 'pred-perturbation-insensitivity' in loss_tag:
                    weighting = get_float_after('pred-perturbation-insensitivity=')
                    loss += weighting * f(expl_pred, expl_perturbed_pred).sum()
                if 'pred-class-sensitivity' in loss_tag:
                    weighting = get_float_after('pred-class-sensitivity=')
                    expl_other, _, _, _ = get_expl(model, x, method, desired_index='other', forward_result=output, full=True)
                    loss -= weighting * f(expl_pred, expl_other)
                if 'perturbed-class-sensitivity' in loss_tag:
                    weighting = get_float_after('perturbed-class-sensitivity=')
                    expl_perturbed_other, _, _, _ = get_expl(model, x_perturbed, method, desired_index='other', forward_result=output_perturbed, full=True)
                    loss -= weighting * f(expl_perturbed_pred, expl_perturbed_other)
                
                
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