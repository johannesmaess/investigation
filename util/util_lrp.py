import numpy as np
import torch
from scipy.sparse import coo_array, coo_matrix
from tqdm import tqdm
import copy

import util.util_tutorial as tut_utils
from util.common import *
from util.util_pickle import load_data, save_data

import matplotlib.pyplot as plt

def layerwise_forward_pass(model, data=None, checks=False, pos_neg=False, get_c=False): # just for MNIST:
    return layerwise_forward_pass_general(list(model.seq), data, checks, pos_neg, get_c, inp_shape=(-1,1,28,28))

def layerwise_forward_pass_general(layers, data=None, checks=False, pos_neg=False, get_c=False, inp_shape=None):
    assert not (pos_neg and get_c)

    layers = tut_utils.toconv(layers)
    if data is None: return layers
    L = len(layers)

    data = data.reshape(inp_shape) if inp_shape else data[:, :, None, None]
    data = data.detach()

    A = [data]+[None]*L
    A_pos = [data]+[None]*L
    A_neg = [None]*(L+1)
    c_list = []

    for l in range(L):
        lay = layers[l]
        if isinstance(lay, torch.nn.Flatten):
            batch_size, *image_size = A[l].shape
            A[l+1] = A[l].reshape((batch_size, np.prod(image_size), 1, 1)).detach()
            A_pos[l+1] = A[l+1]
        else:
            A[l+1] = lay.forward(A[l]).detach()
            A_pos[l+1] = A[l+1]
            
            # calculate positive and negative contributions to neurons seperately
            if (pos_neg or get_c):
                if isinstance(lay, torch.nn.Conv2d):
                    with torch.no_grad():
                        # print(l, lay.weight.shape)
                        lay_pos, lay_neg = copy.deepcopy(lay), copy.deepcopy(lay)
                        lay_pos.weight.data = lay.weight.clone().clip(min=0)
                        lay_pos.bias.data =     lay.bias.clone().clip(min=0)
                        lay_neg.weight.data = lay.weight.clone().clip(max=0)
                        lay_neg.bias.data =     lay.bias.clone().clip(max=0)

                        a_pos = lay_pos.forward(A[l])
                        a_neg = lay_neg.forward(A[l])
                        A_pos[l+1] = a_pos
                        A_neg[l+1] = a_neg

                        assert a_pos.ndim==4, a_pos.shape
                        c = -a_neg / a_pos
                        c[-a_neg >= a_pos] = -np.inf # only include activated neurons
                        c = c.view((len(a_pos), -1)).max(axis=1).values

                        if get_c: c_list.append(c)

    # if checks:
    #     res = model.forward(A[0]).flatten().detach()
    #     res_indirect =      A[-1].flatten().detach()
    #     assert torch.allclose(res, res_indirect, atol=1e-5), f"Too high diff: { np.abs(res - res_indirect).max() }"

    if get_c:
        return np.stack(c_list)

    if pos_neg:
        return A, A_pos, A_neg, layers
    
    return A, layers


def forward_and_explain(model, data, mode):
    A, layers = layerwise_forward_pass(model, data)
    return compute_relevancies(mode, layers, A, output_rels='predicted class', return_only_l=0)

"""
This code was previously used in the notebook to precompute the forward matrices for conv layers.
We might need it in the future.

layers_conv_as_mat = [None]*L
for l in range(L):
    if isinstance(layers[l], torch.nn.Conv2d):
        layers_conv_as_mat[l] = Conv2dAsMatrixLayer(layers[l], A[l][0].shape, A[l+1][0].shape)
        a = layers_conv_as_mat[l].forward(A[l])
        b = layers[l]            .forward(A[l])

        assert a.shape == b.shape, "Inequal shape"
        assert torch.allclose(a, b, atol=1e-5), f"Inequal result, max diff: {(a-b).abs().max()}"

"""



def compute_relevancies(mode, layers, A, output_rels='correct class', target=None, l_out=-1, return_only_l=None, eps=1e-9):
    """
    Applies a LRP backpropagation through all or a subset of layers of the network.
    
    Pass the relevancies in the deepest layer to iterate as "output_rels". Only then, you need to pass "target".
    Specifiy which layer that is by "l_out"  
    Set "return_only_l", to return only the relevancies of this layer. This saves computation for all layers < l.
    """

    L = len(layers)
    if l_out<0: l_out = L+l_out+1
    assert 0<l_out<=L, "Invalid l_out"

    if return_only_l is not None:
        assert 0 <= return_only_l < l_out, f"Invalid return_only_l: {return_only_l}. l_out is {l_out}."

    l_ub = float(mode.split("l<")[1].split(" ")[0]) if "l<" in mode else 1000
    l_lb = float(mode.split("l>")[1].split(" ")[0]) if "l>" in mode else -1000

    if output_rels == 'correct class':
        assert target is not None
        ## mask everything but correct class ##
        output_rels = torch.zeros(A[-1].shape, requires_grad=False)
        output_rels[np.arange(len(target)), target] = A[-1].detach()[np.arange(len(target)), target]
    elif output_rels == 'predicted class':
        ## mask everything but max output class ##
        output_rels = (A[-1] * torch.eq(A[-1], A[-1].max(axis=1, keepdims=True).values)).data
    
    assert isinstance(output_rels, torch.Tensor), f"Pass a valid mode. Not: {type(output_rels)} {output_rels}"
    assert output_rels.shape == A[l_out].shape,   f"Pass a valid Tensor. {output_rels.shape} != {A[l_out].shape}"

    R = [None]*(L+1)
    R[l_out] = output_rels
    
    # summed = R[l_out].flatten(start_dim=1).sum(axis=1)
    # print(torch.quantile(summed, torch.tensor([.01, .05, .1, .25, .5, .75, .95, .99, 1])))

    for l in range(1, l_out)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        # print(l, str(layers[l]).split('(')[0])

        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
            # default: LRP-0.
            rho = lambda p: p;
            helper_layer = tut_utils.newlayer(layers[l], rho)

            if isinstance(layers[l],torch.nn.Conv2d):
                if 'Gamma' in mode and l_lb < l < l_ub:
                    curr_gamma = float(gam) if 'inf' != (gam := mode.split("gamma=")[1].split(" ")[0]) else 1e8
                    if 'print' in mode: print('g', end="")
                    
                    # print('gamma')
                    
                    if 'Gamma.' in mode:
                        rho = lambda p: p + curr_gamma*p.clamp(min=0)
                        helper_layer = tut_utils.newlayer(layers[l], rho)
                    elif 'Gamma mat.' in mode:
                        helper_layer = copy.deepcopy(layers_conv_as_mat[l]) # todo: in the notebook I used a precomputation "layers_conv_as_mat"
                        helper_layer.set_gamma(curr_gamma)

            incr = lambda z: z + eps
            z = incr(helper_layer.forward(A[l]))                            # step 1
            s = (R[l+1]/z).data                                             # step 2
            (z*s).sum().backward(); c = A[l].grad                           # step 3
            R[l] = (A[l]*c).data 

            if mode=='info':
                print(A[l].shape,           '->', z.shape)
                print(A[l].flatten().shape, '->', z.flatten().shape)

        elif isinstance(layers[l], torch.nn.Flatten):
            R[l] = R[l+1].reshape(A[l].shape)
        else:
            R[l] = R[l+1]

        if 'print' in mode: print('\t', l, layers[l])
        
        # summed = R[l].flatten(start_dim=1).sum(axis=1)
        # print(torch.quantile(summed, torch.tensor([.01, .05, .1, .25, .5, .75, .95, .99, 1])))
        
        if return_only_l == l:
            return R[l]

    ## layer 0 => zB rule -> My implementation might suck though ###

    # print(0, str(layers[0]).split('(')[0])

    A[0] = (A[0].data).requires_grad_(True)
    # lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    # hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    lb = torch.full_like(A[0], 0.0001).requires_grad_(True)
    hb = torch.full_like(A[0], 1).requires_grad_(True)

    # torch.autograd.set_detect_anomaly(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= tut_utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    z -= tut_utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data                                       # step 4

    if return_only_l == 0:
        return R[0]

    # return relevancies of all layers
    return R


# compute global LRP transition matrix
def LRP_global_mat(model, point, gamma, l_lb = -1000, l_ub = 1000, delete_unactivated_subnetwork = 'mask', l_inp=0, l_out=-1, eps=1e-9, normalized=False):    
    
    if point.ndim == 3: point = point.flatten()
    assert point.ndim == 1, f"Dont pass batch. 'point' should have 1 dim but shape is {point.shape}"
    if gamma=='inf': gamma=1e8

    # forward pass: get activations & its shape per layer
    A, layers = layerwise_forward_pass(model, point[None])
    A_shapes = [a.shape[1:] for a in A]

    dimensionality = A_shapes[l_out]
    num_basis_vectors = np.prod(dimensionality)
    basis_vectors = torch.eye(num_basis_vectors).reshape(num_basis_vectors, *dimensionality)
    
    if not normalized:
        # instead of passing basis vectors through the network, pass the class activations. 
        # This overweights such rows of the conditional relevance matrix where the prior relevance is high. it's a "joint relevance" of the backward pass.
        assert l_out==-1, 'Never checked functionality of not normalized, with non output layers. Check if Flatten works correctly.'
        basis_vectors = torch.einsum('i...,i->i...', basis_vectors, A[l_out].detach().flatten())

    if delete_unactivated_subnetwork == True:
        # don't calculate the basis vector projections for those output elemnts/neurons that are not activated.
        mask = A[l_out].flatten() > 0
        basis_vectors = basis_vectors[mask]
        num_basis_vectors = sum(mask)
        assert num_basis_vectors, "No neuron in output layer is activated. This is not gonna work."

    # repeat activation per layer, as often as the number of basis vectors
    A_repeated = [torch.cat([a] * num_basis_vectors) for a in A]
    
    # LRP backward and reshape
    R_basis_vector_projections = compute_relevancies(mode=f'Gamma. l>{l_lb} l<{l_ub} gamma={gamma}', layers=layers, A=A_repeated, output_rels=basis_vectors, l_out=l_out, return_only_l=l_inp, eps=eps)
    LRP_backward = R_basis_vector_projections.reshape((num_basis_vectors, -1)).T

    if delete_unactivated_subnetwork == True:
        # delete such rows that correspond to unactivated **input** neurons
        mask = A[l_inp].flatten() > 0
        LRP_backward = LRP_backward[mask]

    if delete_unactivated_subnetwork == 'mask': # don't delete the rows, but set their entries to zero.
        l_out_activation = A_repeated[l_out][0].detach().numpy().flatten()[None]
        assert LRP_backward.shape[1] == l_out_activation.shape[1], f"{LRP_backward.shape[1]},  {l_out_activation.shape[1]}"
        LRP_backward *= (l_out_activation > 0)


    return coo_array(LRP_backward.detach().numpy())


def calc_mats_batch_functional(mat_funcs, gammas, points, tqdm_for='matrix', pickle_key=None, overwrite=False, partition=None):
    
    itg, itp, itm = [lambda x: x]*3
    if tqdm_for=='matrix': itm = tqdm
    if tqdm_for=='point':  itp = tqdm
    if tqdm_for=='gamma':  itg = tqdm
    
    partition = parse_partition(len(mat_funcs), len(points), partition)
    
    if pickle_key is not None:
        pickle_key = (pickle_key[0], pickle_key[1].replace('svals', 'LRP'))
        
        if not overwrite:
            mats = load_data(*pickle_key, warn=False)
            if mats is not False: 
                print("Found unpartitioned, full result. Returning.")
                return mats
            
            mats = load_data(*pickle_key, partition=partition, warn=False)
            if mats is not False: 
                print("Found partitioned result. Returning.")
                return mats
    
    if partition: 
        mat_funcs = [ mat_funcs[partition[0]] ]
        points =       [ points[partition[1]] ]

    mats = np.array([[[mat_func(point=point, gamma=gamma) for gamma in itg(gammas)] for point in itp(points)] for mat_func in itm(mat_funcs)])

    # save result
    if pickle_key is not None:
        print("Matrices vals under key:", pickle_key)
        save_data(*pickle_key, mats, partition=partition)

    return mats



    
### convenience functions for LRP matrix creation

## d3 model
def funcs_cascading__d3__m1_to_1(model, delete=True): # m1 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-3, l_ub=l_ub, delete_unactivated_subnetwork=delete) for l_ub in d3_after_conv_layer[:-1]]
def funcs_cascading__d3__m0_to_1(model, delete=True): # m0 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-1, l_ub=l_ub, delete_unactivated_subnetwork=delete) for l_ub in d3_after_conv_layer[:-1]]
def funcs_inv_cascading__d3__m1_to_1(model, delete=True): # m1 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-3, l_lb=l_ub-2, delete_unactivated_subnetwork=delete) for l_ub in d3_after_conv_layer[:-1][::-1]]
def funcs_individual__d3(model, delete=True):
    return [partial(LRP_global_mat, model=model, l_inp=l_out-1, l_out=l_out, delete_unactivated_subnetwork=delete) for l_out in d3_after_conv_layer[:-1]]

## s4 models
def funcs_individual__s4(model, delete=True):
    return [partial(LRP_global_mat, model=model, l_inp=l_out-1, l_out=l_out, delete_unactivated_subnetwork=delete) for l_out in s4_after_conv_layer]
def funcs_cascading__s4__m1_to_1(model, delete=True): # m1 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-3, l_ub=l_ub, delete_unactivated_subnetwork=delete) for l_ub in s4_after_conv_layer]
def funcs_inv_cascading__s4__m1_to_1(model, delete=True): # m1 to 1
    return [partial(LRP_global_mat, model=model, l_inp=1, l_out=-3, l_lb=l_ub-2, delete_unactivated_subnetwork=delete) for l_ub in s4_after_conv_layer[::-1]]



def transform_xai_mat(mat, act, cutoff=0, first_n=None, first_p=None, first_p_pos=None):
    act = act.flatten().detach()
    
    
    assert (not not cutoff) + (not not first_n) + (not not first_p) + (not not first_p_pos) == 1, f"Pass only one of cutoff, first_n, first_p, first_p_pos. {cutoff} {first_n} {first_p} {first_p_pos}"
    if first_n:
        if not (0 < first_n < len(act)):
            raise ValueError(f"n must be between 0 and the size of the array, {first_n} !! {len(act)}")
        cutoff = np.partition(act, -first_n)[-first_n]
        if cutoff==0: print(f'Warn: {first_n}-th largest is a 0 activation.')
    elif first_p:
        assert 0 < first_p <= 100
        cutoff = np.percentile(act, 100-first_p)
        if cutoff==0: print(f'Warn: Percentile {first_p} is a 0 activation.')
    elif first_p_pos:
        assert 0 < first_p_pos <= 100
        act_pos = act[act>0]
        cutoff = np.percentile(act_pos, 100-first_p_pos)
        # print(cutoff)
        if cutoff==0: print(f'Warn: Percentile {first_p} is a 0 activation.')
        
    # print(act.shape, cutoff)
        
    mask = act>=cutoff
    
    # if len(mask) != mat.shape[1]:
    #     mask = mask[act>0]
    assert len(mask) == mat.shape[1], 'mat does not match passed activations (or the subset of positive activations).'
    
    # print(mask.shape, mask.mean(dtype=float))
    
    if type(mat) in (coo_array, coo_matrix):
        m = mat.toarray()
        m = m[:, mask]
        return coo_array(m)
    else:
        return mat[:, mask]
    
def batch_transform_xai_mat(mats, A, **kwargs):
    new_mats = \
        [[[transform_xai_mat(per_g.copy(), a, **kwargs)
            for per_g in per_p]
                for per_p, a in zip(per_w, A)]
                    for per_w in mats]
    # new_mats = np.array(new_mats)
    return new_mats