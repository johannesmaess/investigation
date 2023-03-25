import numpy as np
import torch
from scipy.sparse import coo_array
from tqdm import tqdm
import copy

import util.util_tutorial as tut_utils
from util.util_pickle import *

import matplotlib.pyplot as plt

def layerwise_forward_pass(model, data=None, checks=False, pos_neg=False, get_c=False): # just for MNIST:
    return layerwise_forward_pass_general(list(model.seq), data, checks, pos_neg, get_c, inp_shape=(-1,1,28,28))

def layerwise_forward_pass_general(layers, data=None, checks=False, pos_neg=False, get_c=False, inp_shape=None):
    assert not (pos_neg and get_c)

    layers = tut_utils.toconv(layers)
    if data is None: return layers
    L = len(layers)

    data = data.reshape(inp_shape) if inp_shape else data[:, :, None, None]

    A = [data]+[None]*L
    A_pos = [None]*(L+1)
    A_neg = [None]*(L+1)
    c_list = []

    for l in range(L):
        lay = layers[l]
        if isinstance(lay, torch.nn.Flatten):
            batch_size, *image_size = A[l].shape
            A[l+1] = A[l].reshape((batch_size, np.prod(image_size), 1, 1))
        else:
            A[l+1] = lay.forward(A[l])
            
            # calculate positive and negative contributions to neurons seperately
            if (pos_neg or get_c) and isinstance(lay, torch.nn.Conv2d):
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

    for l in range(1, l_out)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        # print(l, str(layers[l]).split('(')[0])

        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
            # default: LRP-0.
            rho = lambda p: p;
            helper_layer = tut_utils.newlayer(layers[l], rho)

            if isinstance(layers[l],torch.nn.Conv2d):
                if 'Gamma' in mode:
                    l_ub = float(mode.split("l<")[1].split(" ")[0]) if "l<" in mode else 1000
                    l_lb = float(mode.split("l>")[1].split(" ")[0]) if "l>" in mode else -1000
                    if l_lb < l < l_ub:
                        curr_gamma = float(gam) if 'inf' != (gam := mode.split("gamma=")[1].split(" ")[0]) else 1e8
                        # print(l, curr_gamma)
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
def LRP_global_mat(model, point, gamma, l_lb = -1000, l_ub = 1000, delete_unactivated_subnetwork = 'mask', l_inp=0, l_out=-1, eps=1e-9):    
    if point.ndim == 3: point = point.flatten()
    assert point.ndim == 1, f"Dont pass batch. 'point' should have 1 dim but shape is {point.shape}"
    if gamma=='inf': gamma=1e8

    # forward pass: get activations & its shape per layer
    A, layers = layerwise_forward_pass(model, point[None])
    A_shapes = [a.shape[1:] for a in A]

    dimensionality = A_shapes[l_out]
    num_basis_vectors = np.prod(dimensionality)
    basis_vectors = torch.eye(num_basis_vectors).reshape(num_basis_vectors, *dimensionality)

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

def calc_mats_batch_functional(mat_funcs, gammas, points, tqdm_for='matrix', pickle_key=None, overwrite=False):
    # try to load result
    if pickle_key is not None:
        pickle_key = (pickle_key[0], pickle_key[1].replace('svals', 'LRP'))
        if not overwrite:
            mats = load_data(*pickle_key)
            if mats is not False: return mats

    itg, itp, itm = [lambda x: x]*3
    if tqdm_for=='matrix': itm = tqdm
    if tqdm_for=='point':  itp = tqdm
    if tqdm_for=='gamma':  itg = tqdm

    mats = np.array([[[mat_func(point=point, gamma=gamma) for gamma in itg(gammas)] for point in itp(points)] for mat_func in itm(mat_funcs)])

    # save result
    if pickle_key is not None:
        save_data(*pickle_key, mats)

    return mats