import numpy as np
import torch
from scipy.sparse import coo_array
from tqdm.notebook import tqdm
import copy

import util.util_tutorial as tut_utils

import matplotlib.pyplot as plt

def layerwise_forward_pass(model, data=None, checks=False):
    layers = tut_utils.toconv(list(model.seq))
    if data is None: return layers
    L = len(layers)

    # A = [data]+[None]*L
    A = [data.reshape(-1,1,28,28)]+[None]*L

    for l in range(L):
        if isinstance(layers[l], torch.nn.Flatten):
            batch_size, *image_size = A[l].shape
            A[l+1] = A[l].reshape((batch_size, np.prod(image_size), 1, 1))
        else:
            A[l+1] = layers[l].forward(A[l])

    if checks:
        res = model.forward(A[0]).flatten().detach()
        res_indirect =      A[-1].flatten().detach()
        assert torch.allclose(res, res_indirect, atol=1e-5), f"Too high diff: { np.abs(res - res_indirect).max() }"

    return A, layers

def forward_and_explain(model, data, mode):
    A, layers = layerwise_forward_pass(model, data)
    return compute_relevancies(mode, layers, A, output_rels='predicted class', return_only_l=0)

def compute_relevancies(mode, layers, A, output_rels='correct class', target=None, l_out=-1, return_only_l=None):
    """
    Applies a LRP backpropagation through all or a subset of layers of the network.
    
    Pass the relevancies in the deepest layer to iterate as "output_rels".
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
        print(l, layers[l])

        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):

            # print(l, str(layers[l]).split('(')[0])

            # default: LRP-0.
            rho = lambda p: p;
            helper_layer = tut_utils.newlayer(layers[l], rho)

            if isinstance(layers[l],torch.nn.Conv2d):
                if 'Gamma' in mode:
                    layer_smaller = float(mode.split("l<"    )[1].split(" ")[0])
                    if l < layer_smaller:
                        curr_gamma = float(gam) if 'inf' != (gam := mode.split("gamma=")[1].split(" ")[0]) else 1e8
                        if 'Gamma.' in mode:
                            rho = lambda p: p + curr_gamma*p.clamp(min=0);
                            helper_layer = tut_utils.newlayer(layers[l], rho)
                        elif 'Gamma mat.' in mode:
                            helper_layer = copy.deepcopy(layers_conv_as_mat[l]) # todo: in the notebook I used a precomputation "layers_conv_as_mat"
                            helper_layer.set_gamma(curr_gamma)

            incr = lambda z: z+1e-9
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
def LRP_global_mat(model, point, gamma, l_leq = 1000, delete_unactivated_subnetwork = 'mask', l_inp=0, l_out=-1):    
    assert len(point.shape) == 1, f"Dont pass batch. 'point' should have 1 dim but shape is {point.shape}"
    if gamma=='inf': gamma=1e8

    # forward pass: get activations & its shape per layer
    A, layers = layerwise_forward_pass(model, point[None])
    A_shapes = [a.shape[1:] for a in A]

    dimensionality = A_shapes[l_out]
    dimensionality_flat = np.prod(dimensionality)
    basis_vectors = torch.eye(dimensionality_flat).reshape(dimensionality_flat, *dimensionality)

    if delete_unactivated_subnetwork == True:
        # don't calculate the basis vector projections for those output elemnts/neurons that are not activated.
        mask = A[l_out].flatten() > 0
        basis_vectors = basis_vectors[mask]
        dimensionality_flat = sum(mask)

    # repeat activation per layer, as often as the number of basis vectors
    A_repeated = [torch.cat([a] * dimensionality_flat) for a in A]
    
    # LRP backward and reshape
    R_basis_vector_projections = compute_relevancies(mode=f'Gamma. l<{l_leq} gamma={gamma}', layers=layers, A=A_repeated, output_rels=basis_vectors, l_out=l_out, return_only_l=l_inp)
    LRP_backward = R_basis_vector_projections.reshape((dimensionality_flat, -1)).T

    if delete_unactivated_subnetwork == True:
        # delete such rows that correspond to unactivated **input** neurons
        mask = A[l_inp].flatten() > 0
        LRP_backward = LRP_backward[mask]

    if delete_unactivated_subnetwork == 'mask': # don't delete the rows, but set their entries to zero.
        l_out_activation = A_repeated[l_out][0].detach().numpy().flatten()[None]
        assert LRP_backward.shape[1] == l_out_activation.shape[1], f"{LRP_backward.shape[1]},  {l_out_activation.shape[1]}"
        LRP_backward *= (l_out_activation > 0)


    return coo_array(LRP_backward.detach().numpy())

def calc_mats_batch_functional(mat_funcs, gammas, points, tqdm_for='matrix'):
    itg, itp, itm = [lambda x: x]*3
    if tqdm_for=='matrix': itm = tqdm
    if tqdm_for=='point':  itp = tqdm
    if tqdm_for=='gamma':  itg = tqdm

    return np.array([[[mat_func(point=point, gamma=gamma) for gamma in itg(gammas)] for point in itp(points)] for mat_func in itm(mat_funcs)])