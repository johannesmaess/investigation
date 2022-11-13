import numpy as np
import copy
import torch
import torch.nn as nn
from enum import Enum

from util.util_gamma_rule import \
    global_conv_matrix_from_pytorch_layer, \
    forw_surrogate_matrix


class GammaPerNeuronLayer():
    """
    Helper layer for LRP forward hook implementation.
    Modifies forward pass with a variation of the LRP-gamma rule:
    For every output neuron, increase the magnitude of psotive so far, 
    that the output just barely did not change its sign (but often got quite small already).
    """
    def __init__(self, conv_layer, gamma_scale=.9, diffusion=0.):
        self.gamma_scale = gamma_scale
        self.diffusion = diffusion

        # save the normal conv layer
        self.normal_layer = copy.deepcopy(conv_layer)

        # create a layer with positive weights only
        self.positive_layer = copy.deepcopy(conv_layer)
        self.positive_layer.weight = nn.Parameter(self.positive_layer.weight.clamp(min=0))
        self.positive_layer.bias   = nn.Parameter(self.positive_layer.bias.clamp(min=0))


    def forward(self, x):
        positive_layer_result = self.positive_layer.forward(x)
        normal_layer_result =     self.normal_layer.forward(x)

        # calculate gammas, for which the output neurons activation would change its sign
        gamma = (- normal_layer_result / positive_layer_result).detach()
        # bound them
        gamma = gamma.clamp(min=0, max=10000)
        
        if self.diffusion > 0:
            gamma = self.diffusion * gamma.mean() + (1-self.diffusion) * gamma

        combined_result = normal_layer_result + self.gamma_scale * gamma * positive_layer_result

        return combined_result


class GammaWoSignFlipsLayer():
    """
    Helper layer for LRP forward hook implementation.
    Modifies forward pass with a variation of the LRP-gamma rule:
    Apply the gamma rule, to every neuron, except to those whose output would change its sign.
    """
    def __init__(self, conv_layer, gamma=0.):
        self.gamma = gamma

        # save the normal conv layer
        self.normal_layer = copy.deepcopy(conv_layer)

        # create a layer with positive weights only
        self.positive_layer = copy.deepcopy(conv_layer)
        self.positive_layer.weight = nn.Parameter(self.positive_layer.weight.clamp(min=0))
        self.positive_layer.bias   = nn.Parameter(self.positive_layer.bias.clamp(min=0))

    def forward(self, x):
        positive_layer_result = self.positive_layer.forward(x)
        normal_layer_result =     self.normal_layer.forward(x)

        # calculate if the gamma rule causes a sign flip
        no_flip = (torch.sign(normal_layer_result) == torch.sign(normal_layer_result + positive_layer_result)).detach()

        print("fraction sign flips:", torch.logical_not(no_flip).sum() / np.prod(no_flip.shape))

        # apply LRP-gamma where no flips occur. Apply LRP-0 where flips occur
        return normal_layer_result + positive_layer_result * no_flip


def coo_scipy_to_torch(mat):
    return torch.sparse_coo_tensor(np.stack((mat.row, mat.col)), mat.data, size=mat.shape)

class Mode(Enum):
    NORMAL = 1
    GAMMA = 2
    GAMMA_PER_NEURON = 3

class Conv2dAsMatrixLayer():
    def __init__(self, conv_layer, inp_shape, out_shape, bias=True):
        assert len(inp_shape) == 3, "Pass inp_shape as (in_channels, img_height, img_width). No batch dimension."
        assert len(out_shape) == 3, "Pass out_shape as (out_channels, img_height, img_width). No batch dimension."

        self.mode = Mode.NORMAL
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        
        self.bias_org = None
        if bias and conv_layer.bias is not None:
            self.bias_org = copy.deepcopy(conv_layer.bias.data)

        self.trans_org = global_conv_matrix_from_pytorch_layer(conv_layer, inp_shape, out_shape)

        self.reset()

    def reset(self):
        self.mode = Mode.NORMAL
        self.trans = coo_scipy_to_torch(self.trans_org)
        self.bias = copy.deepcopy(self.bias_org)

    def set_gamma(self, gamma):
        self.mode = Mode.GAMMA
        mat = forw_surrogate_matrix(self.trans_org, curr=None, gamma=gamma, checks=False, recover_activations=False)
        self.trans = coo_scipy_to_torch(mat.tocoo()) # mat is csr format, hence we need another conversion (dont know why csr)
        self.bias = self.bias_org + gamma * self.bias_org.clamp(min=0)

    def set_gamma_per_neuron(self, gamma_scale):
        self.mode = Mode.GAMMA_PER_NEURON
        self.gamma_scale = gamma_scale
        self.reset()

        self.trans_pos = coo_scipy_to_torch((self.trans_org * (self.trans_org > 0)).tocoo())
        self.bias_pos = self.bias.clamp(min=0) if self.bias is not None else None

    def forward_one(self, x):
        if self.mode in (Mode.NORMAL, Mode.GAMMA):
            return (self.trans @ x.view(-1)).reshape(self.out_shape) + (self.bias[:, None, None] if self.bias is not None else 0)

        if self.mode in (Mode.GAMMA_PER_NEURON):
            res     = (self.trans     @ x.view(-1)).reshape(self.out_shape) + (    self.bias[:, None, None] if     self.bias is not None else 0)
            res_pos = (self.trans_pos @ x.view(-1)).reshape(self.out_shape) + (self.bias_pos[:, None, None] if self.bias_pos is not None else 0)


            # calculate gammas, for which the output neurons activation would change its sign
            gamma = (- res / res_pos).detach()
            # bound them
            gamma = gamma.clamp(min=0, max=10000)

            return res + self.gamma_scale * gamma * res_pos


    def forward(self, x):
        if len(x.shape) == 3: # means no batch dimension was passed
            assert self.inp_shape == x.shape
            return self.forward_one(x), f"{self.inp_shape} != {x.shape}"

        assert len(x.shape) == 4, "Invalid input."
        assert self.inp_shape == x.shape[1:], f"{self.inp_shape} != {x.shape[1:]}"
        
        return torch.stack([self.forward_one(sample) for sample in x])

        
