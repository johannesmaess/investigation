import numpy
import copy
import torch
import torch.nn as nn


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

        # calculate positions where the gamma rule would cause  sign flip
        no_flip = (torch.sign(normal_layer_result) == torch.sign(normal_layer_result + positive_layer_result)).detach()

        print("fraction sign flips:", torch.logical_not(no_flip).sum() / numpy.prod(no_flip.shape))

        # apply LRP-gamma where no flips occur. Apply LRP-0 where flips occur
        return normal_layer_result + positive_layer_result * no_flip

