import torch
from torch import nn, functional as F
import matplotlib.pyplot as plt
import numpy as np

# create 2d grid of points

def forward(hidden_sizes = [3], readout = lambda act: act):
    activations = []
    activations_sizes = [3] + hidden_sizes + [3]

    mini,maxi,step = -10,10,1.
    step_z = 5.

    X = torch.tensor(np.mgrid[mini:maxi:step, mini:maxi:step, mini:maxi:step_z].reshape(3, -1).T) * 1.
    X.requires_grad = True
    activations.append(X)

    for inp, out in zip(activations_sizes[:-1], activations_sizes[1:]):
        H = nn.ReLU()(nn.Linear(inp, out, bias=True, dtype=float)(activations[-1]))
        H.retain_grad()
        activations.append(H)

    activations[-1].sum().backward()

    return [readout(act) for act in activations]

def plot(activations, show=[0,-1], color_fn=None):
    fig = plt.figure(figsize=(5*len(activations), 5))

    show = [activations[i] for i in show] if show is not None else activations
    show = [act.detach().numpy() for act in show]
    if color_fn is None:
        def color_fn(activations):
            Y = activations[-1]
            eps = 1
            colors = Y.copy()
            colors += abs(colors.min()) + eps
            colors /= colors.max()
            colors = 1 - colors
            return colors
    colors = color_fn(activations)

    for i, act in enumerate(show):
        ax = fig.add_subplot(1, len(activations), i+1, projection='3d')
        print(colors.min(), colors.max())
        ax.scatter(act[:, 0], act[:, 1], act[:, 2], c=colors)

if __name__ == '__main__':
    torch.manual_seed(8)
    activations = forward(hidden_sizes=[4],)# readout=lambda act: act.grad)
    def color_fn(activations):
        Y = activations[0].grad
        eps = 0
        # colors = Y * 1.
        # colors += abs(colors.min()) + eps
        colors = Y.abs()
        colors /= colors.max()
        colors = 1 - colors
        return colors
    plot(activations, show=None, color_fn=color_fn)
    plt.show()