import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


def plot_vector_field(A=None, trans=None, lims = np.array([-1.1, 1.1]), n_points = 9, scale = 1.5, small_arrows=True, circular_points=False, figsize=(10,10), 
                        ax=None, point_in_green=None):
    assert(A is None or trans is None)
    
    step = (lims[1]-lims[0]) / (n_points - 1)

    if not ax: 
        plt.figure(figsize=figsize)
        plt.xlim(*(lims*scale))
        plt.ylim(*(lims*scale))
        ax = plt

    ax.axvline(0)
    ax.axhline(0)
    # plt.plot((lims*scale), (lims*scale))

    vecs = np.mgrid[lims[0]:lims[1]+1e-5:step, lims[0]:lims[1]+1e-5:step].T.reshape(-1, 2)
    
    if circular_points:
        # mask points out that are outisde of radius from origin.
        norms = LA.norm(vecs, axis=1)
        vecs = vecs[norms <= lims[1]]
    
    projs = (A @ vecs.T).T if (A is not None) else trans(vecs)

    if point_in_green:
        target = A@point_in_green if (A is not None) else trans(point_in_green)
        ax.arrow(*point_in_green, *(target - point_in_green), color='green')
        
    diffs = projs - vecs

    ax.scatter(*vecs.T, color='b')
    ax.scatter(*projs.T, color='r')

    if small_arrows:
        max_len = LA.norm(diffs, axis=1).max()
        diffs_smaller = diffs / max_len * step # reduce length, such that each arrow can not be longer than 'step'

        for vec, diff in zip(vecs, diffs_smaller):
            ax.arrow(*vec, *diff)
    else:  
        for vec, diff in zip(vecs, diffs):
            ax.arrow(*vec, *diff)

def plot_vector_field_batch(matrices_list, figscale=5, **kwargs):
    n = len(matrices_list)
    fig, axs = plt.subplots(1, n, figsize=(n*figscale*1.1, figscale), sharex=True, sharey=True)
    for A, ax in zip(matrices_list, axs):
        ax.set_aspect(1)
        plot_vector_field(A=A, ax=ax, **kwargs)


### print util ###

def print_weights_list(weights_list):
    stringified = np.array([str(w.round(2)).split('\n') for w in weights_list]).T
    for row in stringified:
        print(' '*7, end='')
        for item in row:
            print(f"{item:29s}", end="")
        print()


def print_weights_list_info(weights_list):
    evals_list, evecs_list = [LA.eig(W)[0] for W in weights_list], [LA.eig(W)[1] for W in weights_list]

    print('Weights'); print_weights_list(weights_list)
    print('Evals');   print_weights_list(evals_list)
    print('Evecs');   print_weights_list(evecs_list)



### quickly create matrices ###


def rotation1(deg1, deg2=None):
    """
    If only deg1 is passed, it rotates every data point by deg1.
    If deg1 and deg2 are passed, it rotates the two standard basis vector accordingly.
    """
    if deg2==None: deg2 = deg1 + 90
        
    deg1 = deg1/180 * np.pi
    deg2 = deg2/180 * np.pi
        
    return np.array([
        [np.cos(deg1),  np.cos(deg2)],
        [np.sin(deg1),  np.sin(deg2)]
    ])


def rotation2(discriminant, a_minus_d, a, b_c_distribution, b_is_positive=False):
    """
    Lets you define a "general 2D rotation" matrix (A 2D matrix with complex eigenvectors)
    by a set of meaningful free parameters.
    """

    assert discriminant <= 0, "Positive discriminants leads to non-rotational matrices."
    
    # determined variables
    btimesc = 1/4*(discriminant - a_minus_d**2) # because: (a-d)^2 + 4bc = discriminant
    b = np.abs(btimesc)**(.5 + b_c_distribution)
    c = np.abs(btimesc)**(.5 - b_c_distribution)
    b,c = [(b,-c), (-b,c)][b_is_positive]
    d = a - a_minus_d

    assert np.allclose(btimesc, b*c), f"{btimesc} != {b*c}"
    disc = (a-d)**2 + 4*b*c
    assert np.allclose(disc, discriminant), f"{disc} != {discriminant}"
    disc = (a+d)**2 - 4*(a*d-b*c)
    assert np.allclose(disc, discriminant), f"{disc} != {discriminant}"

    W = np.array([[a,b], [c,d]])
    assert not np.any(np.iscomplex(W)), "Matrix should not have complex entries, but:" + str(W.iscomplex())
    return W.astype(np.float64)