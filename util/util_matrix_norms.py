import numpy as np
from util.naming import *
from util.util_data_summary import prep_data
from util.util_pickle import load_data


def mat_norm_batch(per_weight, p):
    """ 
    Expects Matrices to be nested in three dimension (weight, point, gamma).
    Then computes the operator norm induced by the L-p vector norm on each matrix and returns them in the same shape as the matrices.
    """
    if p==1:
        norm = lambda mat: np.abs(mat).sum(axis=0).max()
    elif p==np.inf:
        norm = lambda mat: np.abs(mat).sum(axis=1).max()
    elif p=="frobenius":
        norm = lambda mat: (mat.data**2).sum()**.5

    return np.array([[[norm(mat) for mat in per_gamma] for per_gamma in per_point] for per_point in per_weight])

def shape_batch(per_weight):
    """
    Extracts shape (m, n) for every matrix in batch. 
    Returns tuple (M, N) wher M and N have 3 dimensional shape like the passed matrices.
    """
    return np.array([[[list(mat.shape) for mat in per_gamma] for per_gamma in per_point] for per_point in per_weight]).transpose((3,0,1,2))

def nonzero_rows_cols_batch(per_weight):
    """
    Extracts number of non-zero rows & columns (m, n) for every matrix in batch. 
    Returns tuple (M, N) wher M and N have 3 dimensional shape like the passed matrices.
    """
    def uni(arr): return len(np.unique(arr))
    return np.array([[[[uni(mat.row), uni(mat.col)] for mat in per_gamma] for per_gamma in per_point] for per_point in per_weight]).transpose((3,0,1,2))

def calc_norm_dict(matrices=None, svals=None, gammas=None, pickle_key=(), cs=None, filter_size=None):
    """
    Returns a collection of useful norms, and bounds on the L2 norm in a dictionary.
    """
    if svals == None: svals, gammas = prep_data(svals, gammas)
    if matrices == None:
        mkey, dkey = pickle_key
        ind = dkey.find('__')
        # key for loading LRP matrices: "LRP__..."
        dkey_lrp = 'LRP' + dkey[ind:]
        matrices = load_data(mkey, dkey_lrp)

    assert svals.shape[0] == matrices.shape[0]  and svals.shape[2] == matrices.shape[2] # same number of weights and gammas
    n_points = min(matrices.shape[1], svals.shape[1])

    if cs is not None: 
        assert cs.shape[0] == matrices.shape[0] # same number of weights
        
        n_points = min(n_points, cs.shape[1])
        cs = cs[:, :n_points]

    svals = svals[:, :n_points]
    matrices = matrices[:, :n_points]
    
    l1 = mat_norm_batch(matrices, p=1)
    linf = mat_norm_batch(matrices, p=np.inf)
    frobenius = mat_norm_batch(matrices, p='frobenius')

    # predict l1 norm by just calculating the coefficient c once
    if cs is not None: 
        gammas = match_gammas(svals)
        cs = cs[:, :, None]
        gammas = gammas[None, None, :]
        l1_analytically = 1 + 2*cs / (1 - cs + gammas)
        a['l1_analytically'] = l1_analytically

    frobenius_b = (svals**2).sum(axis=3)**.5
    rel_err = np.abs((frobenius - frobenius_b) / np.maximum(frobenius, frobenius_b)).max()
    assert rel_err < 0.01, f"Multiplicative difference in Frobenius norm calculation methods too high. {rel_err}"
    del frobenius_b

    l2 = svals[:, :, :, 0]

    Ms, Ns = shape_batch(matrices)
    Ms, Ns = nonzero_rows_cols_batch(matrices) # note: I think it is okay/equivalent to only use non-zero rows in norm bound.

    l1_lower = l1 / np.sqrt(Ms)
    l1_upper = l1 * np.sqrt(Ns)

    linf_lower = linf / np.sqrt(Ns)
    linf_upper = linf * np.sqrt(Ms)

    sqrt_L1_Linf = np.sqrt(l1*linf)
    
    norm_dict = {k:v for k,v in zip(["L1_lower", "L1_upper", "Linf_lower", "Linf_upper", "sqrt_L1_Linf", "L2", "L1", "L1_analytically", "Linf", "frobenius"], 
                                (l1_lower, l1_upper, linf_lower, linf_upper, sqrt_L1_Linf, l2, l1, l1_analytically, linf, frobenius)) }


    if filter_size is not None:
        if filter_size == 'lrp':
            filter_size = np.vectorize(lambda mat: (mat.toarray() != 0).sum(axis=0).max())(matrices)
        else:
            assert len(matrices) == len(filter_size), "Pass the numer of filter entries for every matrix."
            filter_size = np.array(filter_size)
        
        norm_dict["Linf by L1"] = l1 * filter_size
        norm_dict["sqrt_L1_Linf by L1"] = np.sqrt(1+filter_size) * l1

    return norm_dict