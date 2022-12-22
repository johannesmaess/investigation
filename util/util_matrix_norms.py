import numpy as np


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

def calc_norm_dict(matrices, svals):
    """
    Returns a collection of useful norms, and bounds on the L2 norm in a dictionary.
    """
    assert matrices.shape[:3] == svals.shape[:3]
    
    l1 = mat_norm_batch(matrices, p=1)
    linf = mat_norm_batch(matrices, p=np.inf)
    frobenius = mat_norm_batch(matrices, p='frobenius')

    frobenius_b = (svals**2).sum(axis=3)**.5
    rel_err = np.abs((frobenius - frobenius_b) / np.maximum(frobenius, frobenius_b)).max()
    assert rel_err < 0.001, f"Multiplicative difference in Frobenius norm calculation methods too high. {rel_err}"
    del frobenius_b

    l2 = svals[:, :, :, 0]

    Ms, Ns = shape_batch(matrices)
    Ms, Ns = nonzero_rows_cols_batch(matrices) # note: I think it is okay/equivalent to only use non-zero rows in norm bound.

    l1_lower = l1 / np.sqrt(Ms)
    l1_upper = l1 * np.sqrt(Ns)

    linf_lower = linf / np.sqrt(Ns)
    linf_upper = linf * np.sqrt(Ms)

    sqrt_L1_Linf = np.sqrt(l1*linf)
    
    return {k:v for k,v in zip(["L1_lower", "L1_upper", "Linf_lower", "Linf_upper", "sqrt_L1_Linf", "L2", "L1", "Linf", "frobenius"], 
                                (l1_lower, l1_upper, linf_lower, linf_upper, sqrt_L1_Linf, l2, l1, linf, frobenius)) }
