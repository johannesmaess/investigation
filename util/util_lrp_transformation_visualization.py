import numpy as np

def h0_W_h1_R1_C_R0(w11, w12, A1, mask_R=True): # w12 (neuron 2 -> neuro 1)
    """
    return the
    - x: The Input
    - W: Weights, 
    - H: activations, 
    - C: LRP coefficients, 
    - R: Relevancy scores
    for a one-layer, two-dimensional linear NN
    
    Only takes in
    - two degrees of freedom in the weight matrix W
    - one degree of freedom in the input x
    """
    
    assert(0 <= w11 <= 1)
    assert(0 <= w12 <= 1)
    assert(np.all((0 <= A1) * (A1  <= 1)))
    
    W = np.array([[  w11,   w12], 
                  [1-w11, 1-w12]])
    
    h0_list = []
    h1_list = []
    R1_list = []
    C_list = []
    R0_list = []
    
    for a1 in A1:
        h0 = np.array([a1, 1-a1])
        h0_list.append(h0)
        
        h1 = W @ h0
        h1_list.append(h1)

        C = W * h0[None, :]
        C /= C.sum(axis=1, keepdims=True)
        C_list.append(C)

        R1 = h1.copy()
        if (mask_R):
            R1 *= (h1 == h1.max(axis=0))
        R1_list.append(R1)
            
        R0 = C.T @ R1
        R0_list.append(R0)
    
    return np.array(h0_list), \
            W, \
            np.array(h1_list), \
            np.array(R1_list), \
            np.array(C_list), \
            np.array(R0_list)

def norm_arr(*args, norm=1):
    """
    Returns an array with the same elements, as the inputs.
    But normalized to 1. (The manhattan distance / L1 norm is set to 1.)
    """
    x = np.array(args).astype(float)
    x /= np.linalg.norm(x, ord=norm)
    return x