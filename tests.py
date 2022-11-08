def back_matrix_recovery_test():
    """
    Checks if the back matrix recovers original activations,
    when gamma=0.
    """
    gamma=0

    for mat in np.random.normal(0,1, (10, 3, 2)):
        mat /= mat.sum(axis=0, keepdims=True)
        
        for a in np.random.normal(0,1,(10,2)):
            back = back_matrix(mat, a, gamma)
            assert np.allclose(back @ mat @ a, a), f"{back @ mat @ a}, {a}"

