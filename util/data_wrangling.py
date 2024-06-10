import numpy as np
from scipy.sparse import coo_matrix, coo_array

def mask_by_activations(mat=None, outputs=None, cutoff=None, first_n=None, first_p=None, prefilter_positive_activations=False, return_only_masks_and_outputs=False):
    """
    Expects a two-dimensional matrix mat (input neurons x output neurons), and a one-dimensional vector outputs (the activations of the output neurons)
    
    prefilter_positive_activations is useful as it standardizes first_n/first_p to only take the first p/n of the *activated* neurons.
    
    """
    
    assert (mat is not None and outputs is not None) or (outputs is not None and return_only_masks_and_outputs==True)
    
    assert outputs.ndim == 1, outputs.shape
    if mat:
        assert mat.ndim == 2, mat.shape
        assert len(outputs) == mat.shape[0], f"{mat.shape} does not match {outputs.shape}"
    
    
    n_args = ((cutoff is not None) + (first_n is not None) + (first_p is not None))
    if n_args == 0 and prefilter_positive_activations: 
        cutoff = 0
    else:
        assert n_args == 1, f'pass exactly one of cutoff, first_n and first_p: {cutoff}, {first_n} and {first_p}'
    
    if prefilter_positive_activations:
        pre_mask = outputs>0
        if mat: mat = mat[pre_mask]
        outputs = outputs[pre_mask]
        
        if return_only_masks_and_outputs:
            global_mask = pre_mask.clone()
        
    if first_n:
        if first_n >= 0 or first_n < outputs.size:
            raise ValueError("n must be between 0 and the size of the array")
        cutoff = np.partition(outputs, -first_n)[-first_n]
        if cutoff==0: print(f'Warn: {first_n}-th largest is a 0 activation.')
    
    elif first_p:
        assert 0 < first_p <= 100
        cutoff = np.percentile(outputs, 100-first_p)
        if cutoff==0: print(f'Warn: Percentile {first_p} is a 0 activation.')
    
    # the third case is that cutoff was already set.     
    mask = outputs>=cutoff
        
    
    if return_only_masks_and_outputs:
        global_mask[pre_mask] &= mask
        return global_mask, outputs[mask]
    
    if len(mask) != mat.shape[0]:
        print(f'reducing mask shape ({mask.shape}) to subset of rows of positive activations (', end='')
        mask = mask[outputs>0]
        print(f'{mask.shape}) to try to match the passed matrix.')
    assert len(mask) == mat.shape[0], f'mat does not match passed activations (or the subset of positive activations). mat.shape: {mat.shape}, len(activations): {len(mask)}'
    
    if type(mat) in (coo_array, coo_matrix):
        m = mat.toarray()
        m = m[mask]
        return coo_array(m)
    else:
        return mat[mask]

def mask_unactivated__normalize__reshape(all_mats, output, normalize=True, all_logits=False, **kwargs_mask_by_activations):
    """
    all_mats: n_points, n_parameters, n_outputs, *inp_shape (inp_shape can already be flattened or not)
    output is 2d: (n_points, n_outputs)
    """
    assert output.ndim == 2
    
    if not kwargs_mask_by_activations:
        kwargs_mask_by_activations = dict(cutoff=0)
    
    n_subsections = len(all_mats)
    result = [None] * n_subsections
    
    for i, mats in enumerate(all_mats): # iterate over subsections
        
        n_points, n_parameters, n_outputs, *inp_shape = len(mats), len(mats[0]), *mats[0][0].shape
        assert (n_points, n_outputs) == output.shape, f'(n_points, n_outputs): {(n_points, n_outputs)}, output.shape: {output.shape} (mats.shape: {n_points, n_parameters, n_outputs, *inp_shape})'
        
        # flatten and normalize input
        mats = mats.reshape((n_points, n_parameters, n_outputs, -1))
        if normalize: 
            mats /= mats.sum(axis=3, keepdims=True)
        
        if not all_logits:
            # temporarily bring (points, classes) to the first two dim, such that we can mask with output of the same shape.
            mats = mats.transpose((0,2,1,3))
            mats[output <= 0] = 0
            mats = mats.transpose((0,2,1,3))
            
        mats = \
            [[mask_by_activations(per_g, o, **kwargs_mask_by_activations)
                for per_g in per_p]
                    for per_p, o in zip(mats, output)]
        mats = np.array(mats)
        
        print(f'subsection {i}:', (n_points, n_parameters, n_outputs, *inp_shape), ' -> ', mats.shape, mats[0][0].shape)
        result[i] = mats
    
    return result