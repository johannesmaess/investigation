import numpy as np
from quantus.helpers.utils import (
    get_baseline_value,
    blur_at_indices,
    expand_indices,
    get_leftover_shape,
    offset_coordinates,
    calculate_auc
)
from typing import Any, Callable, Sequence, Tuple, Union, Optional
from copy import copy

import matplotlib.pyplot as plt

def batch_auc(batch):
    return [calculate_auc(np.array(curve)) for curve in batch]
def batch_mean_auc(batch):
    return np.mean(batch_auc(batch))

from IPython.display import clear_output

def max_diff_replacement_by_indices(
    arr: np.array,
    indices: Tuple[slice, ...],  # Alt. Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    perturb_baseline: Union[float, int, str, np.array],
    **kwargs,
) -> np.array:
    """Modification / Extension of baseline_replacement_by_indices"""
    indices = expand_indices(arr, indices, indexed_axes)

    arr_perturbed = copy(arr)

    org_val = arr[indices]
    flipped_val = 0 if org_val > 0.5 else 1

    # Perturb the array.
    arr_perturbed[indices] = flipped_val

    if max_diff_replacement_by_indices.counter % 40 == -1:
        clear_output(wait=True)
        print(max_diff_replacement_by_indices.counter)
        plt.imshow(arr_perturbed[0])
        plt.show()
        assert not input(), "Break by user."

    max_diff_replacement_by_indices.counter += 1

    return arr_perturbed
max_diff_replacement_by_indices.counter = 0
