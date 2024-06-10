import numpy as np
from quantus.helpers.utils import (
    get_baseline_value,
    blur_at_indices,
    expand_indices,
    get_leftover_shape,
    offset_coordinates,
    calculate_auc
)
from typing import Any, Callable, Sequence, Tuple, Union, Optional, List
import copy

import cv2 as cv

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
    visualize_every_n_steps: int = -1,
    **kwargs,
) -> np.array:
    """Modification / Extension of baseline_replacement_by_indices"""
    indices = expand_indices(arr, indices, indexed_axes)

    arr_perturbed = copy.copy(arr)

    org_val = arr[indices]
    flipped_val = np.ones_like(org_val)
    flipped_val[org_val > 0.5] = 0
    print(org_val, flipped_val)

    # Perturb the array.
    arr_perturbed[indices] = flipped_val

    if visualize_every_n_steps > 0 \
       and max_diff_replacement_by_indices.counter % visualize_every_n_steps == 0:
        print(max_diff_replacement_by_indices.counter)
        print(arr.shape)
        plt.imshow(arr_perturbed[0])
        plt.show()
        assert not input(), "Break by user."
        clear_output(wait=True)

    max_diff_replacement_by_indices.counter += 1

    return arr_perturbed
max_diff_replacement_by_indices.counter = 0

def interpolation_replacement_by_indices(
    arr: np.array,
    indices: Tuple[slice, ...],  # Alt. Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    perturb_baseline: Union[float, int, str, np.array],
    inpaint_radius: int = 3,
    visualize_every_n_steps: int = -1,
    **kwargs,
) -> np.array:
    """Modification / Extension of baseline_replacement_by_indices"""
    indices = expand_indices(arr, indices, indexed_axes)

    assert arr.ndim == 3 and len(arr) == 1, f'one-chanel pictures supported only. {arr.shape}'
    
    # create mask to determine where to inpaint
    mask = np.zeros_like(arr, dtype=np.uint8)
    mask[indices] = 1
    mask = mask[0]

    # we have to convert to unit8 for opencv
    arr_uint8 = (arr * 255).astype(np.uint8)  # Convert to uint8 if necessary
    arr_uint8 = arr_uint8[0]
    
    arr_perturbed_uint8 = cv.inpaint(arr_uint8, mask, inpaint_radius, cv.INPAINT_TELEA)

    # extract only the inpainted pixel back to our arr, to guarantee no changes through data conversion.
    arr_perturbed = copy.copy(arr)
    arr_perturbed[indices] = arr_perturbed_uint8[None][indices].astype(np.float32) / 255

    if visualize_every_n_steps > 0 \
       and interpolation_replacement_by_indices.counter % visualize_every_n_steps == 0:
        clear_output(wait=True)
        print(max_diff_replacement_by_indices.counter)
        print(arr.shape)
        plt.imshow(arr_perturbed[0])
        plt.show()
        assert not input(), "Break by user."

    interpolation_replacement_by_indices.counter += 1

    return arr_perturbed
interpolation_replacement_by_indices.counter = 0

from quantus.metrics.faithfulness import PixelFlipping
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers import warn

class PixelFlippingExpandingPerturbationSet(PixelFlipping):
    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
    ) -> List[float]:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s: np.ndarray
            The segmentation to be evaluated on an instance-basis.

        Returns
        -------
           : list
            The evaluation results.
        """

        # Reshape attributions.
        a = a.flatten()

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a)

        # Prepare lists.
        n_perturbations = len(range(0, len(a_indices), self.features_in_step))
        preds = [None for _ in range(n_perturbations)]

        for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

            # Perturb input by indices of attributions.
            a_ix = a_indices[
                # (self.features_in_step * i_ix) 
                : (self.features_in_step * (i_ix + 1))
            ]
            x_perturbed = self.perturb_func(
                arr=x.copy(),
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            # warn.warn_perturbation_caused_no_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])
            preds[i_ix] = y_pred_perturb

        return preds