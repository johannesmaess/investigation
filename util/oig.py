import torch
import numpy as np

### backuo of first implementation
class OIntermediateIntegratedGradients():
    """
    Supports two new features:
    - Capture and integrate the gradients at intermediate layers.
    - Compute Path integrals of different lengths at the same time, starting from input, extending towards baseline. For this we use a shared Rieman integral approximation.
    """
    def __init__(self, submodels) -> None:
        """
        Expects a list of submodels, which contain subsets of the layers of the model.
        Before every submodel, the inputs will be stored, and gradients will be captured and integrated.
        """
        self.submodels = submodels

    def attribute(self, inp, baseline_0=None, only_output=None, n_steps=50, baselines_at_steps=None, measure_at_layers=None, multiply_by_inputs=True, like_lrp=False):
        """ . 
        input: tensor of shape (batch_size, *input_size)
        baseline: tensor of shape (batch_size, *input_size)
        only_output: 
        n_steps: number of steps to integrate over
        baselines_at_steps: list of integers, indicating the steps at which to integrate the gradients.
        """
        if baseline_0 is None:
            baseline_0 = torch.zeros(inp.shape)
        else:
            assert isinstance(baseline_0, torch.Tensor), "Baseline must be a tensor."
            assert baseline_0.shape == inp.shape, "Baseline must have the same shape as input."
        if baselines_at_steps is None:
            baselines_at_steps = 1 + np.arange(n_steps)
        elif isinstance(baselines_at_steps, int):
            baselines_at_steps = np.array([baselines_at_steps])
        elif isinstance(baselines_at_steps, list):
            baselines_at_steps = np.array(baselines_at_steps)
        else:
            assert isinstance(baselines_at_steps, np.ndarray), "baselines_at_steps must be None, int, list or np.ndarray"
                    
        # to achieve different integral lengths / "#steps in integral" we take the (equally weighted) mean over different steps in the 
        integration_matrix = torch.zeros((len(baselines_at_steps), n_steps+1))
        for i, at in enumerate(baselines_at_steps):
            integration_matrix[i, (-at):] = 1/at

        # Store inputs
        weights = torch.arange(n_steps+1) / n_steps # the linear combination from the first weight (0) is the baseline. We never use it's gradient, but need it when multiplying with (input - baseline).
        while weights.ndim <= inp.ndim:
            weights = weights[:,None]
        curr_submodel_path_inputs = baseline_0[None] + weights * (inp[None] - baseline_0[None])

        

        submodels_path_inputs = []
        # Compute forward pass, store intermediate results
        for i, submodel in enumerate(self.submodels):
            submodel.zero_grad()
            curr_submodel_path_inputs.requires_grad_(True)
            curr_submodel_path_inputs.retain_grad()
            
            if (measure_at_layers is None) or (i in measure_at_layers):
                submodels_path_inputs.append(curr_submodel_path_inputs)
            
            curr_submodel_path_inputs = submodel(curr_submodel_path_inputs)

        outputs_on_path = curr_submodel_path_inputs
        assert outputs_on_path.ndim == 2, "Only supports 1D outputs for now." #  (batched to 2D)

        if only_output is None:
            only_output = outputs_on_path[-1].argmax()
        else:
            assert type(only_output) is int and only_output < outputs_on_path.shape[1], f"Target must be an integer smaller than the number of outputs. {only_output} !< {outputs_on_path.shape}"

        target_logit = outputs_on_path[-1, only_output]

        outputs_target_class = outputs_on_path[:, only_output]
        
        if like_lrp:
            if target_logit > 0:
                # the following is a debatable design choice. We normalise the inital gradient of the target classto get a standard form of the linear operator similar to how we did it in LRP.
                # the resulting operator can take the entire output layer logit vector (not only ones, but real activations), and transform it to the heatmap that integrated gradients would give for every score.
                initial_gradients = torch.ones_like(outputs_target_class) / target_logit
            else:
                # this is silly commputation. here we should just omit the backward pass.
                initial_gradients = torch.zeros_like(outputs_target_class) 
        else:
            initial_gradients = torch.ones_like(outputs_target_class)

        outputs_target_class.backward(initial_gradients)

        submodel_path_gradients = [inp.grad.clone().detach() for inp in submodels_path_inputs]
        submodel_integrated_gradients = [torch.einsum('ij,j...->i...', integration_matrix, grad) for grad in submodel_path_gradients]

        if multiply_by_inputs:
            submodel_integrated_gradients = [grad * (inp[-1:] - inp[-1-baselines_at_steps]) # (avg. grad on path) * (input - baseline)
                                             for inp, grad 
                                             in zip(submodels_path_inputs, submodel_integrated_gradients)]
            

        submodel_integrated_gradients = [g.detach().numpy().copy() for g in submodel_integrated_gradients]
        return submodel_integrated_gradients
    
    def batch_attribute(self, points, n_steps = 1, baselines_at_steps = None, n_classes = 10, **kwargs):
        n_points = len(points)
        
        if baselines_at_steps == 1:
            baselines_at_steps = [n_steps]

        if baselines_at_steps is None:
            n_baselines = n_steps
        else:
            n_baselines = len(baselines_at_steps)


        grads_list = [[self.attribute(inp, baseline_0=None, only_output=t, n_steps=n_steps, baselines_at_steps=baselines_at_steps, **kwargs)
                        for t in range(n_classes)]
                        for inp in points]                                   # (points, classes, layers, baselines, *(layer_size))
        # transpose
        grads = []                                                           # (layers, points, classes, baselines, *(layer_size))
        for i in range(len(grads_list[0][0])):
            for_layer = [[g2[i] for g2 in g1] for g1 in grads_list]
            for_layer = np.array(for_layer).reshape((n_points, n_classes, n_baselines, -1)).transpose((0,2,1,3))
            grads.append(for_layer)

        return grads