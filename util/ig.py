import torch
import numpy as np


class IntermediateIntegratedGradients():
    """
    Supports two new features:
    - Capture and integrate the gradients at intermediate layers.
    - Compute Path integrals of different lengths at the same time, starting from input, extending towards baseline. For this we use a shared Rieman integral approximation.
    """
    def __init__(self, submodels, device='cpu') -> None:
        """
        Expects a list of submodels, which contain subsets of the layers of the model.
        Before every submodel, the inputs will be stored, and gradients will be captured and integrated.
        """
        self.submodels = submodels
        self.device = device

    def attribute(self, inp, baseline_0=None, output_layer=-1, output_only='argmax', n_steps=50, baselines_at_steps=None, measure_at_layers=None, multiply_by_inputs=True, like_lrp=False, ):
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
        baseline_0 = baseline_0.to(self.device)  # Move baseline to device

        if baselines_at_steps is None:
            baselines_at_steps = 1 + np.arange(n_steps)
        elif isinstance(baselines_at_steps, int):
            baselines_at_steps = np.array([baselines_at_steps])
        elif isinstance(baselines_at_steps, list):
            baselines_at_steps = np.array(baselines_at_steps)
        else:
            assert isinstance(baselines_at_steps, np.ndarray), "baselines_at_steps must be None, int, list or np.ndarray"
        n_baselines = len(baselines_at_steps)

        # to achieve different integral lengths / "#steps in integral" we take the (equally weighted) mean over different steps in the 
        integration_matrix = torch.zeros((n_baselines, n_steps+1), device=self.device)
        for i, at in enumerate(baselines_at_steps):
            integration_matrix[i, (-at):] = 1/at

        # Store inputs
        weights = torch.arange(n_steps+1) / n_steps # the linear combination from the first weight (0) is the baseline. We never use it's gradient, but need it when multiplying with (input - baseline).
        while weights.ndim <= inp.ndim:
            weights = weights[:,None]
        curr_submodel_path_inputs = baseline_0[None] + weights * (inp[None] - baseline_0[None])
        curr_submodel_path_inputs = curr_submodel_path_inputs.to(self.device)

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

        if output_only == 'argmax':
            output_only = outputs_on_path[-1].argmax()
        
        n_output_neurons = np.prod(outputs_on_path.shape[1:])
        if output_only is None:
            output_neuron_indices = np.arange(n_output_neurons)
        else:
            assert type(output_only) is int and output_only < n_output_neurons, f"Target must be an integer smaller than the number of outputs. {output_only} !< {outputs_on_path.shape[1:]}"
            output_neuron_indices = [output_only]
        
        ig_for_all_classes = []
        last_initial_gradients = 0

        for idx_raveled in output_neuron_indices:
            idx = np.unravel_index(idx_raveled, outputs_on_path.shape[1:])
            outputs_target_class = outputs_on_path[:, idx]

            target_logit = outputs_on_path[-1, idx] # does this really work yet if idx is multidimensional?
            if like_lrp:
                if target_logit < 0:
                    if like_lrp=='mask':
                        # add zero gradients for this class for every layer
                        fake_grads_per_layer = []
                        for inp in submodels_path_inputs:
                            grad_shape_flat = np.prod(inp.shape[1:])
                            fake_grads_per_layer.append(np.zeros((n_baselines, grad_shape_flat)))
                        ig_for_all_classes.append(fake_grads_per_layer)
                        continue
                    else:
                        # just skip this class
                        continue
                else:
                    # the following is a debatable design choice. We normalise the inital gradient of the target classto get a standard form of the linear operator similar to how we did it in LRP.
                    # the resulting operator can take the entire output layer logit vector (not only ones, but real activations), and transform it to the heatmap that integrated gradients would give for every score.
                    initial_gradients = torch.ones_like(outputs_target_class, device=self.device) / target_logit
            else:
                initial_gradients = torch.ones_like(outputs_target_class, device=self.device)

            outputs_target_class.backward(initial_gradients - last_initial_gradients, create_graph=True)
            last_initial_gradients = initial_gradients

            submodel_integrated_gradients = [torch.einsum('ij,j...->i...', integration_matrix, inp.grad.detach()) for inp in submodels_path_inputs]

            if multiply_by_inputs:
                submodel_integrated_gradients = [grad * (inp[-1:] - inp[-1-baselines_at_steps]) # (avg. grad on path) * (input - baseline)
                                                for inp, grad 
                                                in zip(submodels_path_inputs, submodel_integrated_gradients)]

            submodel_integrated_gradients = [g.detach().flatten(start_dim=1).numpy().copy() for g in submodel_integrated_gradients]
            ig_for_all_classes.append(submodel_integrated_gradients)

        return ig_for_all_classes
    
    def batch_attribute(self, points, n_steps = 1, baselines_at_steps = None, n_classes = 10, **kwargs):
        n_points = len(points)
        
        if baselines_at_steps == 1:
            baselines_at_steps = [n_steps]

        if baselines_at_steps is None:
            n_baselines = n_steps
        else:
            n_baselines = len(baselines_at_steps)


        grads_list = [self.attribute(inp, baseline_0=None, output_only=None, n_steps=n_steps, baselines_at_steps=baselines_at_steps, **kwargs)
                        for inp in points]                                   # (points, classes, layers, baselines, *(layer_size))
        
        print('grads_list', len(grads_list), len(grads_list[0]), len(grads_list[0][0]), grads_list[0][0][0].shape)
        # transpose
        grads = []                                                           # (layers, points, classes, baselines, *(layer_size))
        n_layers = len(grads_list[0][0])
        for i_layer in range(n_layers):
            for_layer = [[one_class[i_layer] for one_class in one_point] for one_point in grads_list]
            print(for_layer[0][0].shape)
            for_layer = np.array(for_layer)
            print(for_layer.shape)
            for_layer  =for_layer.reshape((n_points, n_classes, n_baselines, -1)).transpose((0,2,1,3))
            print(for_layer.shape)
            grads.append(for_layer)
            print()

        return grads