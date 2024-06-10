# Conventions for file names and their stored data
- mats: matrices representing the backward pass of an explanation method
    - (nn_subsection, points, output_shape_flattened, inp_shape_flattened)
    - normalized to sum to one across input_neurons
    - such pairs (point, output_neuron) that have an unactivated output_neuron, have all values set to 0.
- LRP:
    - old format for LRP backward pass specifically:
    - similar to *mats* but 
        - does not normalize explicitly
        - sometimes has unactivated columns deleted, not masked to 0.