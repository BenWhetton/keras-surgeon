"""Identify which channels to delete."""
import numpy as np
from keras import backend as k
from keras.models import Model

from kerassurgeon import utils


def get_apoz(model, layer, x_val, node_indices=None, batch_size=1):
    """Identify neurons with high Average Percentage of Zeros (APoZ).

    The APoZ a.k.a. (A)verage (P)ercentage (o)f activations equal to (Z)ero,
    is a metric for the usefulness of a channel defined in this paper:
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient
    Deep Architectures" - [Hu et al. (2016)][]
    `high_apoz()` enables the pruning methodology described in this paper to be
    replicated.

    If node_indices are not specified and the layer is shared within the model
    the APoZ will be calculated over all instances of the shared layer.

    Args:
        model: A Keras model.
        layer: The layer whose channels will be evaluated for pruning.
        x_val: The input of the validation set. This will be used to calculate
            the activations of the layer of interest.
        node_indices(list[int]): (optional) A list of node indices.

    Returns:
        List of the APoZ values for each channel in the layer.
    """

    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    mean_calculator = utils.MeanCalculator(sum_axis=0)
    for node_index in node_indices:
        act_layer, act_index = utils.find_activation_layer(layer, node_index)
        # Get activations
        if isinstance(x_val, np.ndarray):
            temp_model = Model(model.inputs,
                               act_layer.get_output_at(act_index))
            a = temp_model.predict_generator(
                x_val, x_val.shape[0] // batch_size)
        elif hasattr(x_val, "__iter__"):
            temp_model = Model(model.inputs,
                               act_layer.get_output_at(act_index))
            a = temp_model.predict_generator(
                x_val, x_val.n // batch_size)
        else:
            get_activations = k.function(
                [utils.single_element(model.inputs), k.learning_phase()],
                [act_layer.get_output_at(act_index)])
            a = get_activations([x_val, 0])[0]
            # Ensure that the channels axis is last
        if data_format == 'channels_first':
            a = np.swapaxes(a, 1, -1)
        # Flatten all except channels axis
        activations = np.reshape(a, [-1, a.shape[-1]])
        zeros = (activations == 0).astype(int)
        mean_calculator.add(zeros)

    return mean_calculator.calculate()


def high_apoz(apoz, method="std", cutoff_std=1, cutoff_absolute=0.99):
    """
    Args:
        apoz: List of the APoZ values for each channel in the layer.
        method: Cutoff method for high APoZ. "std", "absolute" or "both".
        cutoff_std: Channels with a higher APoZ than the layer mean plus
            `cutoff_std` standard deviations will be identified for pruning.
        cutoff_absolute: Channels with a higher APoZ than `cutoff_absolute`
            will be identified for pruning.

    Returns:
        high_apoz_channels: List of indices of channels with high APoZ.

    """
    if method not in {'std', 'absolute', 'both'}:
        raise ValueError('Invalid `mode` argument. '
                         'Expected one of {"std", "absolute", "both"} '
                         'but got', method)
    if method == "std":
        cutoff = apoz.mean() + apoz.std()*cutoff_std
    elif method == 'absolute':
        cutoff = cutoff_absolute
    else:
        cutoff = min([cutoff_absolute, apoz.mean() + apoz.std()*cutoff_std])

    return np.where(apoz >= cutoff)[0]
