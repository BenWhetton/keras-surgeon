"""Identify which channels to delete."""
import numpy as np
from keras import backend as k
from keras.preprocessing.image import Iterator
from keras.models import Model

from kerasprune import utils


def get_apoz(model, layer, x_val):
    """Identify neurons with high Average Percentage of Zeros (APoZ).

    The APoZ a.k.a. (A)verage (P)ercentage (o)f activations equal to (Z)ero,
    is a metric for the usefulness of a channel defined in this paper:
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient
    Deep Architectures" - [Hu et al. (2016)][]
    `high_apoz()` enables the pruning methodology described in this paper to be
    replicated

    Args:
        model: A Keras model.
        layer: The layer whose channels will be evaluated for pruning.
        x_val: The input of the validation set. This will be used to calculate
            the activations of the layer of interest.

    Returns:
        List of the APoZ values for each channel in the layer.
    """

    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    data_format = getattr(layer, 'data_format', 'channels_last')
    node_indices = utils.find_nodes_in_model(model, layer)
    # Perform the forward pass and get the activations of the layer.
    activations = []
    for node_index in node_indices:
        act_layer, act_index = utils.find_activation_layer(layer, node_index)
        # Get activations
        if isinstance(x_val, Iterator):
            temp_model = Model(model.inputs, act_layer.get_output_at(act_index))
            a = temp_model.predict_generator(
                    x_val, x_val.n // x_val.batch_size)
        else:
            get_activations = k.function(
                [utils.single_element(model.inputs), k.learning_phase()],
                [act_layer.get_output_at(act_index)])
            a = get_activations([x_val, 0])[0]
        # Ensure that the channels axis is last
        if data_format == 'channels_first':
            a = np.swapaxes(a, 1, -1)
        # Flatten all except channels axis and add to list
        activations.append(np.reshape(a, [-1, a.shape[-1]]))
    # Concatenate all activations from this layers nodes and calculate the
    # average percentage of zeros for each channel.
    activations = np.concatenate(activations, axis=0)
    return (activations == 0).astype(int).sum(axis=0) / activations.shape[0]


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
