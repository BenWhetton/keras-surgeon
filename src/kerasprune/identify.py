"""Identify which channels to delete."""
import numpy as np
# noinspection PyPep8Naming
from keras import backend as K

from kerasprune import utils


def high_apoz(model,
              layer,
              x_val,
              method="std",
              cutoff_std=1,
              cutoff_absolute=0.99,
              data_format='channels_last'):
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
        method: Cutoff method for high APoZ. "std", "absolute" or "both".
        cutoff_std: Channels with a higher APoZ than the layer mean plus
            `cutoff_std` standard deviations will be identified for pruning.
        cutoff_absolute: Channels with a higher APoZ than `cutoff_absolute`
            will be identified for pruning.
        data_format

    Returns:
        Tuple of lists:
        * high_apoz_channels: List of indices of channels with high APoZ.
        * apoz: List of the APoZ values for all channels in the layer.
    """
    layer_config = layer.get_config()
    if method not in {'std', 'absolute', 'both'}:
        raise ValueError('Invalid `mode` argument. '
                         'Expected one of {"std", "absolute", "both"} '
                         'but got', method)
    if data_format not in {'channels_last', 'channels_first'}:
        raise ValueError('Invalid `data_format` argument. Expected one of '
                         '"channels_last" or "channels_first", but got',
                         data_format)
    if layer_config['activation'] == 'linear':
        raise ValueError('High APoZ cannot be used on a layer with linear '
                         'activation.')

    node_indices = utils.find_nodes_in_model(model, layer)
    # Perform the forward pass and get the activations of the layer.
    activations = []
    for node_index in node_indices:
        get_activations = K.function([utils.single_element(model.inputs),
                                      K.learning_phase()],
                                     [layer.get_output_at(node_index)])
        a = get_activations([x_val, 0])[0]

        # Ensure that the channels axis is last
        if ('data_format' in layer_config.keys()) and \
                (layer_config['data_format'] == 'channels_first'):
            a = np.swapaxes(a, 1, -1)
        # Flatten all except channels axis
        a = np.reshape(a, [-1, a.shape[-1]])
        if not activations:
            activations = a
        else:
            activations = np.append(activations, a, axis=0)

    apoz = (activations == 0).astype(int).sum(axis=0) / activations.shape[0]

    if method == "std":
        high_apoz_channels = np.where(apoz >= apoz.mean() +
                                      apoz.std() * cutoff_std)[0]
    elif method == 'absolute':
        high_apoz_channels = np.where(apoz >= cutoff_absolute)[0]
    else:
        high_apoz_channels = np.where((apoz >= apoz.mean() +
                                       apoz.std() * cutoff_std) |
                                      (apoz >= cutoff_absolute))[0]
    return high_apoz_channels, apoz
