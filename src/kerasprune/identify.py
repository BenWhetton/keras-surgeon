"""Identify which channels to delete."""
import pandas as pd
from keras import backend as K

from kerasprune.utils import find_layers


def high_apoz(model,
              layer_index,
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
        layer_index: The index of the layer whose channels will be evaluated 
            for pruning.
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

    if method not in {'std', 'absolute', 'both'}:
        assert ValueError('Invalid `mode` argument. '
                          'Expected one of {"std", "absolute", "both"} '
                          'but got', method)
    if data_format not in {'channels_last', 'channels_first'}:
        assert ValueError('Invalid `data_format` argument. Expected one of '
                          '"channels_last" or "channels_first", but got',
                          data_format)
    # identify the activation layer and the next layer containing weights.
    (activation_layer_index, next_layer_index) = find_layers(model, layer_index)
    # Perform the forward pass and get the activations of the layer.
    get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[activation_layer_index].output])
    activations = get_activations([x_val, 0])

    if data_format == 'channels_last':
        channel_index = -1
    else:
        channel_index = 1
    # TODO: Check that this works for both channels_first and channels_last
    # TODO: remove pandas dependency
    df = pd.DataFrame(activations[0].reshape(-1, activations[0].shape[channel_index]))
    apoz = (df == 0).astype(int).sum(axis=0) / df.shape[0]

    if method == "std":
        high_apoz_channels = apoz[apoz >= apoz.mean() +
                                  apoz.std() * cutoff_std].index
    elif method == 'absolute':
        high_apoz_channels = apoz[apoz >= cutoff_absolute].index
    else:
        high_apoz_channels = apoz[(apoz >= apoz.mean() +
                                   apoz.std() * cutoff_std) |
                                  (apoz >= cutoff_absolute)].index
    return high_apoz_channels, apoz
