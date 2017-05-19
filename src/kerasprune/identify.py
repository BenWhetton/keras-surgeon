import pandas as pd
from keras import backend as K

from kerasprune.utils import find_layers


def high_apoz(model,
              layer_index,
              x_val,
              mode="std",
              cutoff_std=1,
              cutoff_absolute=0.99,
              data_format='channels_last'):
    if mode not in {'std', 'absolute', 'both'}:
        assert ValueError('Invalid `mode` argument. '
                          'Expected one of {"std", "absolute", "both"} '
                          'but got', mode)
    if data_format not in {'channels_last', 'channels_first'}:
        assert ValueError('Invalid `data_format` argument. '
                          'Expected one of {"channels_last" or "channels_first"} '
                          'but got', data_format)

    # identify the activation layer and the next layer containing weights.
    (activation_layer_index, next_layer_index) = find_layers(model, layer_index)
    # Perform the forward pass and get the activations of the layer.
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[activation_layer_index].output])
    activations = get_activations([x_val, 0])

    if data_format == 'channels_last':
        channel_index = -1
    else:
        channel_index = 1
    # TODO: Check that this works for both channels_first and channels_last!
    # TODO: remove pandas dependency
    df = pd.DataFrame(activations[0].reshape(-1, activations[0].shape[channel_index]))
    apoz = (df == 0).astype(int).sum(axis=0) / df.shape[0]

    if mode == "std":
        high_apoz_neurons = apoz[apoz >= apoz.mean() + apoz.std() * cutoff_std].index
    elif mode == 'absolute':
        high_apoz_neurons = apoz[apoz >= cutoff_absolute].index
    else:
        high_apoz_neurons = apoz[(apoz >= apoz.mean() + apoz.std() * cutoff_std) | (apoz >= cutoff_absolute)].index
    return high_apoz_neurons, apoz
