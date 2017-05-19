import keras

from kerasprune.identify import high_apoz
from kerasprune.prune import delete_channels


def apoz_example(model, layer_index, x_val, y_val):
    # Identify the neurons with a high percentage of zero activations on the data set x_val
    [high_apoz_neurons, apoz_data] = high_apoz(model, layer_index, x_val)

    new_model = delete_channels(model, layer_index, high_apoz_neurons)

    return new_model, apoz_data