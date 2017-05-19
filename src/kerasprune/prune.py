import keras
import numpy as np
from keras.models import Sequential
from kerasprune.utils import find_layers

# This module provides methods to prune keras models.
# TODO: Add conditional statements for layer option use_bias=False.


def delete_channels(model, layer_index, delete_channels_index):
    """Delete one or more channels from a model.
    Any weights associated with the deleted channel in its layer and all layers downstream of it are deleted.
    All other weights in the model are preserved.
    
    This functionality is currently only implemented for Sequential models comprised of the following layer classes:
    * Dense
    * Conv2D
    * Flatten
    * MaxPool2D
    * Activation
    This covers LeNet and AlexNet style architectures.
    
    Notes:
        We use the term channel to refer to a unit for most layers or a filter in convolutional layers.

    Args:
        model: A Keras model.
        layer_index: The index of the layer containing the channels to be deleted.
        delete_channels_index: The indices of all channels to be deleted.

    Returns:
        new_model: A new Keras model with the selected channels removed 
        and all remaining weights set to the input model weights

    """
    print('Pruning', len(delete_channels_index), '/', _get_number_channels(model.layers[layer_index]), 'neurons')

    # identify the activation layer and the next layer containing weights.
    (activation_layer_index, next_layer_index) = find_layers(model, layer_index)

    # Check that this library has been implemented for all layers
    # between the chosen layer and the next weighted layer.
    for layer in model.layers[layer_index:next_layer_index]:
        _check_valid_layer(layer)

    # Remove the units from the model config and initialise the new model
    model_config = model.get_config()
    if isinstance(model.layers[layer_index], keras.layers.convolutional.Conv2D):
        model_config[layer_index]['config']['filters'] -= len(delete_channels_index)
    elif isinstance(model.layers[layer_index], keras.layers.core.Dense):
        model_config[layer_index]['config']['units'] -= len(delete_channels_index)
    else:
        raise TypeError('This function has only been implemented for convolutional and dense layers.')

    new_model = model.from_config(model_config)

    [transform_function, reverse_transform_function] = _get_weights_transformation(model, layer_index, next_layer_index)
    [input_channels_axis, output_channels_axis] = _get_channels_axis(model.layers[layer_index])
    # Remove the weights corresponding to the neurons to be deleted from the old model and use the new weights to
    #  initialise the new model
    for i, layer in enumerate(model.layers):
        if i == layer_index:
            new_model.layers[i].set_weights(_delete_output_weights(layer, delete_channels_index, output_channels_axis))

        elif i == next_layer_index:
            new_model.layers[i].set_weights(_delete_input_weights(layer, delete_channels_index, input_channels_axis,
                                                                  transform_function, reverse_transform_function))

        else:
            # For all other layers, copy across the weights from the un-pruned model.
            new_model.layers[i].set_weights(layer.get_weights())
    return new_model


def _check_valid_layer(layer):
    """Check that this library has been implemented the layer's class."""
    if not isinstance(layer, (keras.layers.Conv2D,
                              keras.layers.Dense,
                              keras.layers.MaxPool2D,
                              keras.layers.Activation,
                              keras.layers.Flatten)):
        assert ValueError('This library has not yet been implemented for ', type(layer), ' layers.')


def _delete_output_weights(layer, channels_index, channels_axis):
    # Delete the weights in the previous layer corresponding to the removed neurons.
    weights = layer.get_weights()
    new_weights = [np.delete(weights[0], channels_index, axis=channels_axis),
                   np.delete(weights[1], channels_index, axis=0)]
    return new_weights


def _delete_input_weights(layer,
                          delete_channels_index,
                          channels_axis,
                          transform_function=(),
                          reverse_transform_function=()):
    weights = layer.get_weights()
    # Apply transform functions to reshape weights to the previous layer's output dimensions
    for f in transform_function[::-1]:
        weights[0] = f(weights[0])
    # Delete the weights corresponding to the neurons (channels) to be deleted.
    new_weights = [np.delete(weights[0], delete_channels_index, axis=channels_axis),
                   weights[1]]
    # Apply the reverse transform functions to return weights to the correct shape
    for f in reverse_transform_function:
        new_weights[0] = f(new_weights[0])
    return new_weights


def _get_channels_axis(layer):
    if isinstance(layer, keras.layers.convolutional.Conv2D):
        if layer.get_config()['data_format'] == 'channels_first':
            input_channels_axis = 0
            output_channels_axis = 1
        else:
            input_channels_axis = -2
            output_channels_axis = -1
    elif isinstance(layer, keras.layers.core.Dense):
        input_channels_axis = 0
        output_channels_axis = -1
    else:
        raise TypeError('This function has only been implemented for convolutional and dense layers.')

    return [input_channels_axis, output_channels_axis]


def _get_weights_transformation(model, layer_index, next_layer_index):
    """This function calculates a list of functions to projects the weights from the end_layer onto the output space from the
    start layer and a list of function to reverse the process.
    This enables easy identification of the input weights of the end_layer corresponding to the outputs of neurons to be
    deleted in the start_layer."""
    transform_function = []
    reverse_transform_function = []
    next_layer_weights_shape = model.layers[next_layer_index].get_weights()[0].shape
    for i, layer in enumerate(model.layers[layer_index+1:next_layer_index]):
        if isinstance(layer, keras.layers.core.Flatten):
            transform_function.append(lambda x: np.reshape(x, list(layer.input_shape[1:]) + [-1]))
            reverse_transform_function.append(lambda x: np.reshape(x, [-1] + [next_layer_weights_shape[-1]]))

    return [transform_function, reverse_transform_function]


def _get_number_channels(layer):
    if isinstance(layer, keras.layers.convolutional.Conv2D):
        return layer.get_config()['filters']
    elif isinstance(layer, keras.layers.core.Dense):
        return layer.get_config()['units']
    else:
        raise TypeError('This function has only been implemented for convolutional and dense layers.')
