"""Prune connections or whole neurons from Keras model layers."""

import numpy as np
from kerasprune.utils import find_layers
# TODO: Add conditional statements for layer option use_bias=False.


def delete_channels(model, layer_index, channels_index):
    """Delete one or more channels (units or filters) from a layer.
    Any weights associated with the deleted channel in its layer and all layers
    downstream of it are deleted.
    All other weights in the model are preserved.
    
    This functionality is currently only implemented for Sequential models 
    comprised of the following layer classes:
    * Dense
    * Conv2D
    * Flatten
    * MaxPool2D
    * Activation
    This covers basic LeNet and AlexNet style architectures.
    
    Notes:
        We use the term channel to interchangeably refer to a unit or a filter 
        depending on the layer type.

    Args:
        model: A Keras model.
        layer_index: The index of the layer containing the channels to be 
        deleted.
        channels_index: The indices of all channels to be deleted.

    Returns:
        A new Keras model with the selected channels removed 
        and all remaining weights set to the input model weights.
        The model must be compiled before use.

    """
    print('Pruning', len(channels_index),
          '/', _get_number_channels(model.layers[layer_index]), 'neurons')

    # identify the activation layer and the next layer containing weights.
    (activation_layer_index, next_layer_index) = find_layers(model, layer_index)

    # Check that this library has been implemented for all layers
    # between the chosen layer and the next weighted layer.
    for layer in model.layers[layer_index:next_layer_index]:
        _check_valid_layer(layer)

    # Remove the units from the model config and initialise the new model
    model_config = model.get_config()
    if model.layers[layer_index].__class__.__name__ == 'Conv2D':
        model_config[layer_index]['config']['filters'] -= len(channels_index)
    elif model.layers[layer_index].__class__.__name__ == 'Dense':
        model_config[layer_index]['config']['units'] -= len(channels_index)
    else:
        raise TypeError('This function has only been implemented for '
                        'convolutional and dense layers.')

    new_model = model.from_config(model_config)

    # This implementation needs serious revision.
    # In future, it should traverse the node structure of the model until it
    # reaches the next layers which must have weights removed, recording any
    # transformations by Flatten, Concatenate or other layers
    [transform, reverse_transform] = _get_transformation(model,
                                                         layer_index,
                                                         next_layer_index)
    [input_axis, output_axis] = _get_channels_axis(model.layers[layer_index])
    # Remove the weights corresponding to the neurons to be deleted from the
    # old model and use the new weights to initialise the new model
    for i, layer in enumerate(model.layers):
        if i == layer_index:
            new_weights = _delete_output_weights(layer, channels_index,
                                                 output_axis)
            new_model.layers[i].set_weights(new_weights)

        elif i == next_layer_index:
            new_weights = _delete_input_weights(layer, channels_index,
                                                input_axis, transform,
                                                reverse_transform)
            new_model.layers[i].set_weights(new_weights)

        else:
            # For all other layers, copy the weights from the un-pruned model.
            new_model.layers[i].set_weights(layer.get_weights())
    return new_model


def _check_valid_layer(layer):
    """Check that this library has been implemented the layer's class."""
    if layer.__class__.__name__ in ('Conv2D',
                                    'Dense',
                                    'MaxPool2D',
                                    'Activation',
                                    'Flatten'):
        assert ValueError('This library has not yet been implemented for ',
                          type(layer), ' layers.')


def _delete_output_weights(layer, channels_index, channels_axis):
    """Delete the weights corresponding to the removed neurons."""
    weights = layer.get_weights()
    new_weights = [np.delete(weights[0], channels_index, axis=channels_axis),
                   np.delete(weights[1], channels_index, axis=0)]
    return new_weights


def _delete_input_weights(layer,
                          delete_channels_index,
                          channels_axis,
                          transform_function=(),
                          reverse_transform_function=()):
    """Delete the input weights in the downstream layer."""
    weights = layer.get_weights()
    # Apply transform to reshape weights to the previous layer's output
    # dimensions
    for f in transform_function[::-1]:
        weights[0] = f(weights[0])
    # Delete the weights corresponding to the neurons (channels) to be deleted.
    new_weights = [np.delete(weights[0], delete_channels_index,
                             axis=channels_axis),
                   weights[1]]
    # Apply the reverse transform to return weights to the correct shape.
    for f in reverse_transform_function:
        new_weights[0] = f(new_weights[0])
    return new_weights


def _get_channels_axis(layer):
    if layer.__class__.__name__ == 'Conv2D':
        if layer.get_config()['data_format'] == 'channels_first':
            input_channels_axis = 0
            output_channels_axis = 1
        else:
            input_channels_axis = -2
            output_channels_axis = -1
    elif layer.__class__.__name__ == 'Dense':
        input_channels_axis = 0
        output_channels_axis = -1
    else:
        raise TypeError('This function has only been implemented for '
                        'convolutional and dense layers.')

    return [input_channels_axis, output_channels_axis]


def _get_transformation(model, layer_index, next_layer_index):
    """Calculate transformations from this layer output to next layer's weights.
    
    This function calculates a list of transformations to projects the weights 
    from the end_layer onto the output space from the start layer and a list of
    functions to reverse the process.
    
    This enables easy identification of the input weights of the end_layer 
    corresponding to the outputs of neurons to be
    deleted in the start_layer."""
    transform_function = []
    reverse_transform_function = []
    next_layer = model.layers[next_layer_index]
    next_layer_weights_shape = next_layer.get_weights()[0].shape
    for i, layer in enumerate(model.layers[layer_index+1:next_layer_index]):
        if layer.__class__.__name__ == 'Flatten':
            transform_function.append(
                lambda x: np.reshape(x, list(layer.input_shape[1:]) + [-1]))
            reverse_transform_function.append(
                lambda x: np.reshape(x, [-1] + [next_layer_weights_shape[-1]]))

    return [transform_function, reverse_transform_function]


def _get_number_channels(layer):
    if layer.__class__.__name__ == 'Conv2D':
        return layer.get_config()['filters']
    elif layer.__class__.__name__ == 'Dense':
        return layer.get_config()['units']
    else:
        raise TypeError('This function has only been implemented for '
                        'convolutional and dense layers.')
