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


# check if layer weights are empty, if they are, check layer type and add relevant transforms and reverse transforms to the lamda list and call
def layer_recurse(layer):
    if not layer.weights:
        # Add transforms and reverse transforms depending on layer type
        1
    else:
        1
        # apply transforms
        # delete weights
        # apply reverse transforms
        # assign new weights to layer

# for each depth level, for each layer, instantiate them wrt previous layers. store new outputs in dict.

def rebuild_model(input_layer):
    for node in input_layer.outbound_nodes:
        node.output_layer(input_layer.output)
        rebuild_model(node)
    return


def rebuild_sequential(model):
    from keras.models import Model, Sequential
    temp_model = Sequential.from_config(model.get_config())
    temp_model.set_weights(model.get_weights())
    layer = temp_model.layers[0]
    first_input = layer.input
    input_tensor = first_input

    # next_input = layer(input)
    while layer.outbound_nodes:
        # While the layer contains outbound nodes
        # Loop invariant: "input_tensor" is the output of the n-1th layer.
        # "layer" is the nth layer
        # The layers up to layer n-1 have been connected in sequence
        input_tensor = layer(input_tensor)
        layer = layer.outbound_nodes[0].outbound_layer

    return Model(inputs=first_input, outputs=layer(input_tensor))


def rebuild_sequential2(model):
    from keras.models import Sequential

    layers = []
    weights = []
    # next_input = layer(input)
    for layer in model.layers:
        layers.append(type(layer).from_config(layer.get_config()))
        weights.append(layer.get_weights())

    new_model = Sequential(layers=layers)
    for i, layer in enumerate(new_model.layers):
        layer.set_weights(weights[i])

    return new_model


def rebuild_sequential_rec(model):
    from keras.models import Model
    input_layer = model.layers[1]
    input_tensor = input_layer.input
    output_tensor = _rebuild_sequential_rec(input_tensor, input_layer)
    return Model(inputs=input_tensor, outputs=output_tensor)


def _rebuild_sequential_rec(input, layer):
    next_input = layer(input)
    if not layer.outbound_nodes:
        return next_input
    next_layer = layer.outbound_nodes[0].outbound_layer
    next_input = _rebuild_sequential_rec(next_input, next_layer)
    return next_input


def rebuild(model):
    """Rebuild the model"""
    from keras.models import Model
    model_inputs = model.inputs
    finished_nodes = set()
    # find the previous layers connecting to this layer.

    def _rebuild_rec(layer, node_index):
        """Rebuilds the instance of this layer and all deeper layers, recursively.
        
        Calculates the output tensor by applying this layer to this node.
        All tensors deeper in the network are also calculated
        
        Args:
            layer: the layer to rebuild
            node_index: The index of the next inbound node in the network. 
                        The layer will be called on the output of this node to 
                        obtain the output.
                        inbound_node = layer.inbound_nodes[node_index].
        Returns:
            The output of the layer when called on the output of the inbound node.
                             
        """
        ###print('getting inputs for: ', layer.name)
        # get the inbound node
        inbound_node = layer.inbound_nodes[node_index]
        if inbound_node in finished_nodes:
            output = layer.get_output_at(node_index)
            return output

        inbound_layers = inbound_node.inbound_layers
        inbound_node_indices = inbound_node.node_indices
        # find this layer's inbound layer(s) recursively.
        inputs = []
        ###print('inbound_layers: ', inbound_layers)
        for inbound_layer, inbound_node_index in zip(inbound_layers,
                                                     inbound_node_indices):
            if not inbound_layer:
                ###print('bottomed out to an unknown input')
                raise ValueError
            elif inbound_layer.get_output_at(inbound_node_index) \
                    in model_inputs:
                ###print('bottomed out')
                inputs.append(inbound_layer.get_output_at(inbound_node_index))
            else:
                inputs.append(_rebuild_rec(inbound_layer, inbound_node_index))
        # call this layer on the outputs of the inbound layers
        ###print("inputs: ", inputs)
        if len(inputs) == 1:
            output = layer(inputs[0])
        else:
            output = layer(inputs)
        finished_nodes.add(inbound_node)
        return output
        # if the previous layer is empty, bottom out

    new_model_outputs = []
    for output_layer, output_layer_node_index in \
            zip(model.output_layers, model.output_layers_node_indices):
        new_model_outputs.append(_rebuild_rec(output_layer,
                                              output_layer_node_index))

    new_model = Model(model_inputs, new_model_outputs)

    return new_model

# recursively explore the tree from outputs to inputs,




# def delete_layer(model, layer_name):
#     find
#     return Model(inputs = inputs)