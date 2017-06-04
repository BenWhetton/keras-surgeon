"""Prune connections or whole neurons from Keras model layers."""

import numpy as np
import logging
from kerasprune.utils import find_layers
from keras.models import Model
# TODO: Add conditional statements for layer option use_bias=False.
logging.basicConfig(level=logging.DEBUG)

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


def rebuild_sequential(model):
    from keras.models import Sequential

    layers = []
    weights = []
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


def replace_layer_instance(model, layer, new_layer, node_index=None, copy=True):
    if layer not in model.layers:
        raise ValueError('layer is not a valid layer in model.')
    if check_for_layer_reuse(model):
        if not node_index:
            raise ValueError('A node_index must be specified if any layers in '
                             'the model are re-used within the model or in '
                             'other models.')
        if copy:
            raise ValueError('The model cannot be cleanly copied if any '
                             'layers in the model are re-used within the '
                             'model or in other models. Set copy=False.')

    if not node_index:
        node_index = 0
    if copy:
        model = clean_copy(model)
        layer = model.get_layer(layer.get_config()['name'])

    # rebuild model up to node
    model_inputs = model.inputs
    output_layers = model.output_layers
    output_layers_node_indices = model.output_layers_node_indices
    inbound_node = layer.inbound_nodes[node_index]
    inbound_layers = inbound_node.inbound_layers
    inbound_node_indices = inbound_node.node_indices

    # rebuild the model up to the deleted layer
    logging.debug('rebuilding model up to the layer before the insertion: {0}'.format(layer))
    inbound_layers_outputs, finished_outputs = rebuild_submodel(model_inputs,
                                                                inbound_layers,
                                                                inbound_node_indices)
    # add the new layer to the outputs of the inbound layers
    if len(inbound_layers_outputs) == 1:
        inbound_layers_outputs = inbound_layers_outputs[0]
    new_output = new_layer(inbound_layers_outputs)
    # replace_inputs = {}

    deleted_layer_output = layer.get_output_at(node_index)

    replace_inputs = {deleted_layer_output: new_output}

    new_model_outputs, _ = rebuild_submodel(model_inputs,
                                            output_layers,
                                            output_layers_node_indices,
                                            replace_inputs,
                                            finished_outputs)
    new_model = Model(model_inputs, new_model_outputs)
    if copy:
        return clean_copy(new_model)
    else:
        return new_model


def insert_layer(model, layer, new_layer, node_index=None, copy=True):
    """Insert new_layer before layer at node_index.
    
    If node_index must be specified if there is more than one inbound node.
    
    Args:
        model: Keras Model object.
        layer: Keras Layer object contained in model.
        new_layer: a layer to be inserted into model before layer.
        node_index: the index of the inbound_node to layer where new layer is 
                    to be inserted.
        
    Returns:
        a new Keras Model object with layer inserted.
        
    Raises:
        blaError: if layer is not contained by model
        valueError: if new_layer is not compatible with the input and output 
                    dimensions of the layers preceding and following it.
        valueError: if node_index does not correspond to one of layer's inbound
                    nodes.
    """
    if layer not in model.layers:
        raise ValueError('layer is not a valid layer in model.')
    if check_for_layer_reuse(model):
        if not node_index:
            raise ValueError('A node_index must be specified if any layers in '
                             'the model are re-used within the model or in '
                             'other models.')
        if copy:
            raise ValueError('The model cannot be cleanly copied if any '
                             'layers in the model are re-used within the '
                             'model or in other models. Set copy=False.')
    if not node_index:
        node_index = 0
    if copy:
        model = clean_copy(model)
        layer = model.get_layer(layer.get_config()['name'])
    # rebuild model up to node
    model_inputs = model.inputs
    output_layers = model.output_layers
    output_layers_node_indices = model.output_layers_node_indices
    inbound_node = layer.inbound_nodes[node_index]
    inbound_layers = inbound_node.inbound_layers
    inbound_node_indices = inbound_node.node_indices

    # rebuild the model up to the deleted layer
    logging.debug('rebuilding model up to the layer before the insertion: {0}'.format(layer))
    inbound_outputs, finished_outputs = rebuild_submodel(model_inputs,
                                                         inbound_layers,
                                                         inbound_node_indices)
    # add the new layer to the output of each inbound layer
    replace_inputs = {}
    inserted_layer_outputs = []
    for inbound_layer, index, prev_output in zip(inbound_layers,
                                                 inbound_node_indices,
                                                 inbound_outputs):
        output = new_layer(prev_output)
        inserted_layer_outputs.append(output)
        replace_inputs[inbound_layer.get_output_at(index)] = output

    new_model_outputs, _ = rebuild_submodel(inserted_layer_outputs,
                                            output_layers,
                                            output_layers_node_indices,
                                            replace_inputs,
                                            finished_outputs)
    new_model = Model(model_inputs, new_model_outputs)
    if copy:
        return clean_copy(new_model)
    else:
        return new_model


def delete_layer_instance(model, layer, node_index=None, copy=True):
    """Delete an instance of a layer from a Keras model.
    
    Args:
        model: Keras Model object.
        layer: Keras Layer object contained in model.
        node_index: the index of the inbound_node to the layer to be deleted.

    Returns:
        Keras Model object with the layer at node_index deleted.
    """
    if layer not in model.layers:
        raise ValueError('layer is not a valid layer in model.')
    if check_for_layer_reuse(model):
        if not node_index:
            raise ValueError('A node_index must be specified if any layers in '
                             'the model are re-used within the model or in '
                             'other models.')
        if copy:
            raise ValueError('The model cannot be cleanly copied if any '
                             'layers in the model are re-used within the '
                             'model or in other models. Set copy=False.')

    if not node_index:
        node_index = 0
    if copy:
        model = clean_copy(model)
        layer = model.get_layer(layer.get_config()['name'])

    # rebuild model up to node
    model_inputs = model.inputs
    output_layers = model.output_layers
    output_layers_node_indices = model.output_layers_node_indices
    inbound_node = layer.inbound_nodes[node_index]
    inbound_layers = inbound_node.inbound_layers
    inbound_node_indices = inbound_node.node_indices

    # rebuild the model up to the deleted layer
    logging.debug('rebuilding model up to deleted layer: {0}'.format(layer))
    deleted_inputs, finished_outputs = rebuild_submodel(model_inputs,
                                                        inbound_layers,
                                                        inbound_node_indices)
    deleted_layer_output = layer.get_output_at(node_index)

    replace_inputs = {deleted_layer_output: deleted_inputs[i]
                      for i in range(len(deleted_inputs))}

    logging.debug('rebuilding the rest of the model')
    new_model_outputs, _ = rebuild_submodel(model_inputs,
                                            output_layers,
                                            output_layers_node_indices,
                                            replace_inputs=replace_inputs,
                                            finished_outputs=finished_outputs)
    new_model = Model(model_inputs, new_model_outputs)
    if copy:
        return clean_copy(new_model)
    else:
        return new_model


def delete_channels_rec(model, layer, channels_index, node_index=None, copy=None):
    """Delete channels from layer"""
    if layer not in model.layers:
        raise ValueError('layer is not a valid layer in model.')
    if check_for_layer_reuse(model):
        if not node_index:
            raise ValueError('A node_index must be specified if any layers in '
                             'the model are re-used within the model or in '
                             'other models.')
        if copy:
            raise ValueError('The model cannot be cleanly copied if any '
                             'layers in the model are re-used within the '
                             'model or in other models. Set copy=False.')

    if not node_index:
        node_index = 0
    if copy:
        model = clean_copy(model)
        layer = model.get_layer(layer.get_config()['name'])

    # Rebuild the model up to layer
    model_inputs = model.inputs
    output_layers = model.output_layers
    output_layers_node_indices = model.output_layers_node_indices
    inbound_node = layer.inbound_nodes[node_index]
    inbound_layers = inbound_node.inbound_layers
    inbound_node_indices = inbound_node.node_indices

    logging.debug(
        'rebuilding model up to the layer before the insertion: {0}'.format(
            layer))
    input_delete_masks = [np.zeros(node.outbound_layer.input_shape[1:],
                                   dtype=bool) for node in model.inbound_nodes]
    inbound_layers_outputs, _, finished_outputs = rebuild_submodel_transform(model_inputs,
                                                                          inbound_layers,
                                                                          inbound_node_indices,
                                                                          input_delete_masks=input_delete_masks)

    # Delete the channels in layer
    [input_axis, output_axis] = _get_channels_axis(layer)
    new_weights = _delete_output_weights(layer, channels_index,
                                         output_axis)
    layer_config = layer.get_config()
    if 'units' in layer_config.keys():
        layer_config['units'] -= len(channels_index)
    elif 'filters' in layer_config.keys():
        layer_config['filters'] -= len(channels_index)
    else:
        raise ValueError('The layer must have either a "units" or "filters" '
                         'property to be able to delete channels.')

    new_layer = type(layer).from_config(layer_config)

    # add the new layer to the outputs of each inbound layers
    if len(inbound_layers_outputs) == 1:
        inbound_layers_outputs = inbound_layers_outputs[0]
    new_output = new_layer(inbound_layers_outputs)
    new_weights = _delete_output_weights(layer, channels_index, output_axis)
    new_layer.set_weights(new_weights)
    # create delete mask for the modified layer. This will be propagated
    # through the model to delete all weights which were connected to
    # the deleted channels
    new_delete_mask = np.zeros(layer.output_shape[1:], dtype=bool)
    index = [slice(None)] * new_delete_mask.ndim
    index[output_axis] = channels_index
    new_delete_mask[tuple(index)] = True

    deleted_layer_output = layer.get_output_at(node_index)

    replace_inputs = {deleted_layer_output: (new_output, new_delete_mask)}

    # Rebuild the rest of the model
    new_model_outputs, _, _ = rebuild_submodel_transform(model_inputs,
                                            output_layers,
                                            output_layers_node_indices,
                                            replace_inputs,
                                            finished_outputs,
                                            input_delete_masks)
    new_model = Model(model_inputs, new_model_outputs)
    if copy:
        return clean_copy(new_model)
    else:
        return new_model


def rebuild_submodel_transform(model_inputs,
                               output_layers,
                               output_layers_node_indices,
                               replace_inputs=None,
                               finished_outputs=None,
                               input_delete_masks=None):
    """Rebuild the model"""
    if not finished_outputs:
        finished_outputs = {}
    if not replace_inputs:
        replace_inputs = {}

    def _rebuild_rec(layer, node_index):
        """Rebuilds the instance of layer and all deeper layers recursively.

        Calculates the output tensor by applying this layer to this node.
        All tensors deeper in the network are also calculated

        Args:
            layer: the layer to rebuild
            node_index: The index of the next inbound node in the network. 
                        The layer will be called on the output of this node to 
                        obtain the output.
                        inbound_node = layer.inbound_nodes[node_index].
        Returns:
            The output of the layer on the data stream indicated by node_index.

        """
        logging.debug('getting inputs for: {0}'.format(layer.name))
        # get the inbound node
        inbound_node = layer.inbound_nodes[node_index]
        if inbound_node in finished_outputs.keys():
            logging.debug('reached finished node: {0}'.format(inbound_node))
            return finished_outputs[inbound_node]

        inbound_layers = inbound_node.inbound_layers
        inbound_node_indices = inbound_node.node_indices
        # find this layer's inputs recursively.
        # find the transforms leading from the input to this layer recursively.
        inputs = []
        delete_masks = []
        logging.debug('inbound_layers: {0}'.format(inbound_layers))
        for inbound_layer, inbound_node_index in zip(inbound_layers,
                                                     inbound_node_indices):
            inbound_layer_output = \
                inbound_layer.get_output_at(inbound_node_index)
            if not inbound_layer:
                logging.debug('bottomed out to an unknown input')
                raise ValueError
            elif inbound_layer_output in model_inputs:
                logging.debug('bottomed out at a model input')
                output = inbound_layer_output
                delete_mask = input_delete_masks[
                    model_inputs.index(inbound_layer_output)]
            elif inbound_layer_output in replace_inputs.keys():
                logging.debug('bottomed out at replaced output: {0}'.format(
                    inbound_layer_output))
                output, delete_mask = replace_inputs[inbound_layer_output]

            else:
                output, delete_mask = _rebuild_rec(inbound_layer,
                                                    inbound_node_index)
            inputs.append(output)
            delete_masks.append(delete_mask)
        # call this layer on the outputs of the inbound layers
        if len(inputs) == 1:
            inputs = inputs[0]
        if len(delete_masks) == 1:
            delete_masks = delete_masks[0]
        output, delete_mask = apply_layer_delete_mask(layer, inputs, delete_masks)
        finished_outputs[inbound_node] = [output, delete_mask]
        return output, delete_mask

    new_model_outputs = []
    output_delete_masks = []
    for output_layer, output_layer_node_index in \
            zip(output_layers, output_layers_node_indices):
        output, delete_mask = _rebuild_rec(output_layer, output_layer_node_index)
        new_model_outputs.append(output)
        output_delete_masks.append(delete_mask)
    return new_model_outputs, output_delete_masks, finished_outputs


def apply_layer_delete_mask(layer, inputs, input_delete_masks):
    # if delete_mask is None, the deleted channels do not affect this layer or
    # any layers above it
    new_weights = None
    if input_delete_masks is not None:
        # otherwise, delete_mask.shape should be: layer.input_shape[1:]
        layer_class = layer.__class__.__name__
        if layer_class == 'Input':
            raise RuntimeError('This should never get here!')
            output_delete_mask = np.zeros(layer.output_shape[1:], dtype=bool)

        elif layer_class == 'Dense':
            new_layer = type(layer).from_config(layer.get_config())
            weights = layer.get_weights()
            new_weights = weights
            # logging.debug('mask: {0}'.format(input_delete_masks))
            # keep_mask = np.dstack([np.logical_not(input_delete_masks)] * weights[0].shape[-1]).squeeze()
            # keep_mask_slice = np.logical_not(input_delete_masks)
            # keep_mask = np.repeat(keep_mask_slice.reshape(list(keep_mask_slice.shape) + [1]), weights[0].shape[-1], axis=new_weights[0].ndim-1)
            new_weights[0] = new_weights[0][np.where(input_delete_masks == False)[0], :]
            # new_weights.append(weights[0][input_delete_masks, :])
            # new_weights.append(weights[1])
            # new_layer.set_weights(new_weights)
            layer = new_layer
            output = layer(inputs)
            layer.set_weights(new_weights)
            output_delete_mask = np.zeros(layer.output_shape[1:], dtype=bool)

        elif layer_class == 'Flatten':
            output_delete_mask = np.reshape(input_delete_masks, [-1, ])
            output = layer(inputs)

        else:
            raise ValueError('"{0}" layers are currently '
                             'unsupported.'.format(layer_class))

        # if layer_class == 'MaxPool2D':
        #     delete_mask = delete_mask
    else:
        output_delete_mask = None

    return output, output_delete_mask


def rebuild(model, copy=True):
    if copy:
        model = clean_copy(model)
    model = clean_copy(model)
    model_inputs = model.inputs
    output_layers = model.output_layers
    output_layers_node_indices = model.output_layers_node_indices
    new_model_outputs, _ = rebuild_submodel(model_inputs,
                                            output_layers,
                                            output_layers_node_indices)
    new_model = Model(model_inputs, new_model_outputs)
    if copy:
        return clean_copy(new_model)
    else:
        return new_model


def rebuild_submodel(model_inputs, output_layers, output_layers_node_indices,
                     replace_inputs=None, finished_outputs=None):
    """Rebuild the model"""
    if not finished_outputs:
        finished_outputs = {}
    if not replace_inputs:
        replace_inputs = {}

    def _rebuild_rec(layer, node_index):
        """Rebuilds the instance of layer and all deeper layers recursively.
        
        Calculates the output tensor by applying this layer to this node.
        All tensors deeper in the network are also calculated
        
        Args:
            layer: the layer to rebuild
            node_index: The index of the next inbound node in the network. 
                        The layer will be called on the output of this node to 
                        obtain the output.
                        inbound_node = layer.inbound_nodes[node_index].
        Returns:
            The output of the layer on the data stream indicated by node_index.
                             
        """
        logging.debug('getting inputs for: {0}'.format(layer.name))
        # get the inbound node
        inbound_node = layer.inbound_nodes[node_index]
        if inbound_node in finished_outputs.keys():
            logging.debug('reached finished node: {0}'.format(inbound_node))
            return finished_outputs[inbound_node]

        inbound_layers = inbound_node.inbound_layers
        inbound_node_indices = inbound_node.node_indices
        # find this layer's inputs recursively.
        inputs = []
        logging.debug('inbound_layers: {0}'.format(inbound_layers))
        for inbound_layer, inbound_node_index in zip(inbound_layers,
                                                     inbound_node_indices):
            inbound_layer_output = \
                inbound_layer.get_output_at(inbound_node_index)
            if not inbound_layer:
                logging.debug('bottomed out to an unknown input')
                raise ValueError
            elif inbound_layer_output in model_inputs:
                logging.debug('bottomed out at a model input')
                inputs.append(inbound_layer_output)
            elif inbound_layer_output in replace_inputs.keys():
                logging.debug('bottomed out at replaced output: {0}'.format(inbound_layer_output))
                inputs.append(replace_inputs[inbound_layer_output])

            else:
                inputs.append(_rebuild_rec(inbound_layer, inbound_node_index))
        # call this layer on the outputs of the inbound layers
        if len(inputs) == 1:
            output = layer(inputs[0])
        else:
            output = layer(inputs)
        finished_outputs[inbound_node] = output
        return output
        # if the previous layer is empty, bottom out

    new_model_outputs = []
    for output_layer, output_layer_node_index in \
            zip(output_layers, output_layers_node_indices):
        new_model_outputs.append(_rebuild_rec(output_layer,
                                              output_layer_node_index))
    return new_model_outputs, finished_outputs


def check_for_layer_reuse(model):
    """Returns True if any layers are reused, False if not."""
    for layer in model.layers:
        if len(layer.inbound_nodes) > 1:
            return True
    return False


def clean_copy(model):
    """Returns a copy of the model without other model uses of its layers."""
    weights = model.get_weights()
    new_model = Model.from_config(model.get_config())
    new_model.set_weights(weights)
    return new_model
