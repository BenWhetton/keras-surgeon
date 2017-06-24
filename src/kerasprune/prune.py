"""Prune connections or whole neurons from Keras model layers."""

import numpy as np
import logging
from keras.models import Model
# TODO: Add conditional statements for layer option use_bias=False.
logging.basicConfig(level=logging.INFO)


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
    layer_class = layer.__class__.__name__
    if layer_class == 'Conv2D':
        if layer.get_config()['data_format'] == 'channels_first':
            input_channels_axis = 0
            output_channels_axis = 1
        else:
            input_channels_axis = -2
            output_channels_axis = -1

    elif layer_class == 'Dense':
        input_channels_axis = 0
        output_channels_axis = -1

    elif layer_class in ('InputLayer', 'Flatten'):
        input_channels_axis = None
        output_channels_axis = None

    else:
        raise TypeError('This function has only been implemented for '
                        'convolutional and dense layers.')

    return [input_channels_axis, output_channels_axis]


def rebuild_sequential(layers):
    """Rebuild a sequential model from a list if layers preserving the weights

    Arguments:
        layers: List of Keras layers
    Returns:
        A Keras Sequential model
    """
    from keras.models import Sequential

    weights = []
    for layer in layers:
        weights.append(layer.get_weights())

    new_model = Sequential(layers=layers)
    for i, layer in enumerate(new_model.layers):
        layer.set_weights(weights[i])
    return new_model


# TODO: account for cases where multiple outputs use instances of same layer
def rebuild_submodel(model_inputs,
                     output_layers,
                     output_layers_node_indices,
                     replace_inputs=None,
                     finished_outputs=None,
                     input_delete_masks=None):
    """Rebuild the model"""
    if not input_delete_masks:
        input_delete_masks = [None] * len(model_inputs)
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
        layer_output = layer.get_output_at(node_index)
        if inbound_node in finished_outputs.keys():
            logging.debug('reached finished node: {0}'.format(inbound_node))
            return finished_outputs[inbound_node]

        elif not layer:
            logging.debug('bottomed out to an unknown input')
            raise ValueError

        elif layer_output in model_inputs:
            logging.debug('bottomed out at a model input')
            outbound_mask = input_delete_masks[
                model_inputs.index(layer_output)]
            return layer_output, outbound_mask

        elif layer_output in replace_inputs.keys():
            logging.debug('bottomed out at replaced output: {0}'.format(
                layer_output))
            output, outbound_mask = replace_inputs[layer_output]
            return output, outbound_mask

        else:
            inbound_layers = inbound_node.inbound_layers
            inbound_node_indices = inbound_node.node_indices
            # Recursively find this layer's inputs and inbound masks from each
            # inbound layer at this inbound node
            inputs = []
            delete_masks = []
            logging.debug('inbound_layers: {0}'.format([layer.name for layer in inbound_layers]))
            for inbound_layer, inbound_node_index in zip(inbound_layers,
                                                         inbound_node_indices):
                output, outbound_mask = _rebuild_rec(inbound_layer,
                                                     inbound_node_index)
                inputs.append(output)
                delete_masks.append(outbound_mask)
            # Apply the delete masks to this layer
            new_layer, outbound_mask = _apply_delete_mask(layer, delete_masks)
            # Call this layer on its inputs at the inbound node
            if len(inputs) == 1:
                inputs = inputs[0]
            output = new_layer(inputs)
            # Add this inbound_node's outputs to the finished outputs lists
            finished_outputs[inbound_node] = (output, outbound_mask)
            logging.debug('layer complete: {0}'.format(layer.name))
            return output, outbound_mask

    new_submodel_outputs = []
    output_delete_masks = []
    for output_layer, output_layer_node_index in \
            zip(output_layers, output_layers_node_indices):
        submodel_output, delete_mask = _rebuild_rec(output_layer,
                                                    output_layer_node_index)
        new_submodel_outputs.append(submodel_output)
        output_delete_masks.append(delete_mask)
    return new_submodel_outputs, output_delete_masks, finished_outputs


def _apply_delete_mask(layer, inbound_delete_masks):
    """Apply the inbound delete mask and return the outbound delete mask"""
    # if delete_mask is None, the deleted channels do not affect this layer or
    # any layers above it
    if all(mask is None for mask in inbound_delete_masks):
        new_layer = layer
        outbound_delete_mask = None
    else:
        if len(inbound_delete_masks) == 1:
            inbound_delete_masks = inbound_delete_masks[0]
        # otherwise, delete_mask.shape should be: layer.input_shape[1:]
        layer_class = layer.__class__.__name__
        if layer_class == 'InputLayer':
            raise RuntimeError('This should never get here!')

        elif layer_class == 'Dense':
            weights = layer.get_weights()
            new_weights = weights
            new_weights[0] = new_weights[0][np.where(inbound_delete_masks == True)[0], :]  # TODO: Fix this rubbish
            config = layer.get_config()
            config['weights'] = new_weights
            new_layer = type(layer).from_config(config)
            outbound_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)  # TODO: can this be None? Think about layer reuse in series.

        elif layer_class == 'Flatten':
            outbound_delete_mask = np.reshape(inbound_delete_masks, [-1, ])
            new_layer = layer

        elif layer_class == 'Conv2D':
            [channels_axis, _] = _get_channels_axis(layer)
            # outbound delete mask set to ones
            # no downstream layers are affected
            outbound_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)
            # Conv layer: trim down inbound_delete_masks to filter shape
            k_size = layer.kernel_size
            if layer.data_format == 'channels_first':
                inbound_delete_masks = inbound_delete_masks[:,
                                                            :k_size[0],
                                                            :k_size[1]]
            elif layer.data_format == 'channels_last':
                inbound_delete_masks = inbound_delete_masks[:k_size[0],
                                                            :k_size[1],
                                                            :]
            # Delete unused weights to obtain new_weights
            weights = layer.get_weights()
            full_delete_mask = np.repeat(
                np.expand_dims(inbound_delete_masks, axis=3),
                weights[0].shape[3], axis=3)
            new_weights = weights
            new_shape = list(new_weights[0].shape)
            new_shape[channels_axis] = -1
            weights_pruned = new_weights[0][full_delete_mask]
            weights_reshaped = np.reshape(weights_pruned, new_shape)
            new_weights[0] = weights_reshaped
            # Instantiate new layer with new_weights
            config = layer.get_config()
            config['weights'] = new_weights
            new_layer = type(layer).from_config(config)

        else:
            raise ValueError('"{0}" layers are currently '
                             'unsupported.'.format(layer_class))

        # if layer_class == 'MaxPool2D':
        #     delete_mask = delete_mask

    return new_layer, outbound_delete_mask


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


def get_node_depth(model, node):
    """Get the depth of the node in the model.

    Arguments:
        model: Keras Model object
        node: Keras Node object

    Returns:
        The node depth as an integer. 0 is the output depth.

    Raises:
        KeyError: if the node is not contained in the model.
    """
    for (depth, nodes_at_depth) in model.nodes_by_depth.items():
        if node in nodes_at_depth:
            return depth
    raise KeyError('The node is not contained in the model.')


def insert_layer(model, layer, new_layer, node_indices=None, copy=True):
    """Insert new_layer before layer at node_indices.

        If node_index must be specified if there is more than one inbound node.

        Args:
            model: Keras Model object.
            layer: Keras Layer object contained in model.
            new_layer: a layer to be inserted into model before layer.
            node_indices: the indices of the inbound_node to layer where new layer is
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
    # No setup required

    # Define the function to be applied to the inputs to the layer at each node
    def _insert_layer(this_layer, node_index, inputs):
        inbound_node = this_layer.inbound_nodes[node_index]
        inbound_layers = inbound_node.inbound_layers
        inbound_node_indices = inbound_node.node_indices
        replace_inputs = {}
        # TODO: This should only be for one node???
        for inbound_layer, index, prev_output in zip(inbound_layers,
                                                     inbound_node_indices,
                                                     inputs):
            # Call the new layer on each inbound layer's outputs
            new_output = new_layer(prev_output)
            # inserted_layer_outputs.append(output)
            # Replace the original inbound layer's output with the new layer's
            # output
            replace_inputs[inbound_layer.get_output_at(index)] = (new_output,
                                                                  None)
        return replace_inputs

    # The same core logic is used for all layer manipulation functions
    new_model = reused_core_logic(model,
                                  layer,
                                  _insert_layer,
                                  node_indices,
                                  copy,
                                  input_delete_masks=None)
    return new_model


def replace_layer(model, layer, new_layer, node_indices=None, copy=True):
    # No setup required

    # Define the function to be applied to the inputs to the layer at each node
    def _replace_layer(this_layer, node_index, inputs):
        # Call the new layer on the rebuild submodel's inputs
        new_output = new_layer(inputs)

        # Replace the original layer's output with the new layer's output
        replaced_layer_output = this_layer.get_output_at(node_index)
        replace_inputs = {replaced_layer_output: (new_output, None)}
        return replace_inputs

    # The same core logic is used for all layer manipulation functions
    new_model = reused_core_logic(model,
                                  layer,
                                  _replace_layer,
                                  node_indices,
                                  copy,
                                  input_delete_masks=None)
    return new_model


def delete_layer(model, layer, node_indices=None, copy=True):
    """Delete an instance of a layer from a Keras model.

        Args:
            model: Keras Model object.
            layer: Keras Layer object contained in model.
            node_index: the index of the inbound_node to the layer to be deleted.

        Returns:
            Keras Model object with the layer at node_index deleted.
    """
    # No setup required

    # Define the function to be applied to the inputs to the layer at each node
    def _delete_layer(this_layer, node_index, inputs):
        # Skip the deleted layer by replacing its outputs with it inputs
        deleted_layer_output = this_layer.get_output_at(node_index)
        replace_inputs = {deleted_layer_output: (inputs, None)}
        return replace_inputs

    # The same core logic is used for all layer manipulation functions
    new_model = reused_core_logic(model,
                                  layer,
                                  _delete_layer,
                                  node_indices,
                                  copy,
                                  input_delete_masks=None)
    return new_model


def delete_channels(model, layer, channels_index, node_indices=None, copy=None):
    # Delete the channels in layer to create new_layer
    [_, output_axis] = _get_channels_axis(layer)
    new_weights = _delete_output_weights(layer, channels_index,
                                         output_axis)
    layer_config = layer.get_config()
    if 'units' in layer_config.keys():
        channels_string = 'units'
    elif 'filters' in layer_config.keys():
        channels_string = 'filters'
    else:
        raise ValueError(
            'The layer must have either a "units" or "filters" '
            'property to be able to delete channels.')

    if any([index + 1 > layer_config[channels_string] for index in
            channels_index]):
        raise ValueError('Channels index value(s) are out of range. '
                         'This layer only has {0} units'
                         .format(layer_config[channels_string]))
    layer_config[channels_string] -= len(channels_index)
    layer_config['weights'] = new_weights
    new_layer = type(layer).from_config(layer_config)

    # Create the mask for determining the weights to delete in shallower layers
    new_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)
    index = [slice(None)] * new_delete_mask.ndim
    index[output_axis] = channels_index
    new_delete_mask[tuple(index)] = False

    # initialise the delete masks for the model input
    input_delete_masks = [np.ones(node.outbound_layer.input_shape[1:],
                                  dtype=bool) for node in model.inbound_nodes]

    # define the function to be applied to the inputs to the layer at each node
    def _delete_inbound_weights(this_layer, node_index, inputs):
        # Call the new layer on the rebuild submodel's inputs
        new_output = new_layer(inputs)

        # Replace the original layer's output with the modified layer's output
        deleted_layer_output = this_layer.get_output_at(node_index)
        replace_inputs = {deleted_layer_output: (new_output, new_delete_mask)}
        return replace_inputs

    # The same core logic is used for all layer manipulation functions
    new_model = reused_core_logic(model,
                                  layer,
                                  _delete_inbound_weights,
                                  node_indices,
                                  copy,
                                  input_delete_masks)
    return new_model


def reused_core_logic(model, layer, modifier_function, node_indices=None, copy=None, input_delete_masks=None):
    """Helper function to modify a model around some or all instances of a specific layer

    """
    # Check inputs
    if layer not in model.layers:
        raise ValueError('layer is not a valid layer in model.')
    if check_for_layer_reuse(model):
        # if not node_indices:
        #     raise ValueError('A node_index must be specified if any layers in '
        #                      'the model are re-used within the model or in '
        #                      'other models.')
        if copy:
            raise ValueError('The model cannot be cleanly copied if any '
                             'layers in the model are re-used within the '
                             'model or in other models. Set copy=False.')

    if not node_indices:
        node_indices = range(len(layer.inbound_nodes))
    if copy:
        model = clean_copy(model)
        layer = model.get_layer(layer.get_config()['name'])

    # For each node associated with the layer,
    # rebuild the model up to the layer
    # then apply the modifier function.
    # The modifier function will modify some aspect of the model at the chosen layer and return replace_inputs a dictionary of keyed with tensors from the original model (some layer inputs) to be replaced by new tensors: the corresponding values of "replace_inputs"
    replace_inputs = {}
    finished_outputs = {}
    node_depths = [get_node_depth(model, layer.inbound_nodes[node_index])
                   for node_index in node_indices]
    sorted_indices = sort_x_by_y(node_indices, node_depths)
    for node_index in sorted_indices:
        # Rebuild the model up to layer instance
        inbound_node = layer.inbound_nodes[node_index]
        submodel_output_layers = inbound_node.inbound_layers
        submodel_output_layer_node_indices = inbound_node.node_indices

        logging.debug('rebuilding model up to the layer before the insertion: '
                      '{0}'.format(layer))
        (submodel_outputs,
         _, # output_delete_masks,
         submodel_finished_outputs) = rebuild_submodel(model.inputs,
                                                       output_layers=submodel_output_layers,
                                                       output_layers_node_indices=submodel_output_layer_node_indices,
                                                       replace_inputs=replace_inputs,
                                                       finished_outputs=finished_outputs,
                                                       input_delete_masks=input_delete_masks)

        finished_outputs.update(submodel_finished_outputs)
        # add the new layer to the outputs of each inbound layers
        submodel_outputs = extract_if_single_element(submodel_outputs)

        replace_inputs.update(modifier_function(layer, node_index, submodel_outputs))

    # Rebuild the rest of the model
    new_model_outputs, _, _ = rebuild_submodel(model.inputs,
                                               model.output_layers,
                                               model.output_layers_node_indices,
                                               replace_inputs,
                                               finished_outputs,
                                               input_delete_masks)
    new_model = Model(model.inputs, new_model_outputs)
    if copy:
        return clean_copy(new_model)
    else:
        return new_model


def sort_x_by_y(x, y):
    x = [x for (_, x) in sorted(zip(y, x), reverse=True)]
    return x


def extract_if_single_element(x):
    if len(x) == 1:
        x = x[0]
    return x
