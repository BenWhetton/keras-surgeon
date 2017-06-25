"""Prune connections or whole neurons from Keras model layers."""

import numpy as np
import logging
from keras.models import Model
# TODO: Add conditional statements for layer option use_bias=False.
logging.basicConfig(level=logging.INFO)


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
                     replace_tensors=None,
                     finished_nodes=None,
                     input_delete_masks=None):
    """Rebuild the model"""
    if not input_delete_masks:
        input_delete_masks = [None] * len(model_inputs)
    if not finished_nodes:
        finished_nodes = {}
    if not replace_tensors:
        replace_tensors = {}

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
        node = layer.inbound_nodes[node_index]
        layer_output = layer.get_output_at(node_index)
        if layer_output in replace_tensors.keys():
            # Check for replaced tensors first to ensure that they are not
            # overridden by finished nodes
            logging.debug('bottomed out at replaced output: {0}'.format(
                layer_output))
            output, outbound_mask = replace_tensors[layer_output]
            return output, outbound_mask

        elif node in finished_nodes.keys():
            logging.debug('reached finished node: {0}'.format(node))
            return finished_nodes[node]

        elif not layer:
            logging.debug('bottomed out to an unknown input')
            raise ValueError

        elif layer_output in model_inputs:
            logging.debug('bottomed out at a model input')
            outbound_mask = input_delete_masks[
                model_inputs.index(layer_output)]
            return layer_output, outbound_mask

        else:
            inbound_layers = node.inbound_layers
            inbound_node_indices = node.node_indices
            # Recursively find this layer's inputs and input masks from each
            # inbound layer at this inbound node
            logging.debug('inbound_layers: {0}'.format([layer.name for layer in
                                                        inbound_layers]))
            inputs, input_masks = zip(*[_rebuild_rec(l, i) for l, i in zip(
                inbound_layers, inbound_node_indices)])
            # Apply the delete masks to this layer
            new_layer, outbound_mask = _apply_delete_mask(layer, input_masks)
            # Call this layer on its inputs at the inbound node
            output = new_layer(extract_if_single_element(inputs))
            # Add this node's outputs to the finished outputs lists
            finished_nodes[node] = (output, outbound_mask)
            logging.debug('layer complete: {0}'.format(layer.name))
            return output, outbound_mask

    submodel_outputs, output_masks = zip(*[_rebuild_rec(l, i) for l, i in
                                           zip(
        output_layers, output_layers_node_indices)])
    return submodel_outputs, output_masks, finished_nodes


def _apply_delete_mask(layer, inbound_delete_masks):
    """Apply the inbound delete mask and return the outbound delete mask"""
    # if delete_mask is None, the deleted channels do not affect this layer or
    # any layers above it
    if all(mask is None for mask in inbound_delete_masks):
        new_layer = layer
        outbound_delete_mask = None
    else:
        inbound_delete_masks = extract_if_single_element(inbound_delete_masks)
        # otherwise, delete_mask.shape should be: layer.input_shape[1:]
        layer_class = layer.__class__.__name__
        if layer_class == 'InputLayer':
            raise RuntimeError('This should never get here!')

        elif layer_class == 'Dense':
            weights = layer.get_weights()
            weights[0] = weights[0][np.where(inbound_delete_masks)[0], :]
            config = layer.get_config()
            config['weights'] = weights
            new_layer = type(layer).from_config(config)
            outbound_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)

        elif layer_class == 'Flatten':
            outbound_delete_mask = np.reshape(inbound_delete_masks, [-1, ])
            new_layer = layer

        elif layer_class == 'Conv2D':
            # outbound delete mask set to ones
            # no downstream layers are affected
            outbound_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)
            # Conv layer: trim down inbound_delete_masks to filter shape
            k_size = layer.kernel_size
            if layer.data_format == 'channels_first':
                inbound_delete_masks = np.swapaxes(inbound_delete_masks, 0, -1)
            index = [slice(None, dim_size, None) for dim_size in k_size]
            inbound_delete_masks = inbound_delete_masks[index + [slice(None)]]
            # Delete unused weights to obtain new_weights
            weights = layer.get_weights()
            # The mask size is equal to the
            full_delete_mask = np.repeat(inbound_delete_masks[..., np.newaxis],
                                         weights[0].shape[-1],
                                         axis=-1)
            weights = weights
            new_shape = list(weights[0].shape)
            new_shape[-2] = -1
            weights_pruned = weights[0][full_delete_mask]
            weights_reshaped = np.reshape(weights_pruned, new_shape)
            weights[0] = weights_reshaped
            # Instantiate new layer with new_weights
            config = layer.get_config()
            config['weights'] = weights
            new_layer = type(layer).from_config(config)

        else:
            raise ValueError('"{0}" layers are currently '
                             'unsupported.'.format(layer_class))

        # if layer_class == 'MaxPool2D':
        #     delete_mask = delete_mask

    return new_layer, outbound_delete_mask


def check_for_layer_reuse(model, layers=None):
    """Returns True if any layers are reused, False if not."""
    if layers is None:
        layers = model.layers
    return any([len(l.inbound_nodes) > 1 for l in layers])
    # for layer in layers:
    #     if len(layer.inbound_nodes) > 1:
    #         return True
    # return False


def clean_copy(model):
    """Returns a copy of the model without other model uses of its layers."""
    weights = model.get_weights()
    new_model = Model.from_config(model.get_config())
    new_model.set_weights(weights)
    return new_model


def get_node_depth(model, node):
    """Get the depth of a node in a model.

    Arguments:
        model: Keras Model object
        node: Keras Node object

    Returns:
        The node depth as an integer. The model outputs are at depth 0.

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
        new_layer: A layer to be inserted into model before layer.
        node_indices: the indices of the inbound_node to layer where the
                      new layer is to be inserted.
        copy: If True, the model will be copied before and after
              manipulation. This keeps both the old and new models' layers
              clean of each-others data-streams.

    Returns:
        a new Keras Model object with layer inserted.

    Raises:
        blaError: if layer is not contained by model
        ValueError: if new_layer is not compatible with the input and output
                    dimensions of the layers preceding and following it.
        ValueError: if node_index does not correspond to one of layer's inbound
                    nodes.
    """
    # No setup required

    # Define the function to be applied to the inputs to the layer at each node
    def _insert_layer(this_layer, node_index, inputs):
        # This will not work for nodes with multiple inbound layers
        # The previous layer and node must also be specified to enable this
        # functionality.
        if len(inputs) >= 2:
            raise ValueError('Cannot insert new layer at node with multiple '
                             'inbound layers.')
        inputs = extract_if_single_element(inputs)
        # Call the new layer on the inbound layer's output
        new_output = new_layer(inputs)
        # Replace the inbound layer's output with the new layer's output
        inbound_layer_index = this_layer.inbound_nodes[node_index].node_indices[0]
        inbound_layer = this_layer.inbound_nodes[node_index].inbound_layers[0]
        old_output = inbound_layer.get_output_at(inbound_layer_index)
        replace_tensor = {old_output: (new_output, None)}
        return replace_tensor

    # The same core logic is used for all layer manipulation functions
    new_model = reused_core_logic(model,
                                  layer,
                                  _insert_layer,
                                  node_indices,
                                  copy)
    return new_model


def replace_layer(model, layer, new_layer, node_indices=None, copy=True):
    # No setup required

    # Define the function to be applied to the inputs to the layer at each node
    def _replace_layer(this_layer, node_index, inputs):
        # Call the new layer on the rebuild submodel's inputs
        inputs = extract_if_single_element(inputs)
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
    """Delete one or more instances of a layer from a Keras model.

    Args:
        model: Keras Model object.
        layer: Keras Layer object contained in model.
        node_indices: The indices of the inbound_node to the layer instances to
                      be deleted.
        copy: If True, the model will be copied before and after
              manipulation. This keeps both the old and new models' layers
              clean of each-others data-streams.

    Returns:
        Keras Model object with the layer at node_index deleted.
    """
    # No setup required

    # Define the function to be applied to the inputs to the layer at each node
    def _delete_layer(this_layer, node_index, inputs):
        # Skip the deleted layer by replacing its outputs with it inputs
        if len(inputs) >= 2:
            raise ValueError('Cannot insert new layer at node with multiple '
                             'inbound layers.')
        inputs = extract_if_single_element(inputs)
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
    new_layer = delete_channel_weights(layer, channels_index)

    # Create the mask for determining the weights to delete in shallower layers
    new_delete_mask = make_delete_mask(layer, channels_index)

    # initialise the delete masks for the model input
    input_delete_masks = [np.ones(node.outbound_layer.input_shape[1:],
                                  dtype=bool) for node in model.inbound_nodes]

    # define the function to be applied to the inputs to the layer at each node
    def _delete_inbound_weights(this_layer, node_index, inputs):
        # Call the new layer on the rebuild submodel's inputs
        inputs = extract_if_single_element(inputs)
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
        # TODO: Check if these exceptions need to be raised
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
    replace_tensors = {}
    finished_nodes = {}
    node_depths = [get_node_depth(model, layer.inbound_nodes[node_index])
                   for node_index in node_indices]
    sorted_indices = sort_x_by_y(node_indices, node_depths)
    for node_index in sorted_indices:
        # Rebuild the model up to layer instance
        inbound_node = layer.inbound_nodes[node_index]
        submodel_output_layers = inbound_node.inbound_layers  # TODO: Consider removing these variables, here for readability purposes only
        submodel_output_layer_node_indices = inbound_node.node_indices

        logging.debug('rebuilding model up to the layer before the insertion: '
                      '{0}'.format(layer))
        (submodel_outputs,
         _,  # output_delete_masks,
         submodel_finished_outputs) = rebuild_submodel(model.inputs,
                                                       output_layers=submodel_output_layers,
                                                       output_layers_node_indices=submodel_output_layer_node_indices,
                                                       replace_tensors=replace_tensors,
                                                       finished_nodes=finished_nodes,
                                                       input_delete_masks=input_delete_masks)
        finished_nodes.update(submodel_finished_outputs)

        # modify the chosen layer in some manner
        replace_tensors.update(modifier_function(layer,
                                                 node_index,
                                                 submodel_outputs)
                               )

    # Rebuild the rest of the model
    new_outputs, _, _ = rebuild_submodel(model.inputs,
                                         model.output_layers,
                                         model.output_layers_node_indices,
                                         replace_tensors,
                                         finished_nodes,
                                         input_delete_masks)
    new_model = Model(model.inputs, new_outputs)
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


def make_delete_mask(layer, channels_index):
    layer_config = layer.get_config()
    if ('data_format' in layer_config.keys()) and (layer_config['data_format'] == 'channels_first'):
        output_channels_axis = 0
    else:
        output_channels_axis = -1
    new_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)
    index = [slice(None)] * new_delete_mask.ndim
    index[output_channels_axis] = channels_index
    new_delete_mask[tuple(index)] = False
    return new_delete_mask


def delete_channel_weights(layer, channels_index):
    """Delete the weights corresponding to the removed neurons."""
    weights = layer.get_weights()
    new_weights = [np.delete(weights[0], channels_index, axis=-1),
                   np.delete(weights[1], channels_index, axis=0)]
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
    return new_layer
