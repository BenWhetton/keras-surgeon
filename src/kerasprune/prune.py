"""Prune connections or whole neurons from Keras model layers."""
# Imports
import logging

import numpy as np
from keras.models import Model

from kerasprune import utils

# Set up logging
logging.basicConfig(level=logging.INFO)


def rebuild_sequential(layers):
    """Rebuild a sequential model from a list of layers.

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


def rebuild_submodel(model_inputs,
                     output_layers,
                     output_layers_node_indices,
                     replace_tensors=None,
                     finished_nodes=None,
                     input_delete_masks=None):
    """Rebuild a subsection of a model given its inputs and outputs.

    Re-constructing subsections of a model enables layers or layer instances
    to be removed or modified and new layers or layer instances to be added.

    Arguments:
        model_inputs: List of the submodel's input tensor(s).
        output_layers: List of the submodel's output layers(s).
        output_layers_node_indices: List of indices of output layers' nodes.
        replace_tensors: Dict mapping model tensors to replacement tensors.
        finished_nodes: Dict mapping finished nodes to lists of their outputs
                        and output masks.
        input_delete_masks: Boolean mask for each input with size=input's size.

    Returns:
        list containing: submodel output tensors, output masks and updated
        finished_nodes dict

    """
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
            # Check for replaced tensors before checking finished nodes
            logging.debug('bottomed out at replaced output: {0}'.format(
                layer_output))
            output, output_mask = replace_tensors[layer_output]
            return output, output_mask

        elif node in finished_nodes.keys():
            logging.debug('reached finished node: {0}'.format(node))
            return finished_nodes[node]

        elif not layer:
            raise ValueError('The graph traversal has reached an empty layer.')

        elif layer_output in model_inputs:
            logging.debug('bottomed out at a model input')
            output_mask = input_delete_masks[model_inputs.index(layer_output)]
            return layer_output, output_mask

        else:
            # Recursively compute this layer's inputs and input masks from each
            # inbound layer at this node
            inbound_node_indices = node.node_indices
            logging.debug('inbound_layers: {0}'.format([layer.name for layer in
                                                        node.inbound_layers]))
            inputs, input_masks = zip(*[_rebuild_rec(l, i) for l, i in zip(
                node.inbound_layers, inbound_node_indices)])

            # Apply masks to the layer weights and call it on its inputs
            new_layer, output_mask = _apply_delete_mask(layer, node_index, input_masks)
            output = new_layer(utils.single_element(inputs))

            finished_nodes[node] = (output, output_mask)
            logging.debug('layer complete: {0}'.format(layer.name))
            return output, output_mask

    # Call the recursive _rebuild_rec method to rebuild the submodel up to each
    # output layer
    submodel_outputs, output_masks = zip(*[_rebuild_rec(l, i) for l, i in zip(
        output_layers, output_layers_node_indices)])
    return submodel_outputs, output_masks, finished_nodes


def _apply_delete_mask(layer, node_index, inbound_delete_masks):
    """Apply the inbound delete mask and return the outbound delete mask

    When specific channels in a layer or layer instance are deleted, the
    mask propagates information about which channels are affected to
    downstream layers.
    If the layer contains weights, the weights which were previously connected
    to the deleted channels are deleted and outbound masks are set to ones
    since further downstream layers aren't affected.
    """

    # if delete_mask is None, the deleted channels do not affect this layer or
    # any layers above/downstream from it
    if all(mask is None for mask in inbound_delete_masks):
        new_layer = layer
        outbound_delete_mask = None
    else:
        inbound_delete_masks = utils.single_element(inbound_delete_masks)
        # otherwise, delete_mask.shape should be: layer.input_shape[1:]
        layer_class = layer.__class__.__name__
        if layer_class == 'InputLayer':
            raise RuntimeError('This should never get here!')

        elif layer_class == 'Dense':
            output_shape = layer.get_output_shape_at(node_index)
            weights = layer.get_weights()
            weights[0] = weights[0][np.where(inbound_delete_masks)[0], :]
            config = layer.get_config()
            config['weights'] = weights
            new_layer = type(layer).from_config(config)
            outbound_delete_mask = np.ones(output_shape[1:], dtype=bool)

        elif layer_class == 'Flatten':
            outbound_delete_mask = np.reshape(inbound_delete_masks, [-1, ])
            new_layer = layer

        elif layer_class in ('Conv1D', 'Conv2D', 'Conv3D'):
            # Set outbound delete mask to ones.
            output_shape = layer.get_output_shape_at(node_index)
            outbound_delete_mask = np.ones(output_shape[1:], dtype=bool)
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
            new_shape = list(weights[0].shape)
            new_shape[-2] = -1  # weights data format is always channels_last
            weights[0] = np.reshape(weights[0][full_delete_mask], new_shape)
            # Instantiate new layer with new_weights
            config = layer.get_config()
            config['weights'] = weights
            new_layer = type(layer).from_config(config)

        elif layer_class in ('MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D'):
            output_shape = layer.get_output_shape_at(node_index)
            if layer.data_format == 'channels_first':
                index = [slice(None)] + [slice(None, dim_size, None) for
                                         dim_size in output_shape[2:]]
            elif layer.data_format == 'channels_last':
                index = [slice(None, dim_size, None) for dim_size in
                         output_shape[1:-1]] + [slice(None)]
            else:
                raise ValueError('Invalid data format')
            outbound_delete_mask = inbound_delete_masks[index]
            new_layer = layer

        elif layer_class in ('Dropout', 'Activation', 'SpatialDropout1D',
                             'SpatialDropout2D', 'SpatialDropout3D'):
            outbound_delete_mask = inbound_delete_masks
            new_layer = layer

        elif layer_class == 'Reshape':
            outbound_delete_mask = np.reshape(inbound_delete_masks,
                                              layer.target_shape)
            new_layer = layer

        elif layer_class == 'Permute':
            outbound_delete_mask = np.transpose(inbound_delete_masks,
                                                layer.dims)
            new_layer = layer

        else:
            raise ValueError('"{0}" layers are currently '
                             'unsupported.'.format(layer_class))

    return new_layer, outbound_delete_mask


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
        # Call the new layer on the inbound layer's output
        new_output = new_layer(utils.single_element(inputs))
        # Replace the inbound layer's output with the new layer's output
        node = this_layer.inbound_nodes[node_index]
        old_output = node.inbound_layers[0].get_output_at(node.node_indices[0])
        replace_tensor = {old_output: (new_output, None)}
        return replace_tensor

    # The same core logic is used for all layer manipulation functions
    new_model = modify_model(model,
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
        new_output = new_layer(utils.single_element(inputs))

        # Replace the original layer's output with the new layer's output
        replaced_layer_output = this_layer.get_output_at(node_index)
        replace_inputs = {replaced_layer_output: (new_output, None)}
        return replace_inputs

    # The same core logic is used for all layer manipulation functions
    new_model = modify_model(model,
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
        inputs = utils.single_element(inputs)
        deleted_layer_output = this_layer.get_output_at(node_index)
        replace_inputs = {deleted_layer_output: (inputs, None)}
        return replace_inputs

    # The same core logic is used for all layer manipulation functions
    new_model = modify_model(model,
                             layer,
                             _delete_layer,
                             node_indices,
                             copy,
                             input_delete_masks=None)
    return new_model


def delete_channels(model, layer, channels, node_indices=None, copy=None):
    """Delete channels from the specified layer.

    This method is designed to facilitate research into pruning networks to
    improve their prediction performance and/or reduce computational load by
    deleting channels.
    All weights associated with the deleted channels in the specified layer
    and downstream layers are deleted.
    If the layer is shared and node_indices is set, channels will be deleted
    from the corresponding layer instances only. This will break the weight
    sharing between pruned and un-pruned instances in subsequent training.
    Channels correspond to filters in conv layers and units in other layers.

    Args:
        model: Model object.
        layer: Layer whose channels are to be deleted.
        channels: Indices of the channels to be deleted
        node_indices: Indices of the nodes where channels are to be deleted.
        copy: If True, the model will be copied before and after
              manipulation. This keeps both the old and new models' layers
              clean of each-others data-streams.

    Returns:
        A new Model with the specified channels and associated weights deleted.

    Raises:
        blaError: if layer is not contained by model
        ValueError: if new_layer is not compatible with the input and output
                    dimensions of the layers preceding and following it.
        ValueError: if node_index does not correspond to one of layer's inbound
                    nodes.
    """

    # Delete the channels in layer to create new_layer
    new_layer = _delete_channel_weights(layer, channels)
    # Create the mask to determine the weights to delete in downstream layers
    new_delete_mask = _make_delete_mask(layer, channels)
    # Initialise the delete masks for the model input as all True.
    input_delete_masks = [np.ones(node.outbound_layer.input_shape[1:],
                                  dtype=bool) for node in model.inbound_nodes]

    # Define the function to be applied to the inputs to the layer at each node
    def _delete_inbound_weights(this_layer, node_index, inputs):
        # Call the new layer on the rebuild submodel's inputs
        new_output = new_layer(utils.single_element(inputs))
        # Replace the original layer's output with the modified layer's output
        deleted_layer_output = this_layer.get_output_at(node_index)
        replace_inputs = {deleted_layer_output: (new_output, new_delete_mask)}
        return replace_inputs

    # Apply the modifications to the specified layer instances in the model
    new_model = modify_model(model,
                             layer,
                             _delete_inbound_weights,
                             node_indices,
                             copy,
                             input_delete_masks)
    return new_model


def modify_model(model,
                 layer,
                 modifier_function,
                 node_indices=None,
                 copy=None,
                 input_delete_masks=None):
    """Helper function to modify a model around instances of a specified layer.

    This method applies a modifier function to each node specified by the
    combination of layer and node_indices.
    The modifier function performs some operations and returns a mapping of
    original tensors to the resulting replacement tensors.
    See uses of this method as examples.

    Arguments:
        model: Model object to be modified.
        layer: Layer to be modified.
        modifier_function: Function to be applied to each specified node.
        node_indices: Indices of the nodes to be modified.
        copy: If True, the model will be copied before and after
              manipulation. This keeps both the old and new models' layers
              clean of each-others data-streams.
        input_delete_masks: Boolean masks to specify input channels (used by
                            delete_channels).

    Returns:
        A modified model object

    """
    # Check inputs
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')
    # If no nodes are specified, apply the modification to all of the layer's
    # inbound nodes which are contained in the model.
    if not node_indices:
        node_indices = utils.bool_to_index(
            utils.check_nodes_in_model(model, layer.inbound_nodes))
    if copy:
        model = utils.clean_copy(model)
        layer = model.get_layer(layer.get_config()['name'])

    # Order the nodes by depth from input to output to ensure that the model is
    # rebuilt in the correct order.
    node_depths = [utils.get_node_depth(model, layer.inbound_nodes[node_index])
                   for node_index in node_indices]
    sorted_indices = reversed(utils.sort_x_by_y(node_indices, node_depths))

    replace_tensors = {}
    finished_nodes = {}
    # For each node i.e. layer instance
    for node_index in sorted_indices:
        # Rebuild the model up to layer instance
        inbound_node = layer.inbound_nodes[node_index]
        submodel_output_layers = inbound_node.inbound_layers
        submodel_output_layer_node_indices = inbound_node.node_indices

        logging.debug('rebuilding model up to the layer before the insertion: '
                      '{0}'.format(layer))
        (submodel_outputs, _, submodel_finished_outputs) = rebuild_submodel(
            model.inputs,
            submodel_output_layers,
            submodel_output_layer_node_indices,
            replace_tensors,
            finished_nodes,
            input_delete_masks)
        finished_nodes.update(submodel_finished_outputs)

        # Apply the modifier function.
        replace_tensors.update(modifier_function(layer,
                                                 node_index,
                                                 submodel_outputs))

    # Rebuild the rest of the model from the modified nodes to the outputs.
    new_outputs, _, _ = rebuild_submodel(model.inputs,
                                         model.output_layers,
                                         model.output_layers_node_indices,
                                         replace_tensors,
                                         finished_nodes,
                                         input_delete_masks)
    new_model = Model(model.inputs, new_outputs)
    if copy:
        return utils.clean_copy(new_model)
    else:
        return new_model


def _make_delete_mask(layer, channel_indices):
    """Make the boolean delete mask for layer's output deleting channels.
    The mask is used to index the weights of the following layers to remove
    weights previously linked to channels which have been deleted.

    Arguments:
        layer: A Keras layer
        channel_indices: the indices of the channels to be deleted

    Returns:
        A Numpy ndarray of booleans of the same size as the output of layer.

    """
    layer_config = layer.get_config()
    if ('data_format' in layer_config.keys()) and \
            (layer_config['data_format'] == 'channels_first'):
        output_channels_axis = 0
    else:
        output_channels_axis = -1
    new_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)
    index = [slice(None)] * new_delete_mask.ndim
    index[output_channels_axis] = channel_indices
    new_delete_mask[tuple(index)] = False
    return new_delete_mask


def _delete_channel_weights(layer, channel_indices):
    """Delete channels from layer and remove un-used weights.

    Arguments:
        layer: A Keras layer
        channel_indices: the indices of the channels to be deleted.

    Returns:
        A new layer with the channels and corresponding weights deleted.
    """
    layer_config = layer.get_config()
    if 'units' in layer_config.keys():
        channels_string = 'units'
    elif 'filters' in layer_config.keys():
        channels_string = 'filters'
    else:
        raise ValueError(
            'The layer must have either a "units" or "filters" '
            'property to be able to delete channels.')

    if any([i + 1 > layer_config[channels_string] for i in channel_indices]):
        raise ValueError('Channels_index value(s) out of range. '
                         'This layer only has {0} channels.'
                         .format(layer_config[channels_string]))

    layer_config[channels_string] -= len(channel_indices)

    weights = layer.get_weights()
    weights = [np.delete(w, channel_indices, axis=-1) for w in weights]
    layer_config['weights'] = weights

    # create new layer from modified config
    return type(layer).from_config(layer_config)
