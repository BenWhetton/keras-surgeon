"""Prune connections or whole neurons from Keras model layers."""
# Imports
import logging

import numpy as np
from keras import layers
from keras.models import Model
from keras.engine.topology import Node

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
    surgeon = Surgeon(model, copy)
    surgeon.add_job('insert_layer', layer, node_indices=node_indices, new_layer=new_layer)
    return surgeon.operate()


def replace_layer(model, layer, new_layer, node_indices=None, copy=True):
    surgeon = Surgeon(model, copy)
    surgeon.add_job('replace_layer', layer, node_indices=node_indices, new_layer=new_layer)
    return surgeon.operate()


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
    surgeon = Surgeon(model, copy)
    surgeon.add_job('delete_layer', layer, node_indices=node_indices)
    return surgeon.operate()


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
    surgeon = Surgeon(model, copy)
    surgeon.add_job('delete_channels', layer, node_indices=node_indices, channels=channels)
    return surgeon.operate()


class Surgeon:
    """Performs network surgery on a model.

    Surgeons can perform multiple network surgeries (jobs) at once. This is much faster than performing them sequenctially.


    """
    def __init__(self, model, copy=None):
        if copy:
            self.model = utils.clean_copy(model)
        else:
            self.model = model
        self._copy = copy
        self._finished_nodes = {}
        self._replace_tensors = {}
        self._channels_map = {}
        self._new_layers_map = {}
        self._insert_layers_map = {}
        self._replace_layers_map = {}
        self.nodes = []
        self._mod_func_map = {}
        self._kwargs_map = {}
        self.valid_jobs = ('delete_layer',
                           'insert_layer',
                           'replace_layer',
                           'delete_channels')

    def add_job(self, job, layer, channels=None, new_layer=None, node_indices=None):
        """Adds a job for the Surgeon to perform on the model.

        Job options are:
        'delete_layer': delete `layer` from the model
        'insert_layer': insert `new_layer` before `layer`
        'replace_layer': replace `layer` with `new_layer`
        'delete_channels' delete `channels` from `layer`

        This enables the Surgeon to perform many modifications at once in a
        single operation.
        Jobs can be added in any order.
        This is much faster than performing many modifications sequentially.

        Args:
            job(string): job identifier. One of `Surgeon.valid_jobs`.
            layer(Layer): A layer from `model` to be modified.
            channels(list[int]): A list of channels affected by the job.
                                 Used in `delete_channels`.
            new_layer(Layer): A new layer used for the job. Used in
                              `insert_layer` and `replace_layer`.
            node_indices(list[int]): (optional) A list of node indices used to
                                    selectively apply the job to a subset of
                                    the layer's nodes. Nodes are selected with:
                                    node[i] = layer.inbound_nodes[node_indices[i]]
        """
        if self._copy:
            layer = self.model.get_layer(layer.name)
        # Check inputs
        if layer not in self.model.layers:
            raise ValueError('layer is not a valid Layer in model.')
        # If no nodes are specified, apply the modification to all of the
        # layer's inbound nodes which are contained in the model.
        layer_node_indices = utils.find_nodes_in_model(self.model, layer)
        if not node_indices:
            node_indices = layer_node_indices
        elif not set(node_indices).issubset(layer_node_indices):
            raise ValueError('One or more nodes specified by `layer` and '
                             '`node_indices` are not in `model`.')

        # Select the modification function and any keyword arguments.
        kwargs = {}
        if job == 'delete_channels':
            if set(node_indices) != set(layer_node_indices):
                kwargs['layer_name'] = layer.name + '_' + job
            kwargs['channels'] = channels
            mod_func = self._delete_channels

        elif job == 'delete_layer':
            mod_func = self._delete_layer

        elif job == 'insert_layer':
            kwargs['new_layer'] = new_layer
            mod_func = self._insert_layer

        elif job == 'replace_layer':
            if set(node_indices) != set(layer_node_indices):
                kwargs['layer_name'] = layer.name + '_' + job
            kwargs['new_layer'] = new_layer
            mod_func = self._replace_layer

        else:
            raise ValueError(job + ' is not a recognised job.')

        # Get nodes
        job_nodes = []
        for node_index in node_indices:
            job_nodes.append(layer.inbound_nodes[node_index])
        if any([node for node in job_nodes if node in self.nodes]):
            raise ValueError('Cannot apply several jobs to the same node.')

        # Add the modification function and keyword arguments to the
        # self._mod_func_map dictionary for later retrieval.
        for node in job_nodes:
            self._mod_func_map[node] = mod_func
            self._kwargs_map[node] = kwargs
        self.nodes.extend(job_nodes)

    def operate(self):
        """Perform all jobs assigned to the surgeon.

        Examples:
            # Delete layer_1 and insert layer_3 before layer_2
            surgeon = Surgeon(model)
            surgeon.add_job('delete_layer', layer_1)
            surgeon.add_job('insert_layer', layer_2, new_layer=layer_3)
            new_model = surgeon.operate()
        """
        # Operate on each node in self.nodes by order of decreasing depth.
        sorted_nodes = sorted(self.nodes,
                              key=lambda x: utils.get_node_depth(self.model, x),
                              reverse=True)
        for node in sorted_nodes:
            # Rebuild submodel up to this node
            sub_output_nodes = [
                node.inbound_layers[i].inbound_nodes[node_index]
                for i, node_index in enumerate(node.node_indices)]

            outputs, output_masks = self.rebuild_submodel(self.model.inputs,
                                                          sub_output_nodes)

            # Perform surgery at this node
            try:
                kwargs = self._kwargs_map[node]
            except KeyError:
                kwargs = {}
            self._mod_func_map[node](node, outputs, output_masks, **kwargs)

        # Finish rebuilding model
        output_nodes = [self.model.output_layers[i].inbound_nodes[node_index]
                        for i, node_index in
                        enumerate(self.model.output_layers_node_indices)]
        new_outputs, _ = self.rebuild_submodel(self.model.inputs,
                                               output_nodes)
        new_model = Model(self.model.inputs, new_outputs)

        if self._copy:
            return utils.clean_copy(new_model)
        else:
            return new_model

    def rebuild_submodel(self,
                         model_inputs,
                         output_nodes,
                         submodel_input_masks=None):
        """Rebuild a subsection of a model given its inputs and outputs.

        Re-constructing subsections of a model enables layers or layer instances
        to be removed or modified and new layers or layer instances to be added.

        Arguments:
            model_inputs: List of the submodel's input tensor(s).
            output_nodes(list[Node]): List of the submodel's output node(s)
            submodel_input_masks: Boolean mask for each submodel input.

        Returns:
            (tuple) containing :
                List of the output tensors of the rebuilt submodel
                List of the output masks of the rebuilt submodel
            tuple[submodel output tensors, output masks]

        """
        if not submodel_input_masks:
            submodel_input_masks = [None] * len(model_inputs)

        def _rebuild_rec(node):
            """Rebuilds the model up to `node` recursively.

            Args:
                node(Node): Node to rebuild up to.
            Returns:
                (tuple) containing :
                The output tensor of the rebuilt submodel
                The output mask of the rebuilt submodel

            """
            layer = node.outbound_layer
            logging.debug('getting inputs for: {0}'.format(layer.name))
            # get the inbound node
            # TODO: Assumes that nodes only have a single output tensor. Check!
            node_output = utils.single_element(node.output_tensors)
            if node_output in self._replace_tensors.keys():
                # Check for replaced tensors before any other checks
                logging.debug('bottomed out at replaced output: {0}'.format(
                    node_output))
                output, output_mask = self._replace_tensors[node_output]
                return output, output_mask

            elif node in self._finished_nodes.keys():
                logging.debug('reached finished node: {0}'.format(node))
                return self._finished_nodes[node]

            elif node_output in model_inputs:
                logging.debug('bottomed out at a model input')
                output_mask = submodel_input_masks[
                    model_inputs.index(node_output)]
                return node_output, output_mask

            else:
                inbound_nodes = utils.get_inbound_nodes(node)
                logging.debug('inbound_layers: {0}'.format(
                    [node.outbound_layer.name for node in inbound_nodes]))
                # Recursively rebuild the model up to this node to obtain this
                # layer's inputs and input masks
                inputs, input_masks = zip(
                    *[_rebuild_rec(n) for n in inbound_nodes])

                # Apply masks to the layer weights and call it on its inputs
                new_layer, output_mask = self._apply_delete_mask(node, input_masks)
                output = new_layer(utils.single_element(list(inputs)))

                self._finished_nodes[node] = (output, output_mask)
                logging.debug('layer complete: {0}'.format(layer.name))
                return output, output_mask

        # Call the recursive _rebuild_rec method to rebuild the submodel up to
        # each output layer
        outputs, output_masks = zip(*[_rebuild_rec(n) for n in output_nodes])
        return outputs, output_masks

    def _delete_layer(self, node, inputs, input_masks):
        # Skip the deleted layer by replacing its outputs with it inputs
        if len(inputs) >= 2:
            raise ValueError('Cannot insert new layer at node with multiple '
                             'inbound layers.')
        inputs = utils.single_element(inputs)
        input_masks = utils.single_element(input_masks)
        deleted_layer_output = utils.single_element(node.output_tensors)
        self._replace_tensors[deleted_layer_output] = (inputs, input_masks)

    def _insert_layer(self, node, inputs, input_masks, new_layer=None):
        # This will not work for nodes with multiple inbound layers
        # The previous layer and node must also be specified to enable this
        # functionality.
        if len(inputs) >= 2:
            raise ValueError('Cannot insert new layer at node with multiple '
                             'inbound layers.')
        # Call the new layer on the inbound layer's output
        new_output = new_layer(utils.single_element(inputs))
        # Replace the inbound layer's output with the new layer's output
        old_output = node.inbound_layers[0].get_output_at(node.node_indices[0])
        input_masks = utils.single_element(input_masks)
        self._replace_tensors[old_output] = (new_output, input_masks)

    def _replace_layer(self, node, inputs, input_masks, new_layer=None, layer_name=None):
        if layer_name:
            new_layer.name = layer_name

        # Call the new layer on the rebuild submodel's inputs
        new_output = new_layer(utils.single_element(inputs))

        # Replace the original layer's output with the new layer's output
        replaced_layer_output = utils.single_element(node.output_tensors)
        input_masks = utils.single_element(input_masks)
        self._replace_tensors[replaced_layer_output] = (new_output, input_masks)

    def _delete_channels(self, node, inputs, input_masks, channels=None, layer_name=None):
        # this_layer = node.outbound_layer
        old_layer = node.outbound_layer

        # If this layer has already been operated on, use the cached copy;
        # otherwise, apply the inbound delete mask and delete channels to
        # obtain the modified layer
        if old_layer in self._new_layers_map.keys():
            new_layer = self._new_layers_map[old_layer]
        else:
            temp_layer, new_mask = self._apply_delete_mask(node, input_masks)
            # This call is needed to initialise input_shape and output_shape
            temp_layer(utils.single_element(inputs))
            new_layer = self._delete_channel_weights(temp_layer, channels)
            if layer_name:
                new_layer.name = layer_name
            self._new_layers_map[old_layer] = new_layer

        new_delete_mask = self._make_delete_mask(old_layer, channels)

        # Call the new layer on the rebuild submodel's inputs
        new_output = new_layer(utils.single_element(inputs))

        # Replace the original layer's output with the modified layer's output
        old_layer_output = utils.single_element(node.output_tensors)
        self._replace_tensors[old_layer_output] = (new_output, new_delete_mask)

    def _apply_delete_mask(self, node, inbound_masks):
        """Apply the inbound delete mask and return the outbound delete mask

        When specific channels in a layer or layer instance are deleted, the
        mask propagates information about which channels are affected to
        downstream layers.
        If the layer contains weights, the weights which were previously
        connected to the deleted channels are deleted and outbound masks are
        set to True since further downstream layers aren't affected.
        If the layer does not contain weights, any transformations performed
        on the layer's input are performed on the mask to create the outbound
        mask.

        Arguments:
            node(Node):
            inbound_masks: Masks from previous layer(s).

        Returns:
            new_layer: Pass through `layer` if it has no weights, otherwise a
                       new `Layer` object with weights corresponding to the
                       inbound mask deleted.
            outbound_mask: Mask corresponding to `new_layer`.
        """

        # TODO: This breaks layer sharing. Write a test for this.

        # if delete_mask is None or all values are True, it does not affect
        # this layer or any layers above/downstream from it
        layer = node.outbound_layer
        if all(mask is None for mask in inbound_masks):
            new_layer = layer
            outbound_mask = None
            return new_layer, outbound_mask
        elif any(mask is None for mask in inbound_masks):
            inbound_masks = [np.ones(shape[1:], dtype=bool)
                             if inbound_masks[i] is None else inbound_masks[i]
                             for i, shape in enumerate(node.input_shapes)]

        output_shape = utils.single_element(node.output_shapes)
        input_shape = utils.single_element(node.input_shapes)
        data_format = getattr(layer, 'data_format', 'channels_last')
        inbound_masks = utils.single_element(inbound_masks)
        # otherwise, delete_mask.shape should be: layer.input_shape[1:]
        layer_class = layer.__class__.__name__
        if layer_class == 'InputLayer':
            raise RuntimeError('This should never get here!')

        elif layer_class == 'Dense':
            if np.all(inbound_masks):
                new_layer = layer
            else:
                weights = layer.get_weights()
                weights[0] = weights[0][np.where(inbound_masks)[0], :]
                config = layer.get_config()
                config['weights'] = weights
                new_layer = type(layer).from_config(config)
            outbound_mask = np.ones(output_shape[1:], dtype=bool)

        elif layer_class == 'Flatten':
            outbound_mask = np.reshape(inbound_masks, [-1, ])
            new_layer = layer

        elif layer_class in ('Conv1D', 'Conv2D', 'Conv3D'):
            if np.all(inbound_masks):
                new_layer = layer
            else:
                if data_format == 'channels_first':
                    inbound_masks = np.swapaxes(inbound_masks, 0, -1)
                # Conv layer: trim down inbound_masks to filter shape
                k_size = layer.kernel_size
                index = [slice(None, dim_size, None) for dim_size in
                         k_size]
                inbound_masks = inbound_masks[index + [slice(None)]]
                # Delete unused weights to obtain new_weights
                weights = layer.get_weights()
                # Each deleted channel was connected to all of the channels
                # in layer; therefore, the mask must be repeated for each
                # channel.
                # `delete_mask`'s size: size(inbound_mask)+[layer.filters]
                # TODO: replace repeat with tile
                delete_mask = np.repeat(inbound_masks[..., np.newaxis],
                                        weights[0].shape[-1],
                                        axis=-1)
                new_shape = list(weights[0].shape)
                new_shape[-2] = -1  # Weights always have channels_last
                weights[0] = np.reshape(weights[0][delete_mask], new_shape)
                # Instantiate new layer with new_weights
                config = layer.get_config()
                config['weights'] = weights
                new_layer = type(layer).from_config(config)
            # Set outbound delete mask to ones.
            outbound_mask = np.ones(output_shape[1:], dtype=bool)

        elif layer_class in ('Cropping1D', 'Cropping2D', 'Cropping3D',
                             'MaxPooling1D', 'MaxPooling2D',
                             'MaxPooling3D',
                             'AveragePooling1D', 'AveragePooling2D',
                             'AveragePooling3D'):
            index = [slice(None, x, None) for x in output_shape[1:]]
            if data_format == 'channels_first':
                index[0] = slice(None)
            elif data_format == 'channels_last':
                index[-1] = slice(None)
            else:
                raise ValueError('Invalid data format')
            outbound_mask = inbound_masks[index]
            new_layer = layer

        elif layer_class in ('UpSampling1D',
                             'UpSampling2D',
                             'UpSampling3D',
                             'ZeroPadding1D',
                             'ZeroPadding2D',
                             'ZeroPadding3D'):

            # Get slice of mask with all singleton dimensions except
            # channels dimension
            index = [slice(1)] * (len(input_shape) - 1)
            tile_shape = list(output_shape[1:])
            if data_format == 'channels_first':
                index[0] = slice(None)
                tile_shape[0] = 1
            elif data_format == 'channels_last':
                index[-1] = slice(None)
                tile_shape[-1] = 1
            else:
                raise ValueError('Invalid data format')
            channels_vector = inbound_masks[index]
            # Tile this slice to create the outbound mask
            outbound_mask = np.tile(channels_vector, tile_shape)
            new_layer = layer

        elif layer_class in ('GlobalMaxPooling1D',
                             'GlobalMaxPooling2D',
                             'GlobalAveragePooling1D',
                             'GlobalAveragePooling2D'):
            # Get slice of mask with all singleton dimensions except
            # channels dimension
            index = [0] * (len(input_shape) - 1)
            if data_format == 'channels_first':
                index[0] = slice(None)
            elif data_format == 'channels_last':
                index[-1] = slice(None)
            else:
                raise ValueError('Invalid data format')
            channels_vector = inbound_masks[index]
            # Tile this slice to create the outbound mask
            outbound_mask = channels_vector
            new_layer = layer

        elif layer_class in ('Dropout',
                             'Activation',
                             'SpatialDropout1D',
                             'SpatialDropout2D',
                             'SpatialDropout3D',
                             'ActivityRegularization',
                             'Masking',
                             'LeakyReLU',
                             'PReLU',
                             'ELU',
                             'ThresholdedReLU',
                             'GaussianNoise',
                             'AlphaDropout'):
            # Pass-through layers
            outbound_mask = inbound_masks
            new_layer = layer

        elif layer_class == 'Reshape':
            outbound_mask = np.reshape(inbound_masks,
                                       layer.target_shape)
            new_layer = layer

        elif layer_class == 'Permute':
            outbound_mask = np.transpose(inbound_masks,
                                         [x - 1 for x in layer.dims])
            new_layer = layer

        elif layer_class == 'RepeatVector':
            outbound_mask = np.repeat(
                np.expand_dims(inbound_masks, 0),
                layer.n,
                axis=0)
            new_layer = layer

        elif layer_class == 'Embedding':
            # Embedding will always be the first layer so it doesn't need
            # to consider the inbound_delete_mask
            outbound_mask = np.ones(
                utils.single_element(node.output_tensors)
                [1:], dtype=bool)
            new_layer = layer

        elif layer_class in ('Multiply', 'Average', 'Maximum', 'Dot',):
            # The inputs must be the same size
            if not utils.all_equal(inbound_masks):
                ValueError(
                    '{0} layers must have the same size inputs. All '
                    'inbound nodes must have the same channels deleted'
                    .format(layer_class))
            outbound_mask = inbound_masks[1]
            new_layer = layer

        elif layer_class == 'Concatenate':
            axis = layer.axis
            if layer.axis < 0:
                axis = axis % len(layer.input_shape[0])
            # the mask has one less dimension than the input (batch)
            outbound_mask = np.concatenate(inbound_masks, axis=axis - 1)
            new_layer = layer

        elif layer_class in ('SimpleRNN', 'GRU', 'LSTM'):
            if np.all(inbound_masks):
                new_layer = layer
            else:
                weights = layer.get_weights()
                weights[0] = weights[0][np.where(inbound_masks[0, :])[0], :]
                config = layer.get_config()
                config['weights'] = weights
                new_layer = type(layer).from_config(config)
            outbound_mask = np.ones(output_shape[1:], dtype=bool)

        elif layer_class == 'BatchNormalization':
            outbound_mask = inbound_masks
            # Get slice of mask with all singleton dimensions except
            # channels dimension
            index = [0] * (len(input_shape))
            index[layer.axis] = slice(None)
            index = index[1:]
            # TODO: Maybe use channel indices everywhere instead of masks?
            channel_indices = np.where(inbound_masks[index] == False)[0]
            weights = [np.delete(w, channel_indices, axis=-1)
                       for w in layer.get_weights()]
            new_layer = layers.BatchNormalization.from_config(
                layer.get_config())
            new_input_shape = list(input_shape)
            new_input_shape[new_layer.axis] -= len(channel_indices)
            new_layer.build(new_input_shape)
            new_layer.set_weights(weights)

        else:
            # Not implemented:
            # - Lambda
            # - SeparableConv2D
            # - Conv2DTranspose
            # - LocallyConnected1D
            # - LocallyConnected2D
            # - TimeDistributed
            # - Bidirectional
            # -
            # Warning/error needed for Reshape if channels axis is split
            raise ValueError('"{0}" layers are currently '
                             'unsupported.'.format(layer_class))

        return new_layer, outbound_mask

    def _delete_channel_weights(self, layer, channel_indices):
        """Delete channels from layer and remove un-used weights.

        Arguments:
            layer: A Keras layer
            channel_indices: the indices of the channels to be deleted.

        Returns:
            A new layer with the channels and corresponding weights deleted.
        """
        layer_config = layer.get_config()
        channels_attr = utils.get_channels_attr(layer)
        channel_count = layer_config[channels_attr]
        # Check inputs
        if any([i + 1 > channel_count for i in channel_indices]):
            raise ValueError('Channels_index value(s) out of range. '
                             'This layer only has {0} channels.'
                             .format(channel_count))
        print('Deleting {0}/{1} channels from layer: {2}'.format(
            len(channel_indices), channel_count, layer.name))
        # numpy.delete ignores negative indices in lists: wrap indices
        channel_indices = [i % channel_count for i in channel_indices]

        # Reduce layer channel count in config.
        layer_config[channels_attr] -= len(channel_indices)

        # Delete weights corresponding to deleted channels from config.
        # For all except recurrent layers, the weights' channels dimension is
        # last.
        # Each recurrent layer type has a different internal weights layout.
        if layer.__class__.__name__ == 'SimpleRNN':
            weights = [np.delete(w, channel_indices, axis=-1)
                       for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        elif layer.__class__.__name__ == 'GRU':
            # Repeat the channel indices for all internal GRU weights.
            channel_indices_gru = [layer.units * m + i for m in range(3)
                                   for i in channel_indices]
            weights = [np.delete(w, channel_indices_gru, axis=-1)
                       for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        elif layer.__class__.__name__ == 'LSTM':
            # Repeat the channel indices for all interal LSTM weights.
            channel_indices_lstm = [layer.units * m + i for m in range(4)
                                    for i in channel_indices]
            weights = [np.delete(w, channel_indices_lstm, axis=-1)
                       for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        else:
            weights = [np.delete(w, channel_indices, axis=-1)
                       for w in layer.get_weights()]
        layer_config['weights'] = weights

        # Create new layer from modified config.
        return type(layer).from_config(layer_config)

    def _make_delete_mask(self, layer, channel_indices):
        """Make the boolean delete mask for layer's output deleting channels.
        The mask is used to index the weights of the following layers to remove
        weights previously linked to channels which have been deleted.

        Arguments:
            layer: A Keras layer
            channel_indices: the indices of the channels to be deleted

        Returns:
            A Numpy ndarray of booleans of the same size as the output of layer.
        """
        data_format = getattr(layer, 'data_format', 'channels_last')
        new_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)
        if data_format == 'channels_first':
            new_delete_mask[channel_indices, ...] = False
        elif data_format == 'channels_last':
            new_delete_mask[..., channel_indices] = False
        else:
            ValueError('Invalid data_format property value')
        return new_delete_mask
