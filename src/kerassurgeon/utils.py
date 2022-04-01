"""Utilities used across other modules."""
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import linear
import tensorflow as tf
import keras
from ._utils import node as node_utils


def clean_copy(model):
    """Returns a copy of the model without other model uses of its layers."""
    weights = model.get_weights()
    new_model = model.__class__.from_config(model.get_config())
    new_model.set_weights(weights)
    return new_model


def get_channels_attr(layer):
    layer_config = layer.get_config()
    if 'units' in layer_config.keys():
        channels_attr = 'units'
    elif 'filters' in layer_config.keys():
        channels_attr = 'filters'
    else:
        raise ValueError('This layer has not got any channels.')
    return channels_attr


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
    for (depth, nodes_at_depth) in model._nodes_by_depth.items():
        if node in nodes_at_depth:
            return depth
    raise KeyError('The node is not contained in the model.')


def check_for_layer_reuse(model, layers=None):
    """Returns True if any layers are reused, False if not."""
    if layers is None:
        layers = model.layers
    return any([len(l.inbound_nodes) > 1 for l in layers])


def find_nodes_in_model(model, layer):
    """Find the indices of layer's inbound nodes which are in model"""
    model_nodes = get_model_nodes(model)
    node_indices = []
    for i, node in enumerate(layer.inbound_nodes):
        if node in model_nodes:
            node_indices.append(i)
    return node_indices


def check_nodes_in_model(model, nodes):
    """Check if nodes are in model"""
    model_nodes = get_model_nodes(model)
    nodes_in_model = [False] * len(nodes)
    for i, node in enumerate(nodes):
        if node in model_nodes:
            nodes_in_model[i] = True
    return nodes_in_model


def get_model_nodes(model):
    """Return all nodes in the model"""
    return [node for v in model._nodes_by_depth.values() for node in v]


def get_shallower_nodes(node):
    possible_nodes = node.outbound_layer.outbound_nodes
    next_nodes = []
    for n in possible_nodes:
        if node in node_utils.parent_nodes(n):
            next_nodes.append(n)
    return next_nodes


def get_node_index(node):
    for i, n in enumerate(node.outbound_layer.inbound_nodes):
        if node == n:
            return i


def find_activation_layer(layer, node_index):
    """

    Args:
        layer(Layer):
        node_index:
    """
    output_shape = layer.get_output_shape_at(node_index)
    maybe_layer = layer
    node = maybe_layer.inbound_nodes[node_index]
    # Loop will be broken by an error if an output layer is encountered
    while True:
        # If maybe_layer has a nonlinear activation function return it and its index
        activation = getattr(maybe_layer, 'activation', linear)
        if activation.__name__ != 'linear':
            if maybe_layer.get_output_shape_at(node_index) != output_shape:
                ValueError('The activation layer ({0}), does not have the same'
                           ' output shape as {1}'.format(maybe_layer.name,
                                                         layer.name))
            return maybe_layer, node_index

        # If not, move to the next layer in the datastream
        next_nodes = get_shallower_nodes(node)
        # test if node is a list of nodes with more than one item
        if len(next_nodes) > 1:
            ValueError('The model must not branch between the chosen layer'
                       ' and the activation layer.')
        node = next_nodes[0]
        node_index = get_node_index(node)
        maybe_layer = node.outbound_layer

        # Check if maybe_layer has weights, no activation layer has been found
        if maybe_layer.weights and (
                not maybe_layer.__class__.__name__.startswith('Global')):
            AttributeError('There is no nonlinear activation layer between {0}'
                           ' and {1}'.format(layer.name, maybe_layer.name))


def sort_x_by_y(x, y):
    """Sort the iterable x by the order of iterable y"""
    x = [x for (_, x) in sorted(zip(y, x))]
    return x


def single_element(x):
    """If x contains a single element, return it; otherwise return x"""
    if isinstance(x, (tf.Tensor, keras.engine.keras_tensor.KerasTensor)):
        return x

    if len(x) == 1:
        x = x[0]
    return x


def get_one_tensor(x):
    if isinstance(x, tf.Tensor):
        return x

    assert len(x) == 1
    return x[0]


def bool_to_index(x):
    return [i for i, v in enumerate(x) if v]


def all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(
            np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


class MeanCalculator:
    def __init__(self, sum_axis):
        self.values = None
        self.n = 0
        self.sum_axis = sum_axis

    def add(self, v):
        if self.values is None:
            self.values = v.sum(axis=self.sum_axis)
        else:
            self.values += v.sum(axis=self.sum_axis)
        self.n += v.shape[self.sum_axis]

    def calculate(self):
        return self.values / self.n
