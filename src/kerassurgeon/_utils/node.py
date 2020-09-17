from . import layer as layer_utils
import collections.abc


def make_list_if_not(x):
    if isinstance(x, collections.abc.Sequence) and not isinstance(x, str):
        return x
    else:
        return [x]


def node_indices(node):
    return make_list_if_not(node.node_indices)


def inbound_layers(node):
    return make_list_if_not(node.inbound_layers)


def parent_nodes(node):
    try:
        return node.parent_nodes
    except AttributeError:
        return [layer_utils.inbound_nodes(inbound_layers(node)[i])[node_index]
                for i, node_index in enumerate(node_indices(node))]
