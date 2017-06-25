from keras.engine import Model


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


def check_for_layer_reuse(model, layers=None):
    """Returns True if any layers are reused, False if not."""
    if layers is None:
        layers = model.layers
    return any([len(l.inbound_nodes) > 1 for l in layers])
    # for layer in layers:
    #     if len(layer.inbound_nodes) > 1:
    #         return True
    # return False