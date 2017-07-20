"""Utilities used across other modules."""

def clean_copy(model):
    """Returns a copy of the model without other model uses of its layers."""
    weights = model.get_weights()
    new_model = model.__class__.from_config(model.get_config())
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
    return [node for v in model.nodes_by_depth.values() for node in v]


def find_activation_layer(model, layer_index):
    if model.layers[layer_index].__class__.__name__ in ('Conv2D', 'Dense'):
        assert ValueError('This functionality is only implemented for Conv2D '
                          'and Dense layers.')
    model_config = model.get_config()
    layer_classes = [layer_config['class_name']
                     for layer_config in model_config]
    try:
        next_layer = next(i for i in range(layer_index + 1, len(layer_classes))
                                  if layer_classes[i] not in
                                  {'Flatten', 'Activation', 'MaxPooling2D'})
    except StopIteration:
        print('No layers with weights found after the chosen layer.'
              'Cannot reduce the output channels.')
        raise

    try:
        activation_layer = next(i for i in range(layer_index, next_layer)
                                if ('activation' in model_config[i]['config'].keys())
                                & (model_config[i]['config']['activation'] != 'linear'))
    except StopIteration:
        print('All activations from the chosen layer onwards are linear.'
              'This functionality requires a nonlinear activation function'
              'to be present.')
        raise

    if not model_config[activation_layer]['config']['activation'] == 'relu':
        assert ValueError('This functionality only works for layers using or '
                          'followed by a "relu" activation.')

    return activation_layer


def sort_x_by_y(x, y):
    """Sort the iterable x by the order of iterable y"""
    x = [x for (_, x) in sorted(zip(y, x))]
    return x


def single_element(x):
    """If x contains a single element, return it; otherwise return x"""
    if len(x) == 1:
        x = x[0]
    return x


def bool_to_index(x):
    return [i for i in range(len(x)) if x]