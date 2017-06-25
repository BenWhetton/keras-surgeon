"""Utilities used across other modules."""


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


def extract_if_single_element(x):
    """If x contains a single element, return it; otherwise return x"""
    if len(x) == 1:
        x = x[0]
    return x