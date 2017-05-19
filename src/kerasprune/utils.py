import keras


def _check_valid_layer(layer):
    # Checks that this functionality has been implemented for the weighted layers' classes.
    if not isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
        assert ValueError('This functionality is only implemented for Conv2D and Dense layers.')


def find_layers(model, layer_index):
    _check_valid_layer(model.layers[layer_index])
    model_config = model.get_config()
    layer_classes = [layer_config['class_name'] for layer_config in model_config]
    try:
        next_weights_layer = next(i for i in range(layer_index + 1, len(layer_classes))
                                  if layer_classes[i] not in
                                  {'Flatten', 'Activation', 'MaxPooling2D'})
    except StopIteration:
        print('No layers with weights found after the chosen layer. Cannot reduce the output channels.')
        raise

    try:
        activation_layer = next(i for i in range(layer_index, next_weights_layer)
                                if ('activation' in model_config[i]['config'].keys())
                                & (model_config[i]['config']['activation'] != 'linear'))
    except StopIteration:
        print('All activations from the chosen layer onwards are linear.'
              ' This functionality requires a nonlinear activation function to be present.')
        raise

    if not model_config[activation_layer]['config']['activation'] == 'relu':
        assert ValueError('This functionality only works for layers using or followed by a "relu" activation.')

    _check_valid_layer(model.layers[next_weights_layer])

    return [activation_layer, next_weights_layer]


