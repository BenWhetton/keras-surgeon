import os

import pytest
import numpy as np
from keras import models
from keras import layers

from kerasprune import utils
from kerasprune import prune

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_rebuild_sequential(model_1):
    model_2 = prune.rebuild_sequential(model_1.layers)
    assert compare_models_seq(model_1, model_2)


def test_rebuild_submodel(model_2):
    outputs, _, _ = prune.rebuild_submodel(model_2.inputs,
                                           model_2.output_layers,
                                           model_2.output_layers_node_indices)
    new_model = models.Model(model_2.inputs, outputs)
    assert compare_models(model_2, new_model)


def test_delete_channels_rec_1():
    inputs = layers.Input(shape=(784,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    prune.delete_channels(model, model.layers[2], [0])


def model_3(data_format):
    if data_format is 'channels_last':
        main_input = layers.Input(shape=[7, 7, 1])
    elif data_format is 'channels_first':
        main_input = layers.Input(shape=[1, 7, 7])
    else:
        raise ValueError(data_format + ' is not a valid "data_format" value.')
    x = layers.Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = layers.Conv2D(3, [3, 3], data_format=data_format)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(3)(x)
    main_output = layers.Dense(1)(x)

    model = models.Model(inputs=main_input, outputs=main_output)

    # Set all of the weights
    w1 = [np.asarray([[[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]],
                      [[[10, 11, 12]], [[13, 14, 15]], [[16, 17, 18]]],
                      [[[19, 20, 21]], [[22, 23, 24]], [[25, 26, 27]]]],
                     dtype='float32'),
          np.asarray([100, 200, 300], dtype='float32')]
    model.layers[1].set_weights(w1)
    w2 = [np.reshape(np.arange(0, 3 * 3 * 3 * 3, dtype='float32'),
                     [3, 3, 3, 3]),
          np.asarray([100, 200, 300], dtype='float32')]
    model.layers[2].set_weights(w2)

    w4 = [np.reshape(np.arange(0, 3 * 3 * 3 * 3, dtype='float32'),
                     [3 * 3 * 3, 3]),
          np.asarray([100, 200, 300], dtype='float32')]
    model.layers[4].set_weights(w4)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@pytest.mark.parametrize('data_format', ['channels_first', 'channels_last'])
@pytest.mark.parametrize('channel_index', [[0],
                                           [1],
                                           [2],
                                           [0, 1]])
def test_delete_channels_conv2d_conv2d(channel_index, data_format):
    model = model_3(data_format)
    layer_index = 1
    new_model = prune.delete_channels(model,
                                      model.layers[layer_index],
                                      channel_index,
                                      copy=True)
    w = model.layers[layer_index].get_weights()
    correct_w = [np.delete(w[0], channel_index, axis=-1),
                 np.delete(w[1], channel_index, axis=0)]
    new_w = new_model.layers[layer_index].get_weights()
    assert weights_equal(correct_w, new_w)


@pytest.mark.parametrize('data_format', ['channels_first', 'channels_last'])
@pytest.mark.parametrize("channel_index", [[0],
                                           [1],
                                           [2],
                                           [0, 1]])
def test_delete_channels_conv2d_conv2d_next_layer(channel_index, data_format):
    model = model_3(data_format)
    layer_index = 1
    next_layer_index = 2
    new_model = prune.delete_channels(model,
                                      model.layers[layer_index],
                                      channel_index)
    w = model.layers[next_layer_index].get_weights()
    correct_w = [np.delete(w[0], channel_index, axis=-2),
                 w[1]]
    new_w = new_model.layers[next_layer_index].get_weights()
    assert weights_equal(correct_w, new_w)


def test_delete_channels_maxpooling2d():
    # This returns a tensor
    inputs = layers.Input(shape=(28, 28, 1))
    # a layer instance is callable on a tensor, and returns a tensor
    x = layers.Conv2D(32, [3, 3], activation='relu')(inputs)
    x = layers.Conv2D(16, [3, 3], activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = models.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    new_model = prune.delete_channels(model, model.layers[2], [0])


def weights_equal(w1, w2):
    if len(w1) != len(w2):
        return False
    else:
        return all([np.array_equal(w1[i], w2[i]) for i in range(len(w1))])

# TODO: Add tests for flatten layer


def test_delete_layer():
    # Create all model layers
    input_1 = layers.Input(shape=[7, 7, 1])
    conv2d_1 = layers.Conv2D(3, [3, 3], data_format='channels_last')
    conv2d_2 = layers.Conv2D(3, [3, 3], data_format='channels_last')
    flatten_1 = layers.Flatten()
    dense_1 = layers.Dense(3)
    dense_2 = layers.Dense(3)
    dense_3 = layers.Dense(3)
    dense_4 = layers.Dense(1)
    # Create the base model
    x = conv2d_1(input_1)
    x = conv2d_2(x)
    x = flatten_1(x)
    x = dense_1(x)
    x = dense_2(x)
    x = dense_3(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(models.Model(input_1, output_1))
    # Create the expected modified model
    x = conv2d_1(input_1)
    x = conv2d_2(x)
    x = flatten_1(x)
    x = dense_1(x)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(models.Model(input_1, output_2))
    # Delete layer dense_2
    model_2 = prune.delete_layer(model_1, model_1.get_layer(dense_2.name))
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_delete_layer_reuse():
    # Create all model layers
    input_1 = layers.Input(shape=[3])
    dense_1 = layers.Dense(3)
    dense_2 = layers.Dense(3)
    dense_3 = layers.Dense(3)
    dense_4 = layers.Dense(3)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    x = dense_3(x)
    x = dense_2(x)
    output_1 = dense_4(x)
    # TODO: use clean_copy once keras issue 4160 has been fixed
    # model_1 = prune.clean_copy(Model(input_1, output_1))
    model_1 = models.Model(input_1, output_1)
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_3(x)
    output_2 = dense_4(x)
    # model_2_exp = prune.clean_copy(Model(input_1, output_2))
    model_2_exp = models.Model(input_1, output_2)
    # Delete layer dense_2
    model_2 = prune.delete_layer(model_1,
                                 model_1.get_layer(dense_2.name),
                                 copy=False)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_replace_layer():
    # Create all model layers
    input_1 = layers.Input(shape=[7, 7, 1])
    dense_1 = layers.Dense(3)
    dense_2 = layers.Dense(3)
    dense_3 = layers.Dense(3)
    dense_4 = layers.Dense(1)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(models.Model(input_1, output_1))
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(models.Model(input_1, output_2))
    # Replace dense_2 with dense_3 in model_1
    model_2 = prune.replace_layer(model_1,
                                  model_1.get_layer(dense_2.name),
                                  dense_3)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_insert_layer():
    # Create all model layers
    input_1 = layers.Input(shape=[7, 7, 1])
    dense_1 = layers.Dense(3)
    dense_2 = layers.Dense(3)
    dense_3 = layers.Dense(3)
    dense_4 = layers.Dense(1)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(models.Model(input_1, output_1))
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_2(x)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(models.Model(input_1, output_2))
    # Insert dense_3 before dense_4 in model_1
    model_2 = prune.insert_layer(model_1,
                                 model_1.get_layer(dense_4.name),
                                 dense_3)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_delete_layer_same_layer_outputs():
    # Create all model layers
    input_1 = layers.Input(shape=(10,))
    dense_1 = layers.Dense(3)
    dense_2 = layers.Dense(3)
    dense_3 = layers.Dense(3)
    dense_4 = layers.Dense(1)
    # Create the base model
    x = dense_1(input_1)
    y = dense_2(x)
    x = dense_3(x)
    output_1 = dense_4(x)
    output_2 = dense_4(y)
    model_1 = utils.clean_copy(models.Model(input_1, [output_1, output_2]))
    # Create the expected modified model
    x = dense_1(input_1)
    y = dense_2(x)
    output_1 = dense_4(x)
    output_2 = dense_4(y)
    model_2_exp = utils.clean_copy(models.Model(input_1, [output_1, output_2]))
    # Delete layer dense_3
    model_2 = prune.delete_layer(model_1,
                                 model_1.get_layer(dense_3.name),
                                 copy=False)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def compare_models(model_1, model_2):
    config_1 = model_1.get_config()
    config_2 = model_2.get_config()
    config_2['name'] = config_1['name']  # make the config names identical
    config_match = (config_1 == config_2)
    weights_match = (all([np.array_equal(weight_1, weight_2)
                          for (weight_1, weight_2) in
                          zip(model_1.get_weights(), model_2.get_weights())]))
    return config_match and weights_match


def compare_models_seq(model_1, model_2):
    config_1 = model_1.get_config()
    config_2 = model_2.get_config()
    config_match = (config_1 == config_2)
    weights_match = (all([np.array_equal(weight_1, weight_2)
                          for (weight_1, weight_2) in
                          zip(model_1.get_weights(), model_2.get_weights())]))
    return config_match and weights_match


if __name__ == '__main__':
    pytest.main([__file__])
