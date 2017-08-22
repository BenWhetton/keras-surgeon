import os

import pytest
import numpy as np
from keras import models
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPool2D, Cropping2D, UpSampling2D
from keras.layers import ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import SimpleRNN, GRU, LSTM, BatchNormalization
from numpy import random

from kerassurgeon import operations
from kerassurgeon import utils
from kerassurgeon import Surgeon

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@pytest.fixture(params=['channels_first', 'channels_last'])
def data_format(request):
    return request.param


@pytest.fixture(params=[[0], [-1], [1, 2]], ids=str)
def channel_index(request):
    return request.param


@pytest.fixture
def model_1():
    """Basic Lenet-style model test fixture with minimal channels"""
    model = Sequential()
    model.add(Conv2D(2,
                     [3, 3],
                     input_shape=[28, 28, 1],
                     data_format='channels_last',
                     activation='relu'))
    model.add(Conv2D(2, [3, 3], activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dense(10, activation='relu'))
    return model


@pytest.fixture
def model_2():
    """Basic Lenet-style model test fixture with minimal channels"""
    model = model_1()
    return Model(model.inputs, model.outputs)


def test_rebuild_sequential(model_1):
    model_2 = operations.rebuild_sequential(model_1.layers)
    assert compare_models_seq(model_1, model_2)


def test_rebuild_submodel(model_2):
    output_nodes = [model_2.output_layers[i].inbound_nodes[node_index]
                    for i, node_index in
                    enumerate(model_2.output_layers_node_indices)]
    surgeon = Surgeon(model_2)
    outputs, _ = surgeon._rebuild_graph(model_2.inputs, output_nodes)
    new_model = Model(model_2.inputs, outputs)
    assert compare_models(model_2, new_model)


def test_delete_channels_rec_1():
    inputs = Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    operations.delete_channels(model, model.layers[2], [0])


def model_3(data_format):
    if data_format is 'channels_last':
        main_input = Input(shape=[7, 7, 1])
    elif data_format is 'channels_first':
        main_input = Input(shape=[1, 7, 7])
    else:
        raise ValueError(data_format + ' is not a valid "data_format" value.')
    x = Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = Conv2D(3, [3, 3], data_format=data_format)(x)
    x = Flatten()(x)
    x = Dense(3)(x)
    main_output = Dense(1)(x)

    model = Model(inputs=main_input, outputs=main_output)

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


def test_delete_channels_conv2d_conv2d(channel_index, data_format):
    model = model_3(data_format)
    layer_index = 1
    new_model = operations.delete_channels(model,
                                           model.layers[layer_index],
                                           channel_index,
                                           copy=True)
    channel_count = model.layers[layer_index].filters
    channel_index = [i % channel_count for i in channel_index]
    w = model.layers[layer_index].get_weights()
    correct_w = [np.delete(w[0], channel_index, axis=-1),
                 np.delete(w[1], channel_index, axis=0)]
    new_w = new_model.layers[layer_index].get_weights()
    assert weights_equal(correct_w, new_w)


def test_delete_channels_conv2d_conv2d_next_layer(channel_index, data_format):
    model = model_3(data_format)
    layer_index = 1
    next_layer_index = 2
    new_model = operations.delete_channels(model,
                                           model.layers[layer_index],
                                           channel_index)
    channel_count = model.layers[layer_index].filters
    channel_index = [i % channel_count for i in channel_index]
    w = model.layers[next_layer_index].get_weights()
    correct_w = [np.delete(w[0], channel_index, axis=-2),
                 w[1]]
    new_w = new_model.layers[next_layer_index].get_weights()
    assert weights_equal(correct_w, new_w)


def test_delete_channels_flatten(channel_index, data_format):
    # Create model
    main_input = Input(shape=list(random.randint(4, 10, size=3)))
    x = Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = Flatten()(x)
    main_output = Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    layer_index = 1
    next_layer_index = 3
    layer = model.layers[layer_index]
    new_model = operations.delete_channels(model, layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.output_shape[1:])
    channel_count = getattr(layer, utils.get_channels_attr(layer))
    channel_index = [i % channel_count for i in channel_index]
    if data_format == 'channels_first':
        delete_indices = [x*flat_sz//channel_count + i for x in channel_index
                          for i in range(0, flat_sz//channel_count, )]
    elif data_format == 'channels_last':
        delete_indices = [x + i for i in range(0, flat_sz, channel_count)
                          for x in channel_index]
    else:
        raise ValueError
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def test_delete_channels_maxpooling2d(channel_index, data_format):
    layer = MaxPool2D([2, 2], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_cropping2d(channel_index, data_format):
    layer = Cropping2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_upsampling2d(channel_index, data_format):
    layer = UpSampling2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_zeropadding2d(channel_index, data_format):
    layer = ZeroPadding2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_averagepooling2d(channel_index, data_format):
    layer = AveragePooling2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_globalaveragepooling2d(channel_index, data_format):
    layer = GlobalAveragePooling2D(data_format=data_format)
    layer_test_helper_2d(layer, channel_index, data_format)


def test_delete_channels_simplernn(channel_index):
    layer = SimpleRNN(9, return_sequences=True)
    recursive_test_helper(layer, channel_index)


def test_delete_channels_gru(channel_index):
    layer = GRU(9, return_sequences=True)
    recursive_test_helper(layer, channel_index)


def test_delete_channels_lstm(channel_index):
    layer = LSTM(9, return_sequences=True)
    recursive_test_helper(layer, channel_index)


def test_delete_channels_batchnormalization(channel_index, data_format):
    if data_format == 'channels_first':
        axis = 1
    else:
        axis = -1

    layer = BatchNormalization(axis=axis)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)

# TODO: Concatenate tests (test for batch axis?)


def recursive_test_helper(layer, channel_index):
    main_input = Input(shape=[32, 10])
    x = layer(main_input)
    x = GRU(4, return_sequences=False)(x)
    main_output = Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    next_layer_index = 2
    del_layer = model.layers[del_layer_index]
    new_model = operations.delete_channels(model, del_layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], channel_index, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_2d(layer, channel_index, data_format):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    main_input = Input(shape=list(random.randint(10, 20, size=3)))
    x = Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = layer(x)
    main_output = Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    next_layer_index = 3
    del_layer = model.layers[del_layer_index]
    new_model = operations.delete_channels(model, del_layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], channel_index, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_flatten_2d_bak(layer, channel_index, data_format):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    main_input = Input(shape=list(random.randint(10, 20, size=3)))
    x = Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = layer(x)
    x = Flatten()(x)
    main_output = Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    layer_index = 2
    next_layer_index = 4
    layer = model.layers[layer_index]
    del_layer = model.layers[del_layer_index]
    new_model = operations.delete_channels(model, del_layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.get_output_shape_at(0)[1:])
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    if data_format == 'channels_first':
        delete_indices = [x * flat_sz // channel_count + i for x in
                          channel_index
                          for i in range(0, flat_sz // channel_count, )]
    elif data_format == 'channels_last':
        delete_indices = [x + i for i in range(0, flat_sz, channel_count)
                          for x in channel_index]
    else:
        raise ValueError
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_flatten_2d(layer, channel_index, data_format):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    main_input = Input(shape=list(random.randint(10, 20, size=3)))
    x = Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = layer(x)
    x = Flatten()(x)
    main_output = Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    layer_index = 2
    next_layer_index = 4
    layer = model.layers[layer_index]
    del_layer = model.layers[del_layer_index]
    surgeon = Surgeon(model)
    surgeon.add_job('delete_channels', del_layer, channels=channel_index)
    new_model = surgeon.operate()
    # new_model = prune.delete_channels(model, del_layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.get_output_shape_at(0)[1:])
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    if data_format == 'channels_first':
        delete_indices = [x * flat_sz // channel_count + i for x in
                          channel_index
                          for i in range(0, flat_sz // channel_count, )]
    elif data_format == 'channels_last':
        delete_indices = [x + i for i in range(0, flat_sz, channel_count)
                          for x in channel_index]
    else:
        raise ValueError
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def weights_equal(w1, w2):
    if len(w1) != len(w2):
        return False
    else:
        return all([np.array_equal(w1[i], w2[i]) for i in range(len(w1))])


def test_delete_layer():
    # Create all model layers
    input_1 = Input(shape=[7, 7, 1])
    conv2d_1 = Conv2D(3, [3, 3], data_format='channels_last')
    conv2d_2 = Conv2D(3, [3, 3], data_format='channels_last')
    flatten_1 = Flatten()
    dense_1 = Dense(3)
    dense_2 = Dense(3)
    dense_3 = Dense(3)
    dense_4 = Dense(1)
    # Create the base model
    x = conv2d_1(input_1)
    x = conv2d_2(x)
    x = flatten_1(x)
    x = dense_1(x)
    x = dense_2(x)
    x = dense_3(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(Model(input_1, output_1))
    # Create the expected modified model
    x = conv2d_1(input_1)
    x = conv2d_2(x)
    x = flatten_1(x)
    x = dense_1(x)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(Model(input_1, output_2))
    # Delete layer dense_2
    model_2 = operations.delete_layer(model_1, model_1.get_layer(dense_2.name))
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_delete_layer_reuse():
    # Create all model layers
    input_1 = Input(shape=[3])
    dense_1 = Dense(3)
    dense_2 = Dense(3)
    dense_3 = Dense(3)
    dense_4 = Dense(3)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    x = dense_3(x)
    x = dense_2(x)
    output_1 = dense_4(x)
    # TODO: use clean_copy once keras issue 4160 has been fixed
    # model_1 = prune.clean_copy(Model(input_1, output_1))
    model_1 = Model(input_1, output_1)
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_3(x)
    output_2 = dense_4(x)
    # model_2_exp = prune.clean_copy(Model(input_1, output_2))
    model_2_exp = Model(input_1, output_2)
    # Delete layer dense_2
    model_2 = operations.delete_layer(model_1,
                                      model_1.get_layer(dense_2.name),
                                      copy=False)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_replace_layer():
    # Create all model layers
    input_1 = Input(shape=[7, 7, 1])
    dense_1 = Dense(3)
    dense_2 = Dense(3)
    dense_3 = Dense(3)
    dense_4 = Dense(1)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(Model(input_1, output_1))
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(Model(input_1, output_2))
    # Replace dense_2 with dense_3 in model_1
    model_2 = operations.replace_layer(model_1,
                                       model_1.get_layer(dense_2.name),
                                       dense_3)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_insert_layer():
    # Create all model layers
    input_1 = Input(shape=[7, 7, 1])
    dense_1 = Dense(3)
    dense_2 = Dense(3)
    dense_3 = Dense(3)
    dense_4 = Dense(1)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(Model(input_1, output_1))
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_2(x)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(Model(input_1, output_2))
    # Insert dense_3 before dense_4 in model_1
    model_2 = operations.insert_layer(model_1,
                                      model_1.get_layer(dense_4.name),
                                      dense_3)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_delete_layer_same_layer_outputs():
    # Create all model layers
    input_1 = Input(shape=(10,))
    dense_1 = Dense(3)
    dense_2 = Dense(3)
    dense_3 = Dense(3)
    dense_4 = Dense(1)
    # Create the base model
    x = dense_1(input_1)
    y = dense_2(x)
    x = dense_3(x)
    output_1 = dense_4(x)
    output_2 = dense_4(y)
    model_1 = utils.clean_copy(Model(input_1, [output_1, output_2]))
    # Create the expected modified model
    x = dense_1(input_1)
    y = dense_2(x)
    output_1 = dense_4(x)
    output_2 = dense_4(y)
    model_2_exp = utils.clean_copy(Model(input_1, [output_1, output_2]))
    # Delete layer dense_3
    model_2 = operations.delete_layer(model_1,
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
