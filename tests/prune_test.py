import pytest
import numpy as np

from kerasprune.prune import delete_channels
from kerasprune.prune import rebuild_sequential
from kerasprune.prune import rebuild


@pytest.fixture
def model_1():
    """Basic Lenet-style model test fixture with minimal channels"""
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten
    model = Sequential()
    model.add(Conv2D(2, [3, 3], input_shape=[28, 28, 1], data_format='channels_last', activation='relu'))
    model.add(Conv2D(2, [3, 3], activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dense(10, activation='relu'))
    return model


@pytest.fixture
def model_2():
    """Basic Lenet-style model test fixture with minimal channels"""
    from keras.models import Model
    from keras.layers import Dense, Conv2D, Flatten
    model = model_1()
    return Model(model.inputs, model.outputs)


def test_delete_channel_conv2d_conv2d(model_1):
    layer_index = 0
    channels_index = [0]
    new_model = delete_channels(model_1, layer_index, channels_index)
    weights = model_1.layers[layer_index].get_weights()
    new_weights = new_model.layers[layer_index].get_weights()
    assert np.array_equal(weights[0][:, :, :, 1:], new_weights[0])
    assert np.array_equal(weights[1][1:], new_weights[1])
    weights = model_1.layers[layer_index+1].get_weights()
    new_weights = new_model.layers[layer_index+1].get_weights()
    assert np.array_equal(weights[0][:, :, 1:, :], new_weights[0])
    assert np.array_equal(weights[1], new_weights[1])


def test_delete_channel_dense_dense(model_1):
    layer_index = 3
    channels_index = [0]
    new_model = delete_channels(model_1, layer_index, channels_index)
    weights = model_1.layers[layer_index].get_weights()
    new_weights = new_model.layers[layer_index].get_weights()
    assert np.array_equal(weights[0][:, 1:], new_weights[0])
    assert np.array_equal(weights[1][1:], new_weights[1])
    weights = model_1.layers[layer_index+1].get_weights()
    new_weights = new_model.layers[layer_index + 1].get_weights()
    assert np.array_equal(weights[0][1:, :], new_weights[0])
    assert np.array_equal(weights[1], new_weights[1])


def test_delete_channel_conv2d_dense(model_1):
    layer_index = 1
    channels_index = [0]
    new_model = delete_channels(model_1, layer_index, channels_index)
    weights = model_1.layers[layer_index].get_weights()
    new_weights = new_model.layers[layer_index].get_weights()
    assert np.array_equal(weights[0][:, :, :, 1:], new_weights[0])
    assert np.array_equal(weights[1][1:], new_weights[1])
    weights = model_1.layers[layer_index+2].get_weights()
    new_weights = new_model.layers[layer_index + 2].get_weights()
    assert np.array_equal(np.delete(weights[0], slice(0, None, 2), axis=0), new_weights[0])
    assert np.array_equal(weights[1], new_weights[1])


def test_rebuild_sequential(model_1):
    new_model = rebuild_sequential(model_1)


def test_rebuild(model_2):
    from tensorflow.examples.tutorials.mnist import input_data
    model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mnist = input_data.read_data_sets('tempData', one_hot=True, reshape=False)
    new_model = rebuild(model_2)
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss = model_2.evaluate(mnist.validation.images, mnist.validation.labels, 128)
    loss2 = model_2.evaluate(mnist.validation.images, mnist.validation.labels, 128)
    assert np.array_equal(loss, loss2)


def test_delete_channels_rec():
    from keras.layers import Input, Dense
    from keras.models import Model
    from kerasprune.prune import delete_channels_rec
    inputs = Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    new_model = delete_channels_rec(model, model.layers[2], [0])



if __name__ == '__main__':
    pytest.main([__file__])
