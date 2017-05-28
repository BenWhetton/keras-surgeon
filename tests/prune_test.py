import pytest
import numpy as np

from kerasprune.prune import delete_channels
from kerasprune.prune import rebuild_sequential


@pytest.fixture
def model_1():
    """Basic Lenet-style model test fixture with minimal channels"""
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten
    model = Sequential()
    model.add(Conv2D(2, [3, 3], input_shape=[10, 10, 1], data_format='channels_last', activation='relu'))
    model.add(Conv2D(2, [3, 3], activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dense(2, activation='relu'))
    return model


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



if __name__ == '__main__':
    pytest.main([__file__])
