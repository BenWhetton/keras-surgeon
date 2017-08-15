from kerasprune.utils import find_activation_layer
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from keras.models import Sequential
import pytest


def test_find_activation_layer():
    conv1_filters = 1
    conv2_filters = 1
    dense_units = 1
    model = Sequential()
    model.add(Conv2D(conv1_filters, [3, 3], input_shape=(28, 28, 1), data_format="channels_last", name='conv_1'))
    model.add(Activation('relu', name='act_1'))
    model.add(MaxPool2D((2, 2), name='pool_1'))
    model.add(Conv2D(conv2_filters, [3, 3], data_format="channels_last", name='conv_2'))
    model.add(Activation('relu', name='act_2'))
    model.add(MaxPool2D((2, 2), name='pool_2'))
    model.add(Flatten(name='flat_1'))
    model.add(Dense(dense_units, name='dense_1'))
    model.add(Activation('relu', name='act_3'))
    model.add(Dense(10, name='dense_2'))
    model.add(Activation('softmax', name='act_4'))
    assert find_activation_layer(model.get_layer('conv_1'), 0) == (model.get_layer('act_1'), 0)
    assert find_activation_layer(model.get_layer('conv_2'),
                                 0) == (model.get_layer('act_2'), 0)
    assert find_activation_layer(model.get_layer('dense_1'),
                                 0) == (model.get_layer('act_3'), 0)
    assert find_activation_layer(model.get_layer('dense_2'),
                                 0) == (model.get_layer('act_4'), 0)


if __name__ == '__main__':
    pytest.main([__file__])
