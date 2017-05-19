from kerasprune.utils import find_layers
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from keras.models import Sequential
import pytest


def test_find_layers():
    conv1_filters = 1
    conv2_filters = 1
    dense_units = 1
    model = Sequential()
    model.add(Conv2D(conv1_filters, [3, 3], input_shape=(28, 28, 1), data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(conv2_filters, [3, 3], data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(dense_units))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    assert find_layers(model, 0) == [1, 3]
    assert find_layers(model, 3) == [4, 7]
    assert find_layers(model, 7) == [8, 9]


if __name__ == '__main__':
    pytest.main([__file__])
