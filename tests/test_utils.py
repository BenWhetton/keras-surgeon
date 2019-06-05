from tensorflow.python.keras.layers import (
    Conv2D,
    Activation,
    MaxPool2D,
    Flatten,
    Dense,
    Input,
)
from tensorflow.python.keras.models import Sequential
import numpy as np
import pytest

from tfkerassurgeon.utils import (
    find_activation_layer,
    get_shallower_nodes,
    get_inbound_nodes,
    MeanCalculator,
)


def test_get_shallower_nodes():
    input_1 = Input((10,))
    input_2 = Input((10,))
    dense_1 = Dense(3)
    dense_2 = Dense(4)
    dense_3 = Dense(5)

    x = dense_1(input_1)
    node_1_1 = get_inbound_nodes(dense_1)[0]
    y = dense_1(input_2)
    node_2_1 = get_inbound_nodes(dense_1)[1]
    assert node_1_1 != node_2_1

    output_1 = dense_2(x)
    node_1_2 = get_inbound_nodes(dense_2)[0]
    output_2 = dense_3(y)
    node_2_2 = get_inbound_nodes(dense_3)[0]

    assert get_shallower_nodes(node_1_1) == [node_1_2]
    assert get_shallower_nodes(node_2_1) == [node_2_2]

    output_3 = dense_2(y)
    node_2_2_2 = get_inbound_nodes(dense_2)[1]

    assert get_shallower_nodes(node_2_1) == [node_2_2, node_2_2_2]


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


def test_mean_calculator():
    mean_calculator = MeanCalculator(sum_axis=0)
    x1 = np.array([[1, 2, 3], [4, 5, 6]])
    x2 = np.array([[7, 8, 9], [10, 11, 12]])
    expected_mean = np.array([5.5, 6.5, 7.5])
    mean_calculator.add(x1)
    mean_calculator.add(x2)
    result = mean_calculator.calculate()
    assert (result == expected_mean).all()


if __name__ == '__main__':
    pytest.main([__file__])
