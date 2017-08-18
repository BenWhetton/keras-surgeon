from kerasprune.utils import find_activation_layer, get_shallower_nodes
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Input
from keras.models import Sequential, Model
import pytest


def test_get_shallower_nodes():
    input_1 = Input((10,))
    input_2 = Input((10,))
    dense_1 = Dense(3)
    dense_2 = Dense(4)
    dense_3 = Dense(5)

    x = dense_1(input_1)
    node_1_1 = dense_1.inbound_nodes[0]
    y = dense_1(input_2)
    node_2_1 = dense_1.inbound_nodes[1]
    assert node_1_1 != node_2_1

    output_1 = dense_2(x)
    node_1_2 = dense_2.inbound_nodes[0]
    output_2 = dense_3(y)
    node_2_2 = dense_3.inbound_nodes[0]

    assert get_shallower_nodes(node_1_1) == [node_1_2]
    assert get_shallower_nodes(node_2_1) == [node_2_2]

    output_3 = dense_2(y)
    node_2_2_2 = dense_2.inbound_nodes[1]

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


if __name__ == '__main__':
    pytest.main([__file__])
