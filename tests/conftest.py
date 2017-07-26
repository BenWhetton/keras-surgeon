import pytest
from keras import models
from keras import layers


@pytest.fixture
def model_1():
    """Basic Lenet-style model test fixture with minimal channels"""
    model = models.Sequential()
    model.add(layers.Conv2D(2,
                            [3, 3],
                            input_shape=[28, 28, 1],
                            data_format='channels_last',
                            activation='relu'))
    model.add(layers.Conv2D(2, [3, 3], activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    return model


@pytest.fixture
def model_2():
    """Basic Lenet-style model test fixture with minimal channels"""
    model = model_1()
    return models.Model(model.inputs, model.outputs)