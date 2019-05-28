# Tf-Keras-Surgeon
Note: see Keras-Surgeon: https://github.com/BenWhetton/keras-surgeon

## Introduction
This is a port of Keras-Surgeon by BenWhetton to work with the new TF 2.0 tf.keras which has a few tweaks from normal keras (which causes bugs)

This is not intended to do anything more than the original, so, most of the code and documentation will remain unchanged.

## From Keras-Surgeon:
# Keras-Surgeon

## Introduction
Keras-surgeon provides simple methods for modifying trained 
[Keras][] models. The following functionality is currently implemented:
* delete neurons/channels from layers
* delete layers
* insert layers
* replace layers

Keras-surgeon is compatible with any model architecture. Any number of 
layers can be modified in a single traversal of the network.

These kinds of modifications are sometimes known as network surgery which 
inspired the name of this package.

The `operations` module contains simple methods to perform network surgery on a 
single layer within a model.\
Example usage:
```python
from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
# delete layer_1 from a model
model = delete_layer(model, layer_1)
# insert new_layer_1 before layer_2 in a model
model = insert_layer(model, layer_2, new_layer_3)
# delete channels 0, 4 and 67 from layer_2 in model
model = delete_channels(model, layer_2, [0,4,67])
```

The `Surgeon` class enables many modifications to be performed in a single operation.\
Example usage:
```python
# delete channels 2, 6 and 8 from layer_1 and insert new_layer_1 before 
# layer_2 in a model
from kerassurgeon import Surgeon
surgeon = Surgeon(model)
surgeon.add_job('delete_channels', model, layer_1, channels=[2, 6, 8])
surgeon.add_job('insert_layer', model, layer_2, new_layer=new_layer_1)
new_model = surgeon.operate()
```
The `identify` module contains methods to identify which channels to prune.


## Documentation
The docstrings and this file contain all of the documentation. Standalone 
documentation may be added in the future.


## Motivation
This project was motivated by my interest in deep learning and desire to 
experiment with some of the pruning methods I have read about in the research 
literature. I could not find an easy way to prune neurons from Keras models.

I hope I have created something which will be useful to others.

## Installation
```
pip install kerassurgeon
```
## Examples:
Examples are in `kerassurgeon.examples`.\
Both examples identify which neurons to prune using the method described in 
[Hu et al. (2016)][]: those which have the highest Average Percentage of Zeros (APoZ).\
Neither example is particularly good at demonstrating the benefits of pruning 
but they show how Keras-surgeon can be used.\
I would welcome any good examples from other users.

### Pruning Lenet trained on MNIST:
`lenet_minst` is a very simple example showing the effects of deleting channels from a 
simple Lenet style network trained on MNIST. It demonstrates using the simple 
methods from `kerasurgeon.operations`.

### Inception V3 fine-tuned on flowers data-set:
This example shows how to delete channels from many layers simultaneously using 
the `Surgeon` Class.\
It is in two parts:  
`inception_flowers_tune` shows how to fine-tune the Inception V3 model on a small flowers 
data set (based on a combination of [Tensorflow tutorial] and [Keras blog post]).\
`inception_flowers_prune` demonstrates deleting channels from many layers 
simultaneously using the `Surgeon` Class.


## Limitations:
Only python 3 is currently supported. Only python 3.5 has been tested.\
The following layers are not fully supported; `delete_channels` might not work 
on models containing these layers (it depends if they are affected by the 
operation):
* `Lambda`
* `SeparableConv2D`
* `Conv2DTranspose`
* `LocallyConnected1D`
* `LocallyConnected2D`
* `TimeDistributed`
* `Bidirectional`
* `Dot`
* `PReLU`

Recurrent layers’ sequence length must be defined.\
The model’s input shape must be defined.


## Future improvements:
### Architecture:
Investigate more efficient ways of modifying a layer in the middle of a model 
without re-building the whole network.

### Performance:
This package has not yet been optimised for performance. It can certainly be improved.

### Tests:
Write unit tests for the utility functions.\
This package pretty tightly coupled with Keras which makes unit testing difficult.
Some component tests have been written but it needs more work.

### Examples
Write better examples.


[Hu et al. (2016)]: http://arxiv.org/abs/1607.03250
[Keras]: https://github.com/fchollet/keras
[Tensorflow tutorial]: https://www.tensorflow.org/tutorials/image_retraining#training_on_flowers
[Keras blog post]: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
