#Keras-surgeon

##Introduction
Keras-surgeon provides simple methods for modifying trained 
[Keras][] models. The following functionality is currently implemented:
* Delete neurons/channels from layers
* Delete layers
* insert layers
* replace layers

Keras-surgeon is compatible with any model architecture. Any number of 
layers can be modified in a single traversal of the network.

These kinds of modifications are sometimes known as network surgery which 
inspired the name of this package.

The `operations` module contains simple methods to perform network surgery on a 
single layer within a model.\
Example usage:


The `Surgeon` class enables many modifications to be performed in a single operation.\
Example usage:
```python
from kerassurgeon import Surgeon
surgeon = Surgeon(model)
surgeon.add_job('delete_channels', model, layer_1, channels=channels)
surgeon.add_job('insert_layer', model, layer_2, new_layer=layer_3)
surgeon.operate()
```
The identify module contains method to identify which channels to prune.


##Motivation
This project was motivated by my interest in deep learning and desire to 
experiment with some of the pruning methods I have read about in the research 
literature. I could not find an easy way to prune neurons from Keras models.

I hope I have created something which will be useful to others.

I am reasonably new to python and this is my first attempt at releasing a
small package to get to grips with the process(documentation, packaging etc.).\
Any feedback on best practices that I may be unaware of would be most welcome.


##Installation
```
pip install kerassurgeon
```
##Examples:
The following examples are both based on a simple method of identifying which neurons to 
prune: high Average Percentage of Zeros (ApoZ) as described in [Hu et al. (2016)][].\
Neither example is particularly good at demonstrating the benefits of pruning 
but they show how Keras-surgeon can be used.\
I would welcome any good examples from other users.

###Pruning Lenet trained on MNIST:
`lenet_minst` is a very simple example showing the effects of deleting channels from a 
simple Lenet style network trained on MNIST. It demonstrates using the simple 
methods from `kerasurgeon.operations`.
Pruning all neurons with ApoZ higher than 1 standard deviation above the mean 
ApoZ in the final convolutional layer and then re-training the network gives 
the following results:

###Inception V3 fine-tuned on flowers data-set:
This example shows how to delete channels from many layers simultaneously using 
the `Surgeon` Class.\
It is in two parts:  
`inception_flowers_tune` shows how to fine-tune the Inception V3 model on a small flowers 
data set (based on a combination of [Tensorflow tutorial] and [Keras blog post]).\
`inception_flowers_prune` shows how to delete channels from many layers simultaneously using 
the `Surgeon` Class.


##Limitations:
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


##Future improvements:
###Architecture:
Investigate more efficient ways of modifying a layer in the middle of a model 
without re-building the whole network.

###Performance:
This package has not yet been optimised for performance. It can certainly be improved.

###Tests:
Investigate why tests are much slower when they are all run than when they are 
run individually.\
Write unit tests for the utility functions.\
This package pretty tightly coupled with Keras which makes unit testing difficult.
Some component tests have been written but it needs more work.
Some layers do not have tests yet.

###Examples
Write better examples.


##Known major bugs:
When using `delete_channels`, shared layers stop being shared if their weights 
are modified as a result of another layer’s channels being deleted. E.g. if a 
shared layer is the next weighted layer downstream from a pruned layer, the 
sharing will be broken.

[Hu et al. (2016)]: http://arxiv.org/abs/1607.03250
[Keras]: https://github.com/fchollet/keras
[Tensorflow tutorial]: https://www.tensorflow.org/tutorials/image_retraining#training_on_flowers
[Keras blog post]: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
