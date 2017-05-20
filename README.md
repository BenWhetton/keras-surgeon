# Keras-Prune
## Introduction
Keras-Prune is a library of functions for pruining [Keras][] model.\
So far, a method to delete neurons from a Keras model including all 
weights connected to it in downstream layers has been implemented.



## Brief description of modules

###kerasprune.prune
This module contains methods to prune models.
delete_channels deletes a unit or filter (a.k.a. a channel) and all weights connected to it in downstream layers.

### kerasprune.identify
This module contains methods to identify the neurons to delete.
So far, one has been implemented: high_apoz() finds neurons with a high (a)verage (p)ercentage (o)f activations equal to (z)ero.

This enables replicating the results from this paper:
"Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures" - [Hu et al. (2016)][]

## Examples
Delete all neurons with an average percentage of zero activations higher than one standard deviation from the mean of all neurons in its layer.
```python
from kerasprune.identify import high_apoz
from kerasprune.prune import delete_channels
[high_apoz_channels, apoz] = high_apoz(model, layer_index, validation_dataset)
new_model = delete_channels(model, layer_index, high_apoz_channels)
new_model.compile(...)
new_model.fit()
```

The examples package contains usage examples for the completed code.

## Status of development
This is a work in progress and currently only works for `keras.models.Sequential` models comprised of the following layer types:
* `keras.layers.Dense`
* `keras.layers.Conv2D`
* `keras.layers.MaxPool2D`
* `keras.layers.Activation`
* `keras.layers.Flatten`

It may work for more layer types but until the corresponding tests have been written it will raise an exception
if it encounters any other layer types between the layer being pruned and any affected downstream layers.

I am reasonably new to python and most of my coding experience to date has been Matlab (it's the industry standard, yada yada).\
This project is motivated by my interest in deep learning. It is also helping me get to grips with python documentation, packaging, testing etc.

Any feedback on best practices that I may be unintentionally ignoring would be most welcome.


[Hu et al. (2016)]: http://arxiv.org/abs/1607.03250
[Keras]: https://github.com/fchollet/keras