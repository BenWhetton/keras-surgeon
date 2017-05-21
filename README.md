# Keras-Prune
## Introduction
Keras-Prune is a library of functions for pruning [Keras][] models.

A number of papers have been written about compressing deep neural 
networks by removing connections and/or entire neurons. This area of 
research is interesting because it might help better understand the 
factors that contribute to the generalisation performance of neural networks.

Currently two public methods have been implemented:
* `kerasprune.prune.delete_neurons`: delete neurons from a Keras model layer including all 
weights connected to it in downstream layers.
* `kerasprune.identify.high_apoz`: identify neurons with a high (a)verage (p)ercentage
(o)f activations equal to (z)ero. This helps reproduce the results of this paper:
"Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures" - [Hu et al. (2016)][]

## Motivation
This project is motivated by my interest in deep learning and desire to experiment with some of the pruning methods I have read about in the academic literature.

I am reasonably new to python and this is my first attempt at releasing a small package to get to grips with the process (documentation, packaging etc.).\
Any feedback on best practices that I may be unaware of would be most welcome.

## Status of development
This is a work in progress and currently only works for `keras.models.Sequential` models comprised of the following layer types:
* `keras.layers.Dense`
* `keras.layers.Conv2D`
* `keras.layers.MaxPool2D`
* `keras.layers.Activation`
* `keras.layers.Flatten`

It may work for more layer types but until the corresponding tests have been written it will raise an exception
if it encounters any other layer types between the layer being pruned and any affected downstream layers.

I am in the process of re-writing most of the code to accomodate other layer types and more complex architectures.


[Hu et al. (2016)]: http://arxiv.org/abs/1607.03250
[Keras]: https://github.com/fchollet/keras



## Examples

The examples subpackage contains usage examples for the completed code.
More and improved examples will be added as they become available.