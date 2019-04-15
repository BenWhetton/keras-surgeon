"""Identify which channels to delete."""
import numpy as np
from keras import backend as k
from keras.models import Model
import types

from kerassurgeon import utils


def get_apoz(model, layer, x_val, node_indices=None, steps=None):
    """Identify neurons with high Average Percentage of Zeros (APoZ).

    The APoZ a.k.a. (A)verage (P)ercentage (o)f activations equal to (Z)ero,
    is a metric for the usefulness of a channel defined in this paper:
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient
    Deep Architectures" - [Hu et al. (2016)][]
    `high_apoz()` enables the pruning methodology described in this paper to be
    replicated.

    If node_indices are not specified and the layer is shared within the model
    the APoZ will be calculated over all instances of the shared layer.

    Args:
        model: A Keras model.
        layer: The layer whose channels will be evaluated for pruning.
        x_val: The input of the validation set. This will be used to calculate
            the activations of the layer of interest.
        node_indices(list[int]): (optional) A list of node indices.

    Returns:
        List of the APoZ values for each channel in the layer.
    """

    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')
    

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    mean_calculator = utils.MeanCalculator(sum_axis=0)
    for node_index in node_indices:
        act_layer, act_index = utils.find_activation_layer(layer, node_index)
        # Get activations
        if hasattr(x_val, "__iter__"):
            
            # temp_model = Model(model.inputs, act_layer.get_output_at(act_index))
            temp_model = Model(model.inputs, layer.get_output_at(node_index))
            a = temp_model.predict_generator_intermediate(
                    x_val, 
                    steps=steps)
            
        else:
            get_activations = k.function(
                [utils.single_element(model.inputs), k.learning_phase()],
                [act_layer.get_output_at(act_index)])
            a = get_activations([x_val, 0])[0]
            # Ensure that the channels axis is last
        if data_format == 'channels_first':
            a = np.swapaxes(a, 1, -1)
        # Flatten all except channels axis
        activations = np.reshape(a, [-1, a.shape[-1]])
        zeros = (activations == 0).astype(int)
        mean_calculator.add(zeros)

    return mean_calculator.calculate()

    
    
def high_apoz(apoz, method="std", cutoff_std=1, cutoff_absolute=0.99):
    """
    Args:
        apoz: List of the APoZ values for each channel in the layer.
        method: Cutoff method for high APoZ. "std", "absolute" or "both".
        cutoff_std: Channels with a higher APoZ than the layer mean plus
            `cutoff_std` standard deviations will be identified for pruning.
        cutoff_absolute: Channels with a higher APoZ than `cutoff_absolute`
            will be identified for pruning.

    Returns:
        high_apoz_channels: List of indices of channels with high APoZ.

    """
    if method not in {'std', 'absolute', 'both'}:
        raise ValueError('Invalid `mode` argument. '
                         'Expected one of {"std", "absolute", "both"} '
                         'but got', method)
    if method == "std":
        cutoff = apoz.mean() + apoz.std()*cutoff_std
    elif method == 'absolute':
        cutoff = cutoff_absolute
    else:
        cutoff = min([cutoff_absolute, apoz.mean() + apoz.std()*cutoff_std])
    
    return np.where(apoz >= cutoff)[0]



def get_kernel_sum(convLayer):
    """
    Args:
        convLayer: Layer of the network to perform the sum on

    Returns:
        total: the sum for each kernel - will be a list of length kernels.shape[-1]

    """
    # this needs to be a convolutional layer!
    layerName = convLayer.__class__.__name__
    
    if layerName == "TimeDistributed":
        convLayer = convLayer.layer
    elif "Conv" not in layerName:
        print("Only use on convolutional layers")
        return False
    
    kernels = convLayer.get_weights()[0]
    
    kernels = np.abs(kernels)
    numAxes = len(kernels.shape) - 1
    total = np.sum(kernels, axis=numAxes)
    for n in range(numAxes-1, -1, -1):
        total = np.sum(total, axis=n)
    
    # previous - for time distributed convolution
    # total = np.sum(kernels, axis=2)
    # total = np.sum(total, axis=1)
    # total = np.sum(total, axis=0)
    # print(total.shape)
    return total
    
    
def get_output_sum(model, layer, x_val, node_indices=None, steps=None):
    """
    Args:
        model: A Keras model.
        layer: The layer whose channels will be evaluated for pruning.
        x_val: The input of the validation set. This will be used to calculate
            the activations of the layer of interest.
        node_indices(list[int]): (optional) A list of node indices.
        steps: number of steps for a generator 

    Returns:
        total: total output given a dataset from each kernel

    """

    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')
    

    layer_node_indices = utils.find_nodes_in_model(model, layer)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    total_sum = None
    for node_index in node_indices:
        act_layer, act_index = utils.find_activation_layer(layer, node_index)
        # Get activations
        if isinstance(x_val,  types.GeneratorType):

            # temp_model = Model(model.inputs, act_layer.get_output_at(act_index))
            temp_model = Model(model.inputs, layer.get_output_at(node_index))
            a = temp_model.predict_generator_intermediate(
                    x_val, 
                    steps=steps)
            
        else:
            get_activations = k.function(
                [utils.single_element(model.inputs), k.learning_phase()],
                [act_layer.get_output_at(act_index)])
            a = get_activations([x_val, 0])[0]
            
        # Ensure that the channels axis is last
        if data_format == 'channels_first':
            a = np.swapaxes(a, 1, -1)
        

        
        numAxes = len(a.shape) - 1
        total = np.sum(a, axis=numAxes)
        for n in range(numAxes-1, -1, -1):
            total = np.sum(total, axis=n)
        
        # previous - for time distributed convolution
        # a = np.abs(a)
        # total = np.sum(a, axis=2)
        # total = np.sum(total, axis=1)
        # total = np.sum(total, axis=0)
        
        if total_sum is None:
            total_sum = total
        else:
            total_sum += total
            
    return total_sum

def lowest_sum(totals, cnt):
    """
    Args:
        totals: list of totals from one of the get_X_sum functions
        cnd: number of nodes to return

    Returns:
        list of nodes to prune

    """
    
    return totals.argsort()[:cnt]
    