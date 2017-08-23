from kerassurgeon.surgeon import Surgeon


def delete_layer(model, layer, *, node_indices=None, copy=True):
    """Delete instances of a layer from a Keras model.

    Args:
        model: A Model.
        layer: A Layer contained in model.
        node_indices: The indices of the inbound_node to the layer instances to
                      be deleted.
        copy: If True, the model will be copied before and after
              manipulation. This keeps both the old and new models' layers
              clean of each-others data-streams.

    Returns:
        Keras Model object with the layer at node_index deleted.
    """
    surgeon = Surgeon(model, copy)
    surgeon.add_job('delete_layer', layer, node_indices=node_indices)
    return surgeon.operate()


def insert_layer(model, layer, new_layer, *, node_indices=None, copy=True):
    """Insert new_layer before instances of layer.

    If node_indices is not specified. The layer will be inserted before all
    instances of the layer in the model.

    Args:
        model: A Model.
        layer: A Layer contained in model.
        new_layer: A layer to be inserted into model before layer.
        node_indices: The indices of the inbound_node to layer where the
                      new layer is to be inserted.
        copy: If True, the model will be copied before and after
              manipulation. This keeps both the old and new models' layers
              clean of each-others data-streams.

    Returns:
        A new Model object with layer inserted.
    """
    surgeon = Surgeon(model, copy)
    surgeon.add_job('insert_layer', layer,
                    node_indices=node_indices, new_layer=new_layer)
    return surgeon.operate()


def replace_layer(model, layer, new_layer, *,  node_indices=None, copy=True):
    """Replace instances of layer with new_layer.

        If node_indices is not specified, all instances of layer will be
        replaced by instances of new_layer

        Args:
            model: A Model.
            layer: A Layer contained in model.
            new_layer: A layer to be inserted into model before layer.
            node_indices: The indices of the inbound_node to layer where the
                          new layer is to be inserted.
            copy: If True, the model will be copied before and after
                  manipulation. This keeps both the old and new models' layers
                  clean of each-others data-streams.

        Returns:
            A new Model object with layer inserted.
        """
    surgeon = Surgeon(model, copy)
    surgeon.add_job('replace_layer', layer,
                    node_indices=node_indices, new_layer=new_layer)
    return surgeon.operate()


def delete_channels(model, layer, channels, *, node_indices=None, copy=None):
    """Delete channels from instances of the specified layer.

    This method is designed to facilitate research into pruning networks to
    improve their prediction performance and/or reduce computational load by
    deleting channels.
    All weights associated with the deleted channels in the specified layer
    and any affected downstream layers are deleted.
    If the layer is shared and node_indices is set, channels will be deleted
    from the corresponding layer instances only. This will break the weight
    sharing between affected and unaffected instances in subsequent training.
    In this case affected instances will be renamed.


    Args:
        model: Model object.
        layer: Layer whose channels are to be deleted.
        channels: Indices of the channels to be deleted
        node_indices: Indices of the nodes where channels are to be deleted.
        copy: If True, the model will be copied before and after
              manipulation. This keeps both the old and new models' layers
              clean of each-others data-streams.

    Returns:
        A new Model with the specified channels and associated weights deleted.

    Notes:
        Channels are filters in conv layers and units in other layers.
    """
    surgeon = Surgeon(model, copy)
    surgeon.add_job('delete_channels', layer, node_indices=node_indices, channels=channels)
    return surgeon.operate()