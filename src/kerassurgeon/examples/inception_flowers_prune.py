"""Prunes channels from Inception V3 fine tuned on a small flowers data set.

see setup instructions in inception_flowers_tune.py
inception_flowers_tune.py must be run first
"""
import math

from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import CSVLogger
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import tensorflow as tf

from kerassurgeon.identify import get_apoz
from kerassurgeon import Surgeon



# dimensions of our images.
img_width, img_height = 299, 299

output_dir = '.inception_flowers_data/'
train_data_dir = output_dir+'data/train/'
validation_data_dir = output_dir+'data/validation/'
tuned_weights_path = output_dir+'tuned_weights.h5'
epochs = 15
batch_size = 16
val_batch_size = 16
percent_pruning = 2
total_percent_pruning = 50


def iterative_prune_model():
    # build the inception v3 network
    base_model = inception_v3.InceptionV3(include_top=False,
                                          weights='imagenet',
                                          pooling='avg',
                                          input_shape=(299, 299, 3))
    print('Model loaded.')

    top_output = Dense(5, activation='softmax')(base_model.output)

    # add the model on top of the convolutional base
    model = Model(base_model.inputs, top_output)
    del base_model
    model.load_weights(tuned_weights_path)
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # Set up data generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=inception_v3.preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    train_steps = train_generator.n // train_generator.batch_size

    test_datagen = ImageDataGenerator(
        preprocessing_function=inception_v3.preprocess_input)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=val_batch_size,
        class_mode='categorical')
    val_steps = validation_generator.n // validation_generator.batch_size

    # Evaluate the model performance before pruning
    loss = model.evaluate(validation_generator)
    print('original model validation loss: ', loss[0], ', acc: ', loss[1])

    total_channels = get_total_channels(model)
    n_channels_delete = int(math.floor(percent_pruning / 100 * total_channels))

    # Incrementally prune the network, retraining it each time
    percent_pruned = 0
    # If percent_pruned > 0, continue pruning from previous checkpoint
    if percent_pruned > 0:
        checkpoint_name = ('inception_flowers_pruning_' + str(percent_pruned)
                           + 'percent')
        model = load_model(output_dir + checkpoint_name + '.h5')

    while percent_pruned <= total_percent_pruning:
        # Prune the model
        apoz_df = get_model_apoz(model, validation_generator)
        percent_pruned += percent_pruning
        print('pruning up to ', str(percent_pruned),
              '% of the original model weights')
        model = prune_model(model, apoz_df, n_channels_delete)

        # Clean up tensorflow session after pruning and re-load model
        checkpoint_name = ('inception_flowers_pruning_' + str(percent_pruned)
                           + 'percent')
        model.save(output_dir + checkpoint_name + '.h5')
        del model
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        model = load_model(output_dir + checkpoint_name + '.h5')

        # Re-train the model
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])
        checkpoint_name = ('inception_flowers_pruning_' + str(percent_pruned)
                           + 'percent')
        csv_logger = CSVLogger(output_dir + checkpoint_name + '.csv')
        model.fit(train_generator,
                  epochs=epochs,
                  validation_data=validation_generator,
                  workers=4,
                  callbacks=[csv_logger])

    # Evaluate the final model performance
    loss = model.evaluate(validation_generator)
    print('pruned model loss: ', loss[0], ', acc: ', loss[1])


def prune_model(model, apoz_df, n_channels_delete):
    # Identify 5% of channels with the highest APoZ in model
    sorted_apoz_df = apoz_df.sort_values('apoz', ascending=False)
    high_apoz_index = sorted_apoz_df.iloc[0:n_channels_delete, :]

    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = Surgeon(model, copy=True)
    for name in high_apoz_index.index.unique().values:
        channels = list(pd.Series(high_apoz_index.loc[name, 'index'],
                                  dtype=np.int64).values)
        surgeon.add_job('delete_channels', model.get_layer(name),
                        channels=channels)
    # Delete channels
    return surgeon.operate()


def get_total_channels(model):
    start = None
    end = None
    channels = 0
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels


def get_model_apoz(model, generator):
    # Get APoZ
    start = None
    end = None
    apoz = []
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            print(layer.name)
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df


if __name__ == '__main__':
    iterative_prune_model()
