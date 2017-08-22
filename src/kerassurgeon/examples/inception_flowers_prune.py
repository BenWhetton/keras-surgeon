"""Prunes channels from Inception V3 fine tuned on a small flowers data set.

see setup instructions in inception_flowers_tune.py
inception_flowers_tune.py must be run first
"""
import math
import gc

from keras.applications import inception_v3
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import CSVLogger
import pandas as pd
import numpy as np

from kerassurgeon.identify import get_apoz
from kerassurgeon import Surgeon

# dimensions of our images.
img_width, img_height = 299, 299

output_dir = 'inception_flowers/'
train_data_dir = output_dir+'data/train/'
validation_data_dir = output_dir+'data/validation/'
tuned_weights_path = output_dir+'tuned_weights.h5'
epochs = 50
batch_size = 16
val_batch_size = 16
percent_pruning = 2
total_percent_pruning = 20


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

    test_datagen = ImageDataGenerator(
        preprocessing_function=inception_v3.preprocess_input)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=val_batch_size,
        class_mode='categorical')

    # Evaluate the model performance before pruning
    loss = model.evaluate_generator(validation_generator,
                                    validation_generator.n //
                                    validation_generator.batch_size)
    print('original model validation loss: ', loss[0], ', acc: ', loss[1])

    # Incrementally prune the network, retraining it each time
    percent_pruned = 0
    while percent_pruned <= total_percent_pruning:
        # Prune the model
        percent_pruned += percent_pruning
        print('pruning up to ', str(percent_pruned),
              '% of the original model weights')
        apoz_df = get_model_apoz(model, validation_generator)
        if percent_pruned == 0:
            total_channels = apoz_df['apoz'].count()
            n_channels_delete = int(
                math.floor(percent_pruning / 100 * total_channels))
        model = prune_model(model, apoz_df, n_channels_delete)

        # Re-train the model
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])
        checkpoint_name = ('inception_flowers_pruning_' + str(percent_pruned)
                           + 'percent')
        csv_logger = CSVLogger(output_dir + checkpoint_name + '.csv')
        model.fit_generator(train_generator,
                            steps_per_epoch=train_generator.n //
                                            train_generator.batch_size,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=validation_generator.n //
                                             validation_generator.batch_size,
                            workers=4,
                            callbacks=[csv_logger])
        model.save(output_dir + checkpoint_name + '.h5')

    # Evaluate the final model performance
    loss = model.evaluate_generator(validation_generator,
                                    validation_generator.n //
                                    validation_generator.batch_size)
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
