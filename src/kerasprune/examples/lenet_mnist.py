# import keras
#
# from kerasprune.identify import high_apoz
# from kerasprune.prune import delete_channels
#
#
# def apoz_example(model, layer_index, x_val, y_val):
#     # Identify the neurons with a high percentage of zero activations on the data set x_val
#     [high_apoz_neurons, apoz_data] = high_apoz(model, layer_index, x_val)
#
#     new_model = delete_channels(model, layer_index, high_apoz_neurons)
#
#     return new_model, apoz_data

import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from tensorflow.examples.tutorials.mnist import input_data

from kerasprune.identify import high_apoz
from kerasprune.prune import delete_channels


# import os

def main():
    # Download data if needed and import.
    mnist = input_data.read_data_sets('tempData', one_hot=True, reshape=False)
    # Create LeNet model
    model = Sequential()
    model.add(Conv2D(20, [3, 3], input_shape=[28, 28, 1], data_format='channels_last', activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(50, [3, 3], activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    # Train LeNet on MNIST
    results = model.fit(mnist.train.images,
                        mnist.train.labels,
                        epochs=200,
                        batch_size=128,
                        verbose=2,
                        validation_data=(mnist.validation.images,
                                         mnist.validation.labels
                                         ),
                        callbacks=[early_stopping, reduce_lr_on_plateau]
                        )

    loss = model.evaluate(mnist.validation.images, mnist.validation.labels, batch_size=128, verbose=2)
    print('original model loss:', loss, '\n')

    layer_index = 5

    while True:
        [high_apoz_channels, apoz] = high_apoz(model, layer_index, mnist.validation.images)
        model = delete_channels(model, model.layers[layer_index], high_apoz_channels)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        loss = model.evaluate(mnist.validation.images, mnist.validation.labels, batch_size=128, verbose=2)
        print('model loss after pruning: ', loss, '\n')

        results = model.fit(mnist.train.images,
                            mnist.train.labels,
                            epochs=200,
                            batch_size=128,
                            verbose=2,
                            validation_data=(mnist.validation.images,
                                             mnist.validation.labels
                                             ),
                            callbacks=[early_stopping, reduce_lr_on_plateau]
                            )

        loss = model.evaluate(mnist.validation.images, mnist.validation.labels, batch_size=128, verbose=2)
        print('model loss after retraining: ', loss, '\n')

main()