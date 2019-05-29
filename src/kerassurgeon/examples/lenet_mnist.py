from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.examples.tutorials.mnist import input_data

from kerassurgeon import identify
from kerassurgeon.operations import delete_channels


def main():
    training_verbosity = 2
    # Download data if needed and import.
    mnist = input_data.read_data_sets('tempData', one_hot=True, reshape=False)
    val_images = mnist.validation.images
    val_labels = mnist.validation.labels

    # Create LeNet model
    model = Sequential()
    model.add(Conv2D(20,
                     [3, 3],
                     input_shape=[28, 28, 1],
                     activation='relu',
                     name='conv_1'))
    model.add(MaxPool2D())
    model.add(Conv2D(50, [3, 3], activation='relu', name='conv_2'))
    model.add(MaxPool2D())
    model.add(layers.Permute((2, 1, 3)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu', name='dense_1'))
    model.add(Dense(10, activation='softmax', name='dense_2'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0,
                                             patience=10,
                                             verbose=training_verbosity,
                                             mode='auto')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=5,
                                            verbose=training_verbosity,
                                            mode='auto',
                                            epsilon=0.0001,
                                            cooldown=0,
                                            min_lr=0)

    # Train LeNet on MNIST
    results = model.fit(mnist.train.images,
                        mnist.train.labels,
                        epochs=200,
                        batch_size=128,
                        verbose=2,
                        validation_data=(val_images, val_labels),
                        callbacks=[early_stopping, reduce_lr])

    loss = model.evaluate(val_images, val_labels, batch_size=128, verbose=2)
    print('original model loss:', loss, '\n')

    layer_name = 'dense_1'

    while True:
        layer = model.get_layer(name=layer_name)
        apoz = identify.get_apoz(model, layer, val_images)
        high_apoz_channels = identify.high_apoz(apoz)
        model = delete_channels(model, layer, high_apoz_channels)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        loss = model.evaluate(val_images,
                              val_labels,
                              batch_size=128,
                              verbose=2)
        print('model loss after pruning: ', loss, '\n')

        results = model.fit(mnist.train.images,
                            mnist.train.labels,
                            epochs=200,
                            batch_size=128,
                            verbose=training_verbosity,
                            validation_data=(val_images, val_labels),
                            callbacks=[early_stopping, reduce_lr])

        loss = model.evaluate(val_images,
                              val_labels,
                              batch_size=128,
                              verbose=2)
        print('model loss after retraining: ', loss, '\n')


if __name__ == '__main__':
    main()

