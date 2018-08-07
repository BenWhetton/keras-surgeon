"""

Before running this script flowers data set must be downloaded either using the 
following linux terminal commands:
```
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```
or by downloading directly from the hyperlink above and copying the files to a
directory of your choice.

The photos must be split into a training and validation set with the directory
structure as follows:
- create a flowers-data/ folder
- create train/ and validation/ subfolders inside flowers-data/
- copy the daisy/, dandelion/, roses/, sunflowers/ and tulips/ directories from
    the downloaded archive into the train/ directory
- create daisy/, dandelion/, roses/, sunflowers/ and tulips/ directories inside
    validation/
- move 100 photos from each category from their subfolders inside train/ to the
    corresponding subfolders inside validation/
There should be a total of 3170 training samples and 500 validation samples
"""

import numpy as np
from keras.applications import inception_v3
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import CSVLogger


# dimensions of the images.
img_width, img_height = 299, 299

output_dir = 'inception_flowers/'
train_data_dir = output_dir+'data/train/'
validation_data_dir = output_dir+'data/validation/'
top_model_weights_path = output_dir+'top_model_weights.h5'
tuned_weights_path = output_dir+'tuned_weights.h5'
nb_train_samples = 3170
nb_validation_samples = 500
top_epochs = 200
tune_epochs = 50
batch_size = 16


def save_bottleneck_features():
    # build the Inception V3 network
    model = inception_v3.InceptionV3(include_top=False,
                                     weights='imagenet',
                                     input_tensor=None,
                                     input_shape=None,
                                     pooling='avg')

    # Save the bottleneck features for the training data set
    datagen = ImageDataGenerator(preprocessing_function=
                                 inception_v3.preprocess_input)
    generator = datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='sparse',
                                            shuffle=False)
    features = model.predict_generator(generator, nb_train_samples // batch_size)
    labels = np.eye(generator.num_classes, dtype='uint8')[generator.classes]
    labels = labels[0:(nb_train_samples // batch_size) * batch_size]
    np.save(open(output_dir+'bottleneck_features_train.npy', 'wb'), features)
    np.save(open(output_dir+'bottleneck_labels_train.npy', 'wb'), labels)

    # Save the bottleneck features for the validation data set
    generator = datagen.flow_from_directory(validation_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode=None,
                                            shuffle=False)
    features = model.predict_generator(generator, nb_validation_samples // batch_size)
    labels = np.eye(generator.num_classes, dtype='uint8')[generator.classes]
    labels = labels[0:(nb_validation_samples // batch_size) * batch_size]
    np.save(open(output_dir+'bottleneck_features_validation.npy', 'wb'), features)
    np.save(open(output_dir+'bottleneck_labels_validation.npy', 'wb'), labels)


def train_top_model():
    # Load the bottleneck features and labels
    train_features = np.load(open(output_dir+'bottleneck_features_train.npy', 'rb'))
    train_labels = np.load(open(output_dir+'bottleneck_labels_train.npy', 'rb'))
    validation_features = np.load(open(output_dir+'bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.load(open(output_dir+'bottleneck_labels_validation.npy', 'rb'))

    # Create the top model for the inception V3 network, a single Dense layer
    # with softmax activation.
    top_input = Input(shape=train_features.shape[1:])
    top_output = Dense(5, activation='softmax')(top_input)
    model = Model(top_input, top_output)

    # Train the model using the bottleneck features and save the weights.
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    csv_logger = CSVLogger(output_dir + 'top_model_training.csv')
    model.fit(train_features, train_labels,
              epochs=top_epochs,
              batch_size=batch_size,
              validation_data=(validation_features, validation_labels),
              callbacks=[csv_logger])
    model.save_weights(top_model_weights_path)


def tune_model():
    # Build the Inception V3 network.
    base_model = inception_v3.InceptionV3(include_top=False,
                                          weights='imagenet',
                                          pooling='avg')
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_input = Input(shape=base_model.output_shape[1:])
    top_output = Dense(5, activation='softmax')(top_input)
    top_model = Model(top_input, top_output)

    # Note that it is necessary to start with a fully-trained classifier,
    # including the top classifier, in order to successfully do fine-tuning.
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model = Model(inputs=base_model.inputs,
                  outputs=top_model(base_model.outputs))

    # Set all layers up to 'mixed8' to non-trainable (weights will not be updated)
    last_train_layer = model.get_layer(name='mixed8')
    for layer in model.layers[:model.layers.index(last_train_layer)]:
        layer.trainable = False

    # Compile the model with a SGD/momentum optimizer and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # Prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        preprocessing_function=inception_v3.preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        preprocessing_function=inception_v3.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    loss = model.evaluate_generator(validation_generator,
                                    nb_validation_samples//batch_size)
    print('Model validation performance before fine-tuning:', loss)

    csv_logger = CSVLogger(output_dir+'model_tuning.csv')
    # fine-tune the model
    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples//batch_size,
                        epochs=tune_epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples//batch_size,
                        workers=4,
                        callbacks=[csv_logger])
    model.save_weights(tuned_weights_path)


if __name__ == '__main__':
    save_bottleneck_features()
    train_top_model()
    tune_model()
