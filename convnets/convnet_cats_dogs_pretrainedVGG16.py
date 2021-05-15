# Binary classification of images 'dogs' vs 'cats'
# Uses convolution base of pre-trained CNN model VGG16 with a custom classifier (feature extraction).
# No image augmentation is done!

import os
import time

import numpy as np
from keras import models, layers, optimizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

# To avoid an error (see the method)
from utils.compatibility import compat_no_algo
from utils.plotting import plot_accuracy_loss

compat_no_algo()

conv_base = VGG16(
    weights='imagenet',
    include_top=False,  # original top 'Dense' layers used for classification are excluded
    input_shape=(150, 150, 3)
)

# Data folders
base_dir = os.environ['DATASETS_DIR']
base_dir = os.path.join(base_dir, 'dogs_vs_cats_small')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validate')
test_dir = os.path.join(base_dir, 'test')

dataGen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20
train_size = 2000
valid_size = 1000
test_size = 1000


def extract_features(directory, sample_count, batch_sz):
    """Extracts features from a pre-trained model"""
    features = np.zeros(shape=(sample_count, 4, 4, 512))  # Last conv layer of VGG16 has shape of (4,4,512)
    labels = np.zeros(sample_count)
    generator = dataGen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_sz,
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        feature_batch = conv_base.predict(inputs_batch)
        features[i * batch_sz: (i + 1) * batch_sz] = feature_batch
        labels[i * batch_sz: (i + 1) * batch_sz] = labels_batch
        i += 1
        if i * batch_sz >= sample_count:
            break
    return features, labels


startTime = time.time()

# Extract features
train_features, train_labels = extract_features(train_dir, train_size, batch_size)
valid_features, valid_labels = extract_features(validation_dir, valid_size, batch_size)
test_features, test_labels = extract_features(test_dir, test_size, batch_size)

# Flatten features
convShape = 4 * 4 * 512
train_features = np.reshape(train_features, (train_size, convShape))
valid_features = np.reshape(valid_features, (valid_size, convShape))
test_features = np.resize(test_features, (test_size, convShape))

# Custom classifier
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=convShape))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=batch_size,
                    validation_data=(valid_features, valid_labels)
                    )

print('It took {} sec'.format(time.time() - startTime))

# Plotting accuracy/loss functions
plot_accuracy_loss(
    history.history['acc'],
    history.history['loss'],
    history.history['val_acc'],
    history.history['val_loss'],
)

# Results
"""
GPU:
Epoch 30/30
100/100 [==============================] - 0s 2ms/step - loss: 0.0834 - acc: 0.9744 - val_loss: 0.2406 - val_acc: 0.9040
It took 17.638522148132324 sec
"""
