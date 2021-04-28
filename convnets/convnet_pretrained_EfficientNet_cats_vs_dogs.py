# using pre-trained EfficientNet for image classification
# 120 'stanford' dog breeds

import os
import time
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4
import matplotlib.pyplot as plt
from utils.plotting import plot_accuracy_loss
from utils.compatibility import compat_no_algo

compat_no_algo()


def format_label(label):
    return label_info.int2str(label)


def input_preprocess(image, label):
    """One-hot / categorical encoding"""
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB4(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


#IMG_SIZE = 224  # determined by EfficientNet model choice (B0)
IMG_SIZE = 380  # determined by EfficientNet model choice (B4)
batch_size = 64
dataset_name = "cats_vs_dogs"
trainSz = 15000
testSz = 7000

dataDir = os.environ['DATASETS_DIR']
data_path = os.path.join(dataDir, dataset_name)

tfds.load(dataset_name, data_dir=data_path)
# Slicing datasets: https://www.tensorflow.org/datasets/splits
trainStr = 'train[:{}]'.format(trainSz)
testStr = 'train[:{}]'.format(testSz)
(ds_train, ds_test), ds_info = tfds.load(dataset_name, split=[trainStr, testStr], with_info=True, as_supervised=True, shuffle_files=True)
NUM_CLASSES = ds_info.features["label"].num_classes

t1 = time.time()

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

label_info = ds_info.features["label"]
for i, (image, label) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("{}".format(format_label(label)))
    plt.axis("off")

img_augmentation = Sequential(
    [
        #preprocessing.Rescaling(1./255),
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

for image, label in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(tf.expand_dims(image, axis=0))
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("{}".format(format_label(label)))
        plt.axis("off")


ds_train = ds_train.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

model = build_model(num_classes=NUM_CLASSES)
print(model.summary())
epochs = 20  # @param {type: "slider", min:8, max:80}
history = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_test, validation_batch_size=batch_size) #verbose=2

t2 = time.time()
print('It took {} sec to train the model'.format(t2 - t1))

plt.clf()
plot_accuracy_loss(
    history.history['accuracy'],
    history.history['loss'],
    history.history['val_accuracy'],
    history.history['val_loss'],
)

t3 = time.time()

# Fine-tuning
model = unfreeze_model(model)
print(model.summary())
epochs = 30  # @param {type: "slider", min:8, max:50}
history = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_test, validation_batch_size=batch_size) #verbose=2

t4 = time.time()
print('It took {} sec to train the model'.format(t4 - t3))

plt.clf()
plot_accuracy_loss(
    history.history['accuracy'],
    history.history['loss'],
    history.history['val_accuracy'],
    history.history['val_loss'],
)

"""
Trained classified for catts vs dogs (2K train, 1K validation).
Epoch 18/20
19/31 [=================>............] - ETA: 0s - loss: 0.2874 - accuracy: 0.9358Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
31/31 [==============================] - 2s 65ms/step - loss: 0.2801 - accuracy: 0.9367 - val_loss: 0.0794 - val_accuracy: 0.9833
"""

"""
Fine-tuning - unfreezing top 20 layers of EfficientNetB0
Epoch 24/30
18/31 [================>.............] - ETA: 0s - loss: 0.0850 - accuracy: 0.9748Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
31/31 [==============================] - 2s 69ms/step - loss: 0.0817 - accuracy: 0.9760 - val_loss: 0.0725 - val_accuracy: 0.9865
"""

"""
15K training set, 7K validation set
Feature extraction:
Epoch 3/20
19/31 [=================>............] - ETA: 0s - loss: 0.2380 - accuracy: 0.9419Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
31/31 [==============================] - 2s 65ms/step - loss: 0.2350 - accuracy: 0.9417 - val_loss: 0.0312 - val_accuracy: 0.9917

Fine-turing:
Epoch 29/30
18/31 [================>.............] - ETA: 0s - loss: 0.0372 - accuracy: 0.9881Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
31/31 [==============================] - 2s 69ms/step - loss: 0.0401 - accuracy: 0.9871 - val_loss: 0.0799 - val_accuracy: 0.9865
"""

"""
EfficientNet B4 (17M params). 15K training set, 7K validation set
Feature exytraction:
Epoch 13/20
234/234 [==============================] - 93s 397ms/step - loss: 0.1211 - accuracy: 0.9671 - val_loss: 0.0173 - val_accuracy: 0.9963
Fine-tuning:
Epoch 30/30
234/234 [==============================] - 95s 405ms/step - loss: 0.0138 - accuracy: 0.9956 - val_loss: 0.0018 - val_accuracy: 0.9994
"""