# using pre-trained EfficientNet for image classification
# 120 'stanford' dog breeds

import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from utils.plotting import plot_accuracy_loss
from utils.compatibility import compat_no_algo

compat_no_algo()


def format_label(label):
    string_label = label_info.int2str(label)
    return string_label.split("-")[1]


def input_preprocess(image, label):
    """One-hot / categorical encoding"""
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

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


IMG_SIZE = 224  # determined by EfficientNet model choice (B0)
batch_size = 64
dataset_name = "stanford_dogs"

dataDir = os.environ['DATASETS_DIR']
data_path = os.path.join(dataDir, dataset_name)

tfds.load(dataset_name, data_dir=data_path)
(ds_train, ds_test), ds_info = tfds.load(dataset_name, split=["train", "test"], with_info=True, as_supervised=True, shuffle_files=True)
NUM_CLASSES = ds_info.features["label"].num_classes

startTime = time.time()

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
epochs = 7  # @param {type: "slider", min:8, max:80}
history = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_test, validation_batch_size=batch_size) #verbose=2

plt.clf()
plot_accuracy_loss(
    history.history['accuracy'],
    history.history['loss'],
    history.history['val_accuracy'],
    history.history['val_loss'],
)

# Fine-tuning
model = unfreeze_model(model)
print(model.summary())
epochs = 10  # @param {type: "slider", min:8, max:50}
history = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_test, validation_batch_size=batch_size) #verbose=2

plt.clf()
plot_accuracy_loss(
    history.history['accuracy'],
    history.history['loss'],
    history.history['val_accuracy'],
    history.history['val_loss'],
)

"""
Trained classified for 120 dog breed classes.
Epoch 21/30
187/187 [==============================] - 13s 72ms/step - loss: 1.2826 - accuracy: 0.6471 - val_loss: 0.8111 - val_accuracy: 0.7732
"""

"""
Fine-tuning - unfreezing top 20 layers of EfficientNetB0
Epoch 2/10
187/187 [==============================] - 14s 75ms/step - loss: 1.0146 - accuracy: 0.6940 - val_loss: 0.6711 - val_accuracy: 0.7993
"""
