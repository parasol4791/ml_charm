# using pre-trained EfficientNet for image classification
# cats vs dogs

import os
import time
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4
import matplotlib.pyplot as plt
from utils.plotting import plot_accuracy_loss
from utils.compatibility import compat_no_algo
from utils.tf_dataset import skipFiles

compat_no_algo()

# Exclude these imatges
excl_bothCatAndDog = [
    'PetImages/Cat/7194.jpg',
    'PetImages/Cat/11222.jpg',
    'PetImages/Cat/10863.jpg',
    'PetImages/Cat/2159.jpg',
    'PetImages/Cat/9444.jpg',
    'PetImages/Cat/11724.jpg',
    'PetImages/Cat/7920.jpg',
    'PetImages/Cat/3731.jpg',
    'PetImages/Cat/724.jpg',
    'PetImages/Cat/10266.jpg',
    'PetImages/Cat/5355.jpg',
    'PetImages/Cat/1450.jpg',
    'PetImages/Cat/3822.jpg',
    'PetImages/Cat/5583.jpg',
    'PetImages/Cat/9250.jpg',
    'PetImages/Cat/7575.jpg',
]
excl_noCatOrDog = [
    'PetImages/Cat/5351.jpg',
    'PetImages/Dog/8736.jpg',
    'PetImages/Cat/10712.jpg',
    'PetImages/Cat/9882.jpg',
    'PetImages/Dog/10801.jpg',
]
excl_canHardToTell = [
    'PetImages/Cat/5324.jpg',
    'PetImages/Cat/3672.jpg',
    'PetImages/Dog/8444.jpg',
    'PetImages/Cat/6402.jpg',
    'PetImages/Cat/8138.jpg',
    'PetImages/Cat/2150.jpg',
]
excl_misclassified = [
    'PetImages/Dog/11731.jpg',
    'PetImages/Dog/4334.jpg',
    'PetImages/Dog/7164.jpg',
]
excl_all = excl_bothCatAndDog + excl_noCatOrDog + excl_canHardToTell + excl_misclassified
print('Excluding {} samples'.format(len(excl_all)))


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
dataDir = os.environ['DATASETS_DIR']
outputs_dir = os.environ['OUTPUTS_DIR']
# Sets in %
trainSz = 60
validSz = 20
testSz = 20

data_path = os.path.join(dataDir, dataset_name)
tfds.load(dataset_name, data_dir=data_path)

# Slicing datasets: https://www.tensorflow.org/datasets/splits
trainValidSz = trainSz + validSz
allSz = trainValidSz + testSz
if allSz > 100:
    raise ValueError('Sum of train, validation, and test sets should be <= 100%, it is {}%'.format(allSz))
trainStr = 'train[:{}%]'.format(trainSz)
validStr = 'train[{}%:{}%]'.format(trainSz, trainValidSz)
testStr = 'train[{}%:{}%]'.format(trainValidSz, allSz)
(ds_train, ds_valid, ds_test), ds_info = tfds.load(dataset_name, split=[trainStr, validStr, testStr], with_info=True, as_supervised=True, shuffle_files=True)
print('Training set: {}'.format(tf.data.experimental.cardinality(ds_train).numpy()))
print('Validation set: {}'.format(tf.data.experimental.cardinality(ds_valid).numpy()))
print('Test set: {}'.format(tf.data.experimental.cardinality(ds_test).numpy()))
NUM_CLASSES = ds_info.features["label"].num_classes

t1 = time.time()

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_valid = ds_valid.map(lambda image, label: (tf.image.resize(image, size), label))
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

ds_valid = ds_valid.map(input_preprocess)
ds_valid = ds_valid.batch(batch_size=batch_size, drop_remainder=True)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

# Extract features
modelName_classifier = 'cats_n_dogs_efficientnet_b4_classifier.h5'
if 0:
    model = build_model(num_classes=NUM_CLASSES)
    print(model.summary())
    epochs = 15  # @param {type: "slider", min:8, max:80}
    history = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_valid, validation_batch_size=batch_size) #verbose=2

    model.save(os.path.join(outputs_dir, modelName_classifier))

    t2 = time.time()
    print('It took {} sec to train the model'.format(t2 - t1))

    plt.clf()
    plot_accuracy_loss(
        history.history['accuracy'],
        history.history['loss'],
        history.history['val_accuracy'],
        history.history['val_loss'],
    )
else:
    modelPath = os.path.join(outputs_dir, modelName_classifier)
    model = tf.keras.models.load_model(modelPath, custom_objects=None, compile=True, options=None)


# Fine-tuning
modelName_FineTuned = 'cats_n_dogs_efficientnet_b4_fine-tuned.h5'
if 0:
    t3 = time.time()
    model = unfreeze_model(model)
    print(model.summary())
    epochs = 3  # @param {type: "slider", min:8, max:50}
    history = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_valid, validation_batch_size=batch_size) #verbose=2

    model.save(os.path.join(outputs_dir, modelName_FineTuned))

    t4 = time.time()
    print('It took {} sec to train the model'.format(t4 - t3))

    plt.clf()
    plot_accuracy_loss(
        history.history['accuracy'],
        history.history['loss'],
        history.history['val_accuracy'],
        history.history['val_loss'],
    )
else:
    modelPath = os.path.join(outputs_dir, modelName_FineTuned)
    model = tf.keras.models.load_model(modelPath, custom_objects=None, compile=True, options=None)


# Test evaluation
if 0:
    res = model.evaluate(ds_test, batch_size=batch_size)
    print('Test loss, accuracy: {}'.format(res))

ds = tfds.load(dataset_name, shuffle_files=False, batch_size=512)
"""ds = ds['train']
images = np.concatenate([x['image'] for x in ds], axis=0)
fNames = np.concatenate([x['image/filename'] for x in ds], axis=0)
labels = np.concatenate([x['label'] for x in ds], axis=0)"""
t1 = time.time()
err = 0
fNames = []
for x in ds['train']:
    img = tf.image.resize(x['image'], size)
    pred = model.predict(img)
    pred1 = pred[:,1]
    label = x['label'].numpy()
    for pr, lbl, fn in zip(pred1, label, x['image/filename']):
        predDig = (pr > 0.5)
        if predDig != lbl:
            fName = fn.numpy().decode()
            fNames.append(fName)
            print(fName, lbl, pr)
            err += 1
print(err)
t2 = time.time()
print('It took {} sec to train the model'.format(t2 - t1))

for fn in fNames:
    print('\'{}\','.format(fn))

"""images = []
labels = []
for img, lbl in ds_test:
    if images == []:
        images = img
        labels = lbl
    else:
        images = np.concatenate([images, img], axis=0)
        labels = np.concatenate([labels, lbl], axis=0)
#labels = np.concatenate([lbl for _, lbl in ds_valid], axis=0)
#pred = model.predict(ds_test)
n = 4600
pred = model.predict(x=images[:n], batch_size=batch_size)
#diffs = labels[:,0] - pred[:,0]
print('Total test: {}'.format(len(pred)))
nErr = 0
plt.clf()
plt.show()
for i, (img, y, h) in enumerate(zip(images[:n], labels[:n], pred)):
    y = 1 if y[0] > 0.5 else 0
    h = 1 if h[0] > 0.5 else 0
    if h != y:
        nErr += 1
        plt.imshow(img.astype("uint8"))
        print(i)
        #print(img['image/filename'].numpy().decode())
print('Errors: {}'.format(nErr))"""


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