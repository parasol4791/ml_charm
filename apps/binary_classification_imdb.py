# Training network to classify movie reviews into positive (1) and negative (0)
# Only first 10,000 most common words are used, the rest are discarded
# Using one-hot encoding for words

import time

import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers, losses, metrics
from keras.datasets import imdb

from utils.data_preprocessing import vectorize, decodeSequence
from utils.plotting import plot_accuracy_loss

NUM_WORDS = 10000

# To resolve an error while loading imdb dataset
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

tf.config.optimizer.set_jit(False)  # Start with XLA enabled

startTime = time.time()
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)
# Verify the max number of words
# maxIdx = max( [max(sequence) for sequence in train_data] )
# print(maxIdx)

# Decode a sequence of indices
print(decodeSequence(imdb, train_data[0]), train_labels[0])

x_train = vectorize(train_data, NUM_WORDS)
x_test = vectorize(test_data, NUM_WORDS)

y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(NUM_WORDS,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# WIth customized optimizer
model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)

# Validation on training set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=1000, validation_data=(x_val, y_val))
history = model.fit(x_train, y_train, epochs=10, batch_size=1000, validation_data=(x_test, y_test))

print('It took {} sec'.format(time.time() - startTime))

plot_accuracy_loss(
    history.history['binary_accuracy'],
    history.history['loss'],
    history.history['val_binary_accuracy'],
    history.history['val_loss'],
)

"""
CPU:
Epoch 10/10
25/25 [==============================] - 1s 26ms/step - loss: 0.1271 - binary_accuracy: 0.9614 - val_loss: 0.3040 - val_binary_accuracy: 0.8803
It took 15.316411256790161 sec
"""

"""
GPU:
Epoch 10/10
25/25 [==============================] - 0s 20ms/step - loss: 0.1289 - binary_accuracy: 0.9584 - val_loss: 0.3073 - val_binary_accuracy: 0.8797
It took 11.001086711883545 sec
"""
