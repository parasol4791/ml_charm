# Training network to classify movie reviews into positive (1) and negative (0)
# LSTM RNN model is used
# Only first 10,000 most common words are used, the rest are discarded
# Using self-trained embeddings for words encoding

import time

import numpy as np
from keras import models, layers, regularizers
from keras.datasets import imdb
from keras.preprocessing import sequence

from utils.data_preprocessing import decodeSequence
from utils.plotting import plot_accuracy_loss
from utils.compatibility import compat_lstm_cancel_error

NUM_WORDS = 20000
maxlen = 1000  # maximum lenght of words in a review to be processed

compat_lstm_cancel_error()

startTime = time.time()
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)
# Verify the max number of words
# maxIdx = max( [max(sequence) for sequence in train_data] )
# print(maxIdx)

# Decode a sequence of indices
print(decodeSequence(imdb, train_data[0]), train_labels[0])

x_train = sequence.pad_sequences(train_data, maxlen=maxlen)
x_test = sequence.pad_sequences(test_data, maxlen=maxlen)

y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Embedding(NUM_WORDS, 32))
model.add(layers.LSTM(32, dropout=0.5))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))

print('It took {} sec'.format(time.time() - startTime))

plot_accuracy_loss(
    history.history['acc'],
    history.history['loss'],
    history.history['val_acc'],
    history.history['val_loss'],
)

# Results:
"""
GPU:
Epoch 3/3
196/196 [==============================] - 9s 44ms/step - loss: 0.2370 - acc: 0.9118 - val_loss: 0.2825 - val_acc: 0.8821
It took 32.71308493614197 sec
"""