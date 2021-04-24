# Single label, multi-class classification

import time

import numpy as np
from keras import models, layers
from keras.datasets import reuters
from keras.utils import to_categorical

from utils.data_preprocessing import decodeSequence, vectorize
from utils.plotting import plot_accuracy_loss

# To resolve an error while loading imdb dataset
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

startTime = time.time()
NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=NUM_WORDS)
print(len(train_data))
print(len(test_data))
print(decodeSequence(reuters, train_data[0]))

# Vectorize input data
x_train = vectorize(train_data, NUM_WORDS)
x_test = vectorize(test_data, NUM_WORDS)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Split validation data
x_val = x_train[:1000]
y_val = y_train[:1000]

x_train_part = x_train[1000:]
y_train_part = y_train[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_part, y_train_part, epochs=10, batch_size=512, validation_data=(x_val, y_val))

print('It took {} sec'.format(time.time() - startTime))

plot_accuracy_loss(
    history.history['accuracy'],
    history.history['loss'],
    history.history['val_accuracy'],
    history.history['val_loss'],
)

# Predict test categories
matches = 0
predictions = model.predict(x_test)
print('Max prob   Pred categ  Label category')
for pred, label in zip(predictions, test_labels):
    predCat = np.argmax(pred)
    if predCat == label:
        matches += 1
    print(max(pred), predCat, label)
print('{} out of {} matches. Match rate is {}'.format(matches, len(test_labels), matches / len(test_labels)))

"""
CPU:
Epoch 10/10
16/16 [==============================] - 0s 16ms/step - loss: 0.2495 - accuracy: 0.9446 - val_loss: 0.8983 - val_accuracy: 0.8180
It took 5.0334391593933105 sec
"""

"""
GPU:
Epoch 10/10
16/16 [==============================] - 0s 6ms/step - loss: 0.2291 - accuracy: 0.9469 - val_loss: 0.8989 - val_accuracy: 0.8200
It took 3.0891404151916504 sec
"""
