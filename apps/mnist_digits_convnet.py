# Training a convolutional network model for recognizing hand written digits (0-9)

import time

import matplotlib.pyplot as plt
from keras import models, layers
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# To avoid an error (see the method)
from utils.compatibility import compat_no_algo

compat_no_algo()


def showDigit(digit, label):
    """Shows a representation of a single digit"""
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.show()


startTime = time.time()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Show a few first digits
# for i in range(10):
#    showDigit(train_images[i], train_labels[i])

network = models.Sequential()
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPooling2D(2, 2))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D(2, 2))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
print(network.summary())

network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))
print(network.summary())

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# Converts each digit (0-9) into a binary array
# For instance, '2' is represented by [0,0,1,0,0,0,0,0,0,0] ('1' is on index 2, the rest are zeros)
# This is done to make a final representation as an array of probabilities for an image to be any of the 10 digits!
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test loss: ', test_loss)
print('Test accuracy: ', test_acc)
print('It took {} sec'.format(time.time() - startTime))

"""
CPU:
Epoch 5/5
938/938 [==============================] - 27s 28ms/step - loss: 0.0205 - accuracy: 0.9935
313/313 [==============================] - 2s 4ms/step - loss: 0.0247 - accuracy: 0.9922
Test loss:  0.024745531380176544
Test accuracy:  0.9922000169754028
It took 134.6048243045807 sec
"""

"""
GPU:
Epoch 5/5
938/938 [==============================] - 2s 2ms/step - loss: 0.0193 - accuracy: 0.9944
313/313 [==============================] - 0s 963us/step - loss: 0.0309 - accuracy: 0.9908
Test loss:  0.03085430897772312
Test accuracy:  0.9908000230789185
It took 10.864889860153198 sec
"""
