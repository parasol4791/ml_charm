# Training a model for recognizing hand written digits (0-9)

from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import time


def showDigit(digit, label):
    '''Shows a representation of a single digit'''
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.show()


startTime = time.time()
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Show a few first digits
#for i in range(10):
#    showDigit(train_images[i], train_labels[i])

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape( (60000, 28*28) )
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape( (10000, 28*28) )
test_images = test_images.astype('float32') / 255

# Converts each digit (0-9) into a binary array
# For instance, '2' is represented by [0,0,1,0,0,0,0,0,0,0] ('1' is on index 2, the rest are zeros)
# This is done to make a final representation as an array of probabilities for an image to be any of the 10 digits!
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test loss: ', test_loss)
print('Test accuracy: ', test_acc)
print('It took {} sec'.format(time.time() -  startTime))

# Results
""" ...
Epoch 5/5
60000/60000 [==============================] - 4s 64us/step - loss: 0.0377 - acc: 0.9888
10000/10000 [==============================] - 0s 38us/step
Test loss:  0.07058643732223427
Test accuracy:  0.979
"""

"""
GPU:
Epoch 5/5
469/469 [==============================] - 0s 896us/step - loss: 0.0359 - accuracy: 0.9893
313/313 [==============================] - 0s 668us/step - loss: 0.0636 - accuracy: 0.9808
Test loss:  0.06355077028274536
Test accuracy:  0.9807999730110168
It took 3.6985509395599365 sec
"""