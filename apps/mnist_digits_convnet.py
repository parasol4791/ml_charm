
# Training a convolutional network model for recognizing hand written digits (0-9)

from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def showDigit(digit, label):
    '''Shows a representation of a single digit'''
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.show()


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Show a few first digits
#for i in range(10):
#    showDigit(train_images[i], train_labels[i])

network = models.Sequential()
network.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
network.add(layers.MaxPooling2D(2,2))
network.add(layers.Conv2D(64, (3,3), activation='relu'))
network.add(layers.MaxPooling2D(2,2))
network.add(layers.Conv2D(64, (3,3), activation='relu'))
print(network.summary())

network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))
print(network.summary())

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape( (60000, 28, 28, 1) )
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape( (10000, 28, 28, 1) )
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

# Results
""" ...
Epoch 5/5
60000/60000 [==============================] - 34s 567us/step - loss: 0.0197 - acc: 0.9939
10000/10000 [==============================] - 2s 171us/step
Test loss:  0.02593975819397747
Test accuracy:  0.9916"""