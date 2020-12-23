# Training network to classify movie reviews into positive (1) and negative (0)
# Only first 10,000 most common words are used, the rest are discarded

from keras.datasets import imdb
from keras import models, layers, optimizers, losses, metrics
import numpy as np
import matplotlib.pyplot as plt
from utils.data_preprocessing import vectorize, decodeSequence

NUM_WORDS = 10000


# To resolve an error while loading imdb dataset
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)
# Verify the max number of words
#maxIdx = max( [max(sequence) for sequence in train_data] )
#print(maxIdx)

# Decode a sequence of indices
print( decodeSequence(imdb, train_data[0]), train_labels[0] )

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

#history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=1000, validation_data=(x_val, y_val))
history = model.fit(x_train, y_train, epochs=10, batch_size=1000, validation_data=(x_test, y_test))

hDict = history.history
print(hDict.keys())
loss_values = hDict['loss']
val_loss_values = hDict['val_loss']
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

acc_values = hDict['binary_accuracy']
val_acc_values = hDict['val_binary_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




