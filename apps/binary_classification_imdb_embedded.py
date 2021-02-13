# Training network to classify movie reviews into positive (1) and negative (0)
# Only first 10,000 most common words are used, the rest are discarded
# Using compact embedded vectors to encode words

import os
import pandas as pd
import numpy as np
import logging
import time
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from utils.data_preprocessing import shuffle, embeddings_GloVe
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from utils.plotting import plot_accuracy_loss


def convert_zero_one(val):
    """Converts sentiment from string to 0/1 values"""
    if val == 'negative':
        return 0
    elif val == 'positive':
        return 1
    else:
        logging.error('Unsupported sentiment value {}'.format(val))
        raise


# Training params
maxlen = 10000  # use only that many first words in each review
max_words = 10000  # Only that many top words are used from the dataset
embeddings_dim = 8  # dimensionality of embedding coefficients
train_sz = 20000
valid_sz = 20000
use_pretrained_emb = False

t0 = time.time()
data_dir = os.environ['DATASETS_DIR']
file_path = os.path.join(data_dir, 'imdb', 'IMDB Dataset.csv')

# Read csv file, convert sentiment to 0/1 values
df = pd.read_csv(file_path, converters={'sentiment': convert_zero_one})
reviews = np.array(df['review'])
labels = np.array(df['sentiment'])
print(reviews.shape)
print(labels.shape)

# Tokenize reviews (convert words -> integers)
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews)
reviews = tokenizer.texts_to_sequences(reviews)
reviews = pad_sequences(reviews, maxlen=maxlen, padding='post', truncating='post')  # Only first maxlen words in a review are processed

# Shuffle data
reviews, labels = shuffle(reviews, labels)

# Split into train/validation/test data
idx = train_sz
x_train = reviews[:idx]
y_train = labels[:idx]
idx1 = idx + valid_sz
x_valid = reviews[idx:idx1]
y_valid = labels[idx:idx1]
x_test = reviews[idx1:]
y_test = labels[idx1:]
print('Train: ', x_train.shape)
print('Valid: ', x_valid.shape)
print('Test: ', x_test.shape)

word_index = tokenizer.word_index
print('Found {} unique tokens'.format(len(word_index)))

if use_pretrained_emb:
    word_vectors = embeddings_GloVe(dim=embeddings_dim)  # 6B tokens
    print(len(word_vectors))

    # Build an embeddings matrix (word indes : embedding coefficients)
    word_index = tokenizer.word_index
    embed_matrix = np.zeros((max_words, embeddings_dim))
    for word, i in word_index.items():
        if i < max_words:
            coeffs = word_vectors.get(word)  # attemt getting pretrained embedding coefficients for a word
            if coeffs is not None:
                embed_matrix[i] = coeffs

t1 = time.time()
print('It took {} sec to load data'.format(t1-t0))

# Build a model
model = Sequential()
# Embedding layer is 2D matrix number of words x embedding dimensionality
model.add(Embedding(max_words, embeddings_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
logging.info(model.summary())

# Load pretrained embedded word vectors to Embedded layer (the first layer in the model)
if use_pretrained_emb:
    model.layers[0].set_weights([embed_matrix])
    model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_valid, y_valid))

t2 = time.time()
print('It took {} sec to train the model'.format(t2-t1))

plot_accuracy_loss(
    history.history['acc'],
    history.history['loss'],
    history.history['val_acc'],
    history.history['val_loss'],
)

# Results
"""
GPU, self-trained embeddings, 200 train samples, severy overfitting
Epoch 10/10
1/1 [==============================] - 0s 292ms/step - loss: 0.0310 - acc: 1.0000 - val_loss: 0.7087 - val_acc: 0.5154
It took 4.261695146560669 sec to train the model
"""
"""
GPU, ptetrained GloVe embeddings, 200 training samples. Pretrained embeddings perfor a bit better!
Epoch 10/10
7/7 [==============================] - 0s 49ms/step - loss: 0.0459 - acc: 1.0000 - val_loss: 0.9884 - val_acc: 0.5387
It took 4.550364971160889 sec to train the model
"""
"""
GPU, self-trained embeddings, 20,000 samples
Epoch 10/10
625/625 [==============================] - 2s 3ms/step - loss: 1.9385e-09 - acc: 1.0000 - val_loss: 1.6083 - val_acc: 0.7840
It took 21.027542114257812 sec to train the model
"""
"""
GPU, ptetrained GloVe embeddings, 20,000 samples. Where there are enough samples pre-trained embeddings do not work well!!!
Epoch 10/10
625/625 [==============================] - 1s 2ms/step - loss: 0.1887 - acc: 0.9225 - val_loss: 1.0424 - val_acc: 0.6760
It took 12.788779020309448 sec to train the model
"""

"""
GPU, self-trained embeddings (8d), 20K train samples, 10K words 
Epoch 10/10
625/625 [==============================] - 6s 10ms/step - loss: 0.0372 - acc: 0.9877 - val_loss: 0.6951 - val_acc: 0.8711
It took 64.8416862487793 sec to train the model
"""


