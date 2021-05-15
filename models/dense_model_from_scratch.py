# Simple Sequential model with a single Dense hidden layer
# for a binary classification - implemented from scratch
# using schikit-learn and numpy
# Generating a dataset from Gaussian quantiles - 2 features X1, X2 represent (x1,x2) coordinates
# in cartesian space
# Y represents 'red' dot (0), or 'blue' dot (1).
# Classification objecitve is to train NN to predict dot type, based on its coordinates

import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from utils.plotting import plot_accuracy_loss
import matplotlib.pyplot as plt

# These imports are for comparing native model with an equivalent one from Keras
#from keras import models, layers, optimizers


def sigmoid(x):
    """Implements a numerically stable sigmoid function.
       For pos x: 1 / (1 + exp(-x)
       For neg x: exp(x) / (1 + exp(x)"""
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x,dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x,dtype=float)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """Implements 'relu' activation"""
    return np.maximum(0, x)


def make_samples(N):
    """Generate N number of samples of 2 classes, based on Gaussian quantilies.
       The output is X - coordinates (X1,X2) of a sample; Y - sample label.
       X shape is (2,N), Y shape is (1,N)"""
    X, Y = make_gaussian_quantiles(
        mean=None,
        cov=0.7,
        n_samples=N,
        n_features=2,
        n_classes=2,
        shuffle=True,
        random_state=None
    )
    Y = Y.reshape(Y.shape[0], 1)
    return X, Y


def init_parameters(n_x, n_h, n_y):
    """Initialize model weights, biasees"""
    scale = 0.1
    np.random.seed(3)
    params = {
        'W1': np.random.randn(n_h, n_x + 1) * scale,  # hidden layer, with bias
        'W2': np.random.randn(n_y, n_h + 1) * scale,  # output layer, with bias
    }
    return params


def forward_step(X, params):
    """Forward step - compute predictions. Returns layer activations"""
    bias = np.ones(X.shape(0), 1)
    xWithBias = np.append(bias, x, axis=1)

    Z1 = np.dot(xWithBias, params['W1'].T)
    A1 = np.tanh(Z1)
    A1withBias = np.append(bias, A1)
    Z2 = np.dot(A1withBias, params['W2'].T)
    A2 = sigmoid(Z2)
    return A1, A2


def compute_loss(Y, A2):
    """Compute cross-entropy loss - how much predictions differ from actual values"""
    # Loss = - Sum_samples[ Sum_classes[ Y_s_c * log(A2_s_c) ] ]
    logprobs = -np.multiply(Y, np.log(A2)) - np.multiply((1 - Y), np.log(1 - A2))
    loss = np.average(logprobs)
    return loss


def compute_gradients(X, Y, A1, A2, params):
    """Back-propagation - compute gradients"""
    scale = 1.0 / Y.shape[0]  # inverse of a number of samples
    # Diffs
    dZ2 = A2 - Y
    dZ1 = np.dot(dZ2, params['W2'])
    dZ1 = np.multiply( np.multiply(dZ1, A1), (1-A1))  # .*A1.*(1-A1)
    grads = {
        'dW2': scale * np.dot(dZ2.T, A1),  # [n_labeles x n_hidden]
        'db2': scale * np.sum(dZ2, axis=0, keepdims=True),
        'dW1': scale * np.dot(dZ1.T, X),
        'db1': scale * np.sum(dZ1, axis=0, keepdims=True)
    }
    return grads


def update_params(params, grads, lr):
    """Uptade model parameters, givern the learning rate"""
    new_params = {
        'W1': params['W1'] - lr * grads['dW1'],
        'b1': params['b1'] - lr * grads['db1'],
        'W2': params['W2'] - lr * grads['dW2'],
        'b2': params['b2'] - lr * grads['db2']
    }
    return new_params


def predict(X, params):
    """Predict class, based on sample coordinates, and model parameters.
       Prediction is considered to be 1, if probability of the last activation is over 0.5, and 0 otherwise"""
    A1, A2 = forward_step(X, params)
    return A2 > 0.5


def accuracy(Y, predictions):
    """Computes accuracy - a number of correct predictions / total number of samples"""
    correct_predictions = np.dot(Y, predictions.T) + np.dot((1 - Y), (1 - predictions.T))
    return float(correct_predictions) / Y.shape[1] * 100.


def init_history():
    """Initialize model history"""
    history = {
        'acc': [],
        'acc_val': [],
        'loss': [],
        'loss_val': []
    }
    return history


def init_batch_history(history):
    """Initializes history parameters relevant to a batch"""
    history['acc_batch_train'] = []
    history['acc_batch_valid'] = []
    history['loss_batch_train'] = []
    history['loss_batch_valid'] = []
    return history


def finish_epoch_history(history):
    """Moves history parameters from batches to epoch"""
    history['acc'].append(np.average(history['acc_batch_train']))
    history['acc_val'].append(np.average(history['acc_batch_valid']))
    history['loss'].append(np.average(history['loss_batch_train']))
    history['loss_val'].append(np.average(history['loss_batch_valid']))
    return history


def sigle_pass(x_train, y_train, batch_size, params, history):
    """Pass overall samples, in batches, to train the model and update model parameters"""
    n_samples = x_train.shape[0]
    history = init_batch_history(history)

    for idx0 in range(0, n_samples, batch_size):
        idx = idx0 + batch_size
        idx1 = idx if idx < n_samples else n_samples - 1
        X = x_train[idx0: idx1, :]
        Y = y_train[idx0: idx1, :]
        A1, A2 = forward_step(X, params)  # returns layer activations
        history['loss_batch_train'].append(compute_loss(Y, A2))
        grads = compute_gradients(X, Y, A1, A2, params)
        params = update_params(params, grads, lr)
        predictions = predict(X, params)
        history['acc_batch_train'].append(accuracy(Y, predictions))
    history = finish_epoch_history(history)
    return params, history

# 1. Define model architecture
# Layer 0 (input): n_features nodes, i.e. 2 nodes. In Keras, this layer is implied by 'input_dim' argument in the first model layer
# Layer 1 (hidden): 4 nodes
# Layer 2 (output): 1 node - probability of a sample class being 'blue'
n_x = 2
n_h = 4
n_y = 1

n_train = 10000
n_validate = n_train
batch_size = 200
epochs = 60
lr = 1.0e-3  # learning rate
# X, Y = make_samples(n_samples)
# plt.scatter(X[0, :], X[1, :], c=Y[0], cmap=plt.cm.Spectral)
# plt.show()

history = init_history()  # history of training/validation
params = init_parameters(n_x, n_h, n_y)
x_train, y_train = make_samples(n_train)
x_val, y_val = make_samples(n_validate)
for e in range(1, epochs+1):
    print('Epoch', e)
    # A single pass over all samples in batches
    params, history = sigle_pass(x_train, y_train, batch_size, params, history)
    print(history['acc'][-1], history['loss'][-1])


# Compare with Keras model
"""model = models.Sequential()
model.add(layers.Dense(4, activation='tanh', input_shape=(2,)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train.T, y_train.T, batch_size=batch_size, epochs=epochs, validation_data=(x_val.T, y_val.T))
plot_accuracy_loss(
    history.history['acc'],
    history.history['loss'],
    history.history['val_acc'],
    history.history['val_loss'],
)"""
