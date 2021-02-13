import time

import pandas as pd

from models.simple import build_model_linear_regression
from utils.plotting import plot_the_model, plot_the_loss_curve


def train_model(model, feature, label, nepochs, batch_size):
    """Train the model by feeding it data"""
    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=nepochs)

    # Gather the trained model's weight and bias.
    train_weight = model.get_weights()[0]
    train_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of history.
    train_epochs = history.epoch

    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

    # Specifically gather the model's root mean
    # squared error at each epoch.
    train_rmse = hist["root_mean_squared_error"]

    return train_weight, train_bias, train_epochs, train_rmse


my_feature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
my_label = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])
print('Hello')

learning_rate = 0.5
epochs = 50
my_batch_size = 12

startTime = time.time()
my_model = build_model_linear_regression(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
print('It took {} sec'.format(time.time() - startTime))
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

"""
CPU:
Epoch 50/50
1/1 [==============================] - 0s 42ms/step - loss: 1.1437 - root_mean_squared_error: 1.0695
It took 6.442007303237915 sec
"""

"""
GPU:
Epoch 50/50
1/1 [==============================] - 0s 849us/step - loss: 0.8852 - root_mean_squared_error: 0.9409
It took 0.5409915447235107 sec
"""
