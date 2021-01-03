import pandas as pd
import numpy as np
from models.simple import build_model_linear_regression
from utils.plotting import plot_the_model, plot_the_loss_curve
import tensorflow as tf
import time


def train_model(model, df, feature, label, nepochs, batch_sz):
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_sz,
                        epochs=nepochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    train_epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    train_rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, train_epochs, train_rmse


def predict_house_values(start, n, train_df, feature, label):
    """Predict house values based on a feature."""

    batch = train_df[feature][start:start + n]
    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print("%5.2f %6.2f %15.2f" % (train_df[feature][start + i],
                                      train_df[label][start + i],
                                      predicted_values[i][0]))


startTime = time.time()
# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
tf.config.optimizer.set_jit(False) # Start with XLA enabled

# Import the dataset.
training_df = pd.read_csv(
    filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Scale the label.
training_df["median_house_value"] /= 1000.0

# Add a synthetic parameter - room density
training_df['rooms_per_person'] = training_df['total_rooms'] / training_df['population']

# Print the first rows of the pandas DataFrame.
print(training_df.head())
print(training_df.describe())
print(training_df.corr())

# Remove house values of 500K, they seem to be capped, and distort the data
training_df = training_df[training_df['median_house_value'] < 500.0]
print('After removing median_house_value capped values')
print(training_df.describe())

# Remove rows with > 4 standard deviations
"""featuresToClean = ['rooms_per_person', 'median_income', 'population', 'households', 'total_rooms']
for f in featuresToClean:
    ndev = 4 #2.5 if f == 'median_house_value' else 6
    mean = training_df[f].mean()
    std = training_df[f].std()
    training_df = training_df[training_df[f] < mean + ndev * std]
    print('After removing {} outliers'.format(f))
    print(training_df.describe())"""
print(training_df.corr())

# The following variables are the hyperparameters.
learning_rate = 0.05
epochs = 30
batch_size = 30

# Specify the feature and the label.
my_feature = "median_income"
my_label = "median_house_value"  # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based 
# solely on total_rooms.  

# Invoke the functions.
my_model = build_model_linear_regression(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias)
print("It took {} sec\n".format(time.time() - startTime))

training_df_sorted = training_df.sort_values([my_feature])
plot_the_model(weight, bias, training_df_sorted[my_feature].array, training_df_sorted[my_label].array)
plot_the_loss_curve(epochs, np.array(rmse))

sample_start = 100
predict_house_values(sample_start, 10, training_df, my_feature, my_label)


"""
GPU (presumably, with CUDA 10.1):
Epoch 30/30
539/539 [==============================] - 0s 249us/step - loss: 5414.7966 - root_mean_squared_error: 73.5833
The learned weight for your model is 40.1823
The learned bias for your model is 43.0378
It took 5.544398546218872 sec
"""
"""
GPU (CUDA 11.2):
Epoch 30/30
539/539 [==============================] - 0s 523us/step - loss: 5544.6722 - root_mean_squared_error: 74.4520
The learned weight for your model is 40.2807
The learned bias for your model is 43.3878
It took 10.856559991836548 sec
"""