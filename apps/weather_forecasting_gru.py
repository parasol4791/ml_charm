# Weather forecasting problem, using time series from weather sensors in Jena lab, Germany
# wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip

import os
import time

import numpy as np
import pandas as pd
from keras import models, layers, optimizers

# from matplotlib import pyplot as plt
from utils.plotting import plot_loss
from utils.compatibility import compat_lstm_cancel_error


def generator_timeseries(
        name,
        input_data,  # numpy array
        lookback_sz,  # number of timestaps in the past
        delay_sz,  # number of timesteps in the future
        min_idx,  # min index in the data to start sampling
        max_idx=None,  # max index in the data to end sampling
        shuffle=False,
        batch_size=128,
        step_sz=6,  # sampling frequency
        target_column_index=0  # index of a column with target quantity
):
    """Generator for time series. It yields batches of input_data with lookback_sz timesteps in the past
       and delay_sz timesteps into the future"""
    #min_idx = min_idx or 0
    max_sz = len(input_data) - 1 - delay_sz
    max_idx = max_idx or max_sz  # to allow for delay_sz timesteps in the future
    max_idx = min(max_idx, max_sz)
    i = min_idx + lookback_sz
    while 1:
        #print(name, i)
        if shuffle:
            rows = np.random.randint(min_idx + lookback_sz, max_idx, size=batch_size)
        else:
            if i >= max_idx - delay:
                i = min_idx + lookback_sz
            rows = np.arange(i, min(i + batch_size, max_idx))
            i += len(rows)

        samples = np.zeros((len(rows), lookback_sz // step_sz, input_data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(row - lookback_sz, row, step_sz)
            samples[j] = input_data[indices]
            targets[j] = input_data[row + delay_sz][target_column_index]
        yield samples, targets


def naive_predict(timeseries, delay_sz):
    """Always predict a parameter in a 'timeseries'
       that is 'delay_sz' steps into the future, is the same as it is now"""
    sz = len(timeseries)
    sum_mae = 0  # mean absolute error
    i = 0
    while (i + delay_sz) < sz:
        sum_mae += abs(timeseries[i + delay_sz] - timeseries[i])
        i += 1
    return sum_mae / i


data_dir = os.environ['DATASETS_DIR']
file_path = os.path.join(data_dir, 'jena_climate', 'jena_climate_2009_2016.csv')

# Get data from file
with open(file_path) as f:
    df = pd.read_csv(f)

# Remove date/time
df = df.drop('Date Time', 1)

# Normalize all data
mean = df.mean()
std = df.std()
df = (df - mean) / std
print(df.loc[0])
data = np.array(df)

# Samples are taken every 10 min
lookback = 1440  # 10 days
delay = 144  # 24 h
step = 6  # 1 h
batch_sz = 128
epochs = 20
do_shuffle = True
target_idx = df.columns.get_loc('T (degC)')
print(df['T (degC)'][:20])
num_train = 200000
num_val = 100000

# Create generators
idx1 = num_train
train_gen = generator_timeseries('train', data, lookback, delay,
                                 min_idx=0, max_idx=idx1, shuffle=do_shuffle, batch_size=batch_sz, step_sz=step,
                                 target_column_index=target_idx)
idx2 = idx1 + num_val
val_gen = generator_timeseries('val', data, lookback, delay,
                               min_idx=idx1, max_idx=idx2, shuffle=do_shuffle, batch_size=batch_sz, step_sz=step,
                               target_column_index=target_idx)
test_gen = generator_timeseries('test', data, lookback, delay,
                                min_idx=idx2, max_idx=None, shuffle=do_shuffle, batch_size=batch_sz, step_sz=step,
                                target_column_index=target_idx)

# Number of steps per epoch
n_train = num_train - lookback - delay
train_steps = n_train if do_shuffle else np.ceil(n_train / batch_sz)
n_val = num_val - lookback - delay
val_steps = n_val if do_shuffle else np.ceil(n_val // batch_sz)
n_test = len(data) - idx2 - lookback - delay
test_steps = n_test if do_shuffle else np.ceil(n_test // batch_sz)

# MAE when predicting temperature is the same 24 h from now
temp_ts = np.array(df['T (degC)'][idx1: idx2])
mae = naive_predict(temp_ts, delay)
print(mae)
print(mae * std[target_idx])

t1 = time.time()

# Dense model baseline
"""model = models.Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(1))  # no activation for a regression model

model.compile(optimizer=optimizers.RMSprop(), loss='mae')
history = model.fit(train_gen,
                    batch_size=4,
                    steps_per_epoch=1000,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=1000
                    )
"""
t2 = time.time()
print('Base-line Dense model took {} sec'.format(t2 - t1))


# Recurrent GRU model
compat_lstm_cancel_error()

model = models.Sequential()
model.add(layers.GRU(16, input_shape=(None, data.shape[-1]),))
# Stacked RNN with dropouts - takes much longer, no significant improvement in validation accuracy
# model.add(layers.GRU(32, input_shape=(None, data.shape[-1]), activation='relu', dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
# model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5), )
model.add(layers.Dense(1))  # no activation for a regression model

model.compile(optimizer=optimizers.RMSprop(), loss='mae')
history = model.fit(train_gen,
                    batch_size=batch_sz,
                    steps_per_epoch=200,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=200
                    )

t3 = time.time()
print('It took {} sec to run GRU model'.format(t3 - t2))

# Plot accuracy and loss
plot_loss(
    history.history['loss'],
    history.history['val_loss'],
)

"""plt.plot(df['T (degC)'][:])
plt.title('T, deg C')
plt.figure()
plt.plot(df['p (mbar)'][:])
plt.title('p, mbar')
plt.show()"""

# Results
"""
Common sense assumption future T = T now
mae loss = 0.2928478871521155
avg diff in deg C = 2.4668583760615674
"""
"""
Dense model with Dropout - a bit worse than a simple common sense one!!!
Epoch 19/20
1000/1000 [==============================] - 9s 9ms/step - loss: 0.3151 - val_loss: 0.3044
Base-line Dense model took 185.6866159439087 sec
"""
"""
Recurrent GRU model (dropouts and stacking recurrent layers do not help, just sharply increases computation time)
Epoch 15/20
200/200 [==============================] - 2s 11ms/step - loss: 0.2761 - val_loss: 0.2747
It took 45.50749492645264 sec to run GRU model
"""