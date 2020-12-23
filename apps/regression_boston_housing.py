import numpy as np
from keras import models, layers
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
from utils.plotting import smooth_curve

def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Dense( 64, activation='relu', input_shape=(train_data.shape[1],) ))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# Has 13 feasures, target house prices are in k$, typically between 10 and 50
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)

# Normalizing training data, since all features are on a different scale
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# Since the training dataset is small (404 samples), we'll use K-fold validation
# I.e. we'll split training dataset into 4 parts, and build 4 models, using i-th subset, then average the 

k = 4
setSize = len(train_targets) // k
numEpochs = 100
all_scores = []
all_mae_histories = []
for i in range(k):
    bound1 = i * setSize
    bound2 = (i+1) * setSize
    val_data = train_data[bound1 : bound2]
    val_targets = train_targets[bound1 : bound2]
    train_data_part = np.concatenate( [train_data[:bound1], train_data[bound2:]], axis=0)
    train_targets_part = np.concatenate( [train_targets[:bound1], train_targets[bound2:]], axis=0)

    model = build_model(train_data)

    # Evaluation of an average MAE over 4 K-scores
    #model.fit(train_data_part, train_targets_part, epochs=numEpochs, batch_size=1, verbose=0)
    #val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    #all_scores.append(val_mae)
#print('Average MAE: ', np.mean(all_scores))

    # Alternatively, plotting validation scores, averaged over all models
    numEpochs = 200
    history = model.fit(train_data_part, train_targets_part, validation_data=(val_data, val_targets), epochs=numEpochs, batch_size=1, verbose=0)
    mae_hist = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_hist)
# Average across models
avg_mae_hist = [ np.mean( [x[i] for x in all_mae_histories] ) for i in range(numEpochs) ]
# Plot valuation evolution
x = range(1,len(avg_mae_hist)+1)
plt.plot(x, avg_mae_hist)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Plot a smooth curve: remove first 10 values, plot weighted moving average
plt.clf()
plt.plot(x[10:], smooth_curve(avg_mae_hist[10:]))
plt.title('Smooth validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()






