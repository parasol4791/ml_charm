
import os, shutil
from utils.files import imageDirStats
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
import os

# Compatibility to avoid error:
# tensorflow.python.framework.errors_impl.NotFoundError:  No algorithm worked!
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


dataDir = os.environ['DATASETS_DIR']
orig_path = os.path.join(dataDir, 'dogs_vs_cats')
startTime = time.time()

# Min image sizes for directories
origTrainCatsDir = os.path.join(orig_path, 'training_set/cats/')
dirs = [origTrainCatsDir]
origTrainDogsDir = os.path.join(orig_path, 'training_set/dogs/')
dirs.append(origTrainDogsDir)
origTestCatsDir = os.path.join(orig_path, 'test_set/cats/')
dirs.append(origTestCatsDir)
origTestDogsDir = os.path.join(orig_path, 'test_set/dogs/')
dirs.append(origTestDogsDir)

for dir in dirs:
    print(dir, imageDirStats(dir))

# Results:
""" 
C:/Datasets/dogs_vs_cats/training_set/cats/ {'nImages': 4000, 'minH': (59, 41), 'minW': (59, 41), 'maxH': (1023, 768), 'maxW': (1023, 768)}
C:/Datasets/dogs_vs_cats/training_set/dogs/ {'nImages': 4005, 'minH': (57, 50), 'minW': (59, 45), 'maxH': (1050, 702), 'maxW': (1050, 702)}
C:/Datasets/dogs_vs_cats/test_set/cats/ {'nImages': 1011, 'minH': (59, 106), 'minW': (60, 39), 'maxH': (500, 399), 'maxW': (335, 500)}
C:/Datasets/dogs_vs_cats/test_set/dogs/ {'nImages': 1012, 'minH': (92, 113), 'minW': (195, 33), 'maxH': (500, 400), 'maxW': (428, 500)}
"""

# Extract a smaller subset of images for training/validation/testing and move them to new directories
trainSz = 1000 # 1000 for cats, and 1000 for dogs
valSz = 500
testSz = 500

# Create new dirs
path = os.path.join(dataDir, 'dogs_vs_cats_small')
if not os.path.exists(path):
    os.mkdir(path)

trainDir = os.path.join(path, 'train')
if not os.path.exists(trainDir):
    os.mkdir(trainDir)
trainDirDogs = os.path.join(trainDir, 'dogs')
if not os.path.exists(trainDirDogs):
    os.mkdir(trainDirDogs)
trainDirCats = os.path.join(trainDir, 'cats')
if not os.path.exists(trainDirCats):
    os.mkdir(trainDirCats)

valDir = os.path.join(path, 'validate')
if not os.path.exists(valDir):
    os.mkdir(valDir)
valDirDogs = os.path.join(valDir, 'dogs')
if not os.path.exists(valDirDogs):
    os.mkdir(valDirDogs)
valDirCats = os.path.join(valDir, 'cats')
if not os.path.exists(valDirCats):
    os.mkdir(valDirCats)

testDir = os.path.join(path, 'test')
if not os.path.exists(testDir):
    os.mkdir(testDir)
testDirDogs = os.path.join(testDir, 'dogs')
if not os.path.exists(testDirDogs):
    os.mkdir(testDirDogs)
testDirCats = os.path.join(testDir, 'cats')
if not os.path.exists(testDirCats):
    os.mkdir(testDirCats)

# Copy over the files
for animal in ['cat', 'dog']:
    # Training
    fNames = ['{}.{}.jpg'.format(animal, i) for i in range(1,trainSz+1)]
    for fName in fNames:
        src = os.path.join(orig_path, 'training_set', '{}s'.format(animal), fName)
        dest = os.path.join(path, 'train', '{}s'.format(animal), fName)
        shutil.copyfile(src, dest)

    # Validation
    fNames = ['{}.{}.jpg'.format(animal, i) for i in range(trainSz+1, trainSz+valSz+1)]
    for fName in fNames:
        src = os.path.join(orig_path, 'training_set', '{}s'.format(animal), fName)
        dest = os.path.join(path, 'validate', '{}s'.format(animal), fName)
        shutil.copyfile(src, dest)

    # Testing
    fNames = ['{}.{}.jpg'.format(animal, i) for i in range(4001, 4001+testSz)]
    for fName in fNames:
        src = os.path.join(orig_path, 'test_set', '{}s'.format(animal), fName)
        dest = os.path.join(path, 'test', '{}s'.format(animal), fName)
        shutil.copyfile(src, dest)
 
print(trainDirCats, len(os.listdir(trainDirCats)))
print(trainDirDogs, len(os.listdir(trainDirDogs)))
print(valDirCats, len(os.listdir(valDirCats)))
print(valDirDogs, len(os.listdir(valDirDogs)))        
print(testDirCats, len(os.listdir(testDirCats)))
print(testDirDogs, len(os.listdir(testDirDogs)))

# Build a model
model =  models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) # binary classification, a probability of being either a 'cat' or a 'dog'

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1.e-4), metrics=['acc'])

trainDataGen = ImageDataGenerator(rescale=1./255.)
testDataGen = ImageDataGenerator(rescale=1./255.)

trainGen = trainDataGen.flow_from_directory(trainDir, target_size=(150,150), batch_size=20, class_mode='binary')
valGen = trainDataGen.flow_from_directory(valDir, target_size=(150,150), batch_size=20, class_mode='binary')

#for dataBatch, labelBatch in trainGen:
#    print('Data batch shape: ', dataBatch.shape)
#    print('Labels shape: ', labelBatch.shape)
#    break

hist = model.fit(trainGen, steps_per_epoch=100, epochs=30, validation_data=valGen, validation_steps=50)
model.save('cats_n_dogs_small.h5')
print(hist.history.keys())
print('It took {} sec'.format( time.time() - startTime ))


acc = hist.history['acc']
loss = hist.history['loss']
valAcc = hist.history['val_acc']
valLoss = hist.history['val_loss']
epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, valAcc, 'b', label='Validation acc')
plt.title('Training & validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, valLoss, 'b', label='Validation loss')
plt.title('Training & validation loss')
plt.legend()
plt.show()

"""
CPU (laptop, 4 cores):
Epoch 30/30
100/100 [==============================] - 58s 583ms/step - loss: 0.0486 - acc: 0.9855 - val_loss: 0.9603 - val_acc: 0.7340
It took 1773.8488445281982 sec
"""

"""
CPU (desktop, 8 cores):
Epoch 30/30
100/100 [==============================] - 15s 152ms/step - loss: 0.0363 - acc: 0.9973 - val_loss: 0.9803 - val_acc: 0.7330
It took 458.14360189437866 sec
"""

"""
GPU:
Epoch 30/30
100/100 [==============================] - 2s 23ms/step - loss: 0.0423 - acc: 0.9913 - val_loss: 0.9683 - val_acc: 0.7340
It took 72.86363244056702 sec
"""


