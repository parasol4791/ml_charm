# Binary classification of images 'dogs' vs 'cats'
# Uses convolution neural network (CNN) with augmentation

import os
import shutil
import time

from keras import models, layers, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator

# To avoid an error (see the method)
from utils.compatibility import compat_no_algo
from utils.files import imageDirStats, make_dir
from utils.plotting import plot_accuracy_loss

compat_no_algo()

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

for d in dirs:
    print(d, imageDirStats(d))

# Results:
""" 
C:/Datasets/dogs_vs_cats/training_set/cats/ {'nImages': 4000, 'minH': (59, 41), 'minW': (59, 41), 'maxH': (1023, 768), 'maxW': (1023, 768)}
C:/Datasets/dogs_vs_cats/training_set/dogs/ {'nImages': 4005, 'minH': (57, 50), 'minW': (59, 45), 'maxH': (1050, 702), 'maxW': (1050, 702)}
C:/Datasets/dogs_vs_cats/test_set/cats/ {'nImages': 1011, 'minH': (59, 106), 'minW': (60, 39), 'maxH': (500, 399), 'maxW': (335, 500)}
C:/Datasets/dogs_vs_cats/test_set/dogs/ {'nImages': 1012, 'minH': (92, 113), 'minW': (195, 33), 'maxH': (500, 400), 'maxW': (428, 500)}
"""

# Extract a smaller subset of images for training/validation/testing and move them to new directories
trainSz = 1000  # 1000 for cats, and 1000 for dogs
valSz = 500
testSz = 500

# Create new dirs
path = make_dir(dataDir, 'dogs_vs_cats_small')
trainDir = make_dir(path, 'train')
trainDirDogs = make_dir(trainDir, 'dogs')
trainDirCats = make_dir(trainDir, 'cats')
valDir = make_dir(path, 'validate')
valDirDogs = make_dir(valDir, 'dogs')
valDirCats = make_dir(valDir, 'cats')
testDir = make_dir(path, 'test')
testDirDogs = make_dir(testDir, 'dogs')
testDirCats = make_dir(testDir, 'cats')

# Copy over the files
for animal in ['cat', 'dog']:
    # Training
    fNames = ['{}.{}.jpg'.format(animal, i) for i in range(1, trainSz + 1)]
    for fName in fNames:
        src = os.path.join(orig_path, 'training_set', '{}s'.format(animal), fName)
        dest = os.path.join(path, 'train', '{}s'.format(animal), fName)
        shutil.copyfile(src, dest)

    # Validation
    fNames = ['{}.{}.jpg'.format(animal, i) for i in range(trainSz + 1, trainSz + valSz + 1)]
    for fName in fNames:
        src = os.path.join(orig_path, 'training_set', '{}s'.format(animal), fName)
        dest = os.path.join(path, 'validate', '{}s'.format(animal), fName)
        shutil.copyfile(src, dest)

    # Testing
    fNames = ['{}.{}.jpg'.format(animal, i) for i in range(4001, 4001 + testSz)]
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
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0003)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))  # categorical classification, a probability of being either a 'cat' or a 'dog'

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=4.e-4), metrics=['acc'])

# Using image augmentation to increase sample set, and improve model accuracy
trainDataGen = ImageDataGenerator(
    rescale=1. / 255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
testDataGen = ImageDataGenerator(rescale=1. / 255.)

trainGen = trainDataGen.flow_from_directory(trainDir, target_size=(150, 150), batch_size=20, class_mode='categorical')
# Do not augment validation data!!!
valGen = testDataGen.flow_from_directory(valDir, target_size=(150, 150), batch_size=20, class_mode='categorical')

# for dataBatch, labelBatch in trainGen:
#    print('Data batch shape: ', dataBatch.shape)
#    print('Labels shape: ', labelBatch.shape)
#    break

history = model.fit(trainGen, epochs=100, steps_per_epoch=100, validation_data=valGen, validation_steps=50)
outputs_dir = os.environ['OUTPUTS_DIR']
model.save(os.path.join(outputs_dir, 'cats_n_dogs_small.h5'))
print(history.history.keys())
print('It took {} sec'.format(time.time() - startTime))

plot_accuracy_loss(
    history.history['acc'],
    history.history['loss'],
    history.history['val_acc'],
    history.history['val_loss'],
)

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
"""
With augmented data & 100 epochs
Epoch 100/100
100/100 [==============================] - 7s 69ms/step - loss: 0.3408 - acc: 0.8592 - val_loss: 0.4513 - val_acc: 0.8370
dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
It took 693.2404601573944 sec
"""
"""
With dropout layers & 150 epochs
Epoch 149/150
100/100 [==============================] - 7s 69ms/step - loss: 0.3521 - acc: 0.8424 - val_loss: 0.4032 - val_acc: 0.8290
dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
It took 1031.9605205059052 sec
"""
