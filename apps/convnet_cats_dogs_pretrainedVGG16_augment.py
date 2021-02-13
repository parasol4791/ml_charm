# Binary classification of images 'dogs' vs 'cats'
# Uses convolution base of pre-trained CNN model VGG16 with a custom classifier (feature extraction).
# Image augmentation is added.
# Joint training of custom classifier with a few top layers of conv base is done (fine-tuning).

import os
import time

from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

# To avoid an error (see the method)
from utils.compatibility import compat_no_algo
from utils.plotting import plot_accuracy_loss

compat_no_algo()

startTime = time.time()

# Prepare input directories
data_dir = os.environ['DATASETS_DIR']
base_dir = os.path.join(data_dir, 'dogs_vs_cats_small')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validate')
test_dir = os.path.join(base_dir, 'test')

# Use image augmentation to increase sample set, and improve model accuracy
trainDataGen = ImageDataGenerator(
    rescale=1. / 255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
testDataGen = ImageDataGenerator(rescale=1. / 255.)

batch_sz = 10
train_steps = 200
valid_steps = 100
trainGen = trainDataGen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_sz, class_mode='binary')
# Do not augment validation data!!!
valGen = testDataGen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=batch_sz,
                                         class_mode='binary')

# Get convolution base from VGG16
conv_base = VGG16(
    weights='imagenet',
    include_top=False,  # original top 'Dense' layers used for classification are excluded
    input_shape=(150, 150, 3)
)

# Build a model using VGG16 conv base, plus a custom classifier
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


def freezeConvBaseAll():
    """ Freeze all layers of conv base of VGG16 model, to preserve them, and train only custom classifier """
    print('# trainable weights before freeze: {}'.format(len(model.trainable_weights)))
    conv_base.trainable = False
    # Only 2 Dense layers are trainable now, with weight matrix / bias vector for each, 4 trainable sets in total
    print('# trainable weights after freeze: {}'.format(len(model.trainable_weights)))


def freezeConvBasePart():
    conv_base.trainable = True
    is_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            is_trainable = True
        layer.trainable = is_trainable


# First, freeze all conv base, train custom classifier.
# Then unfreeze a few top layers of conv base, and fine-tune (jointly train with the classifier)
freezeBaseAll = False
if freezeBaseAll:
    freezeConvBaseAll()
else:
    freezeConvBasePart()

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

history = model.fit(trainGen, epochs=40, steps_per_epoch=train_steps, validation_data=valGen,
                    validation_steps=valid_steps)
outputs_dir = os.environ['OUTPUTS_DIR']
model.save(os.path.join(outputs_dir, 'cats_n_dogs_small_fromVGG16_augment.h5'))
print(history.history.keys())
print('It took {} sec'.format(time.time() - startTime))

# Plot accuracy and loss
plot_accuracy_loss(
    history.history['acc'],
    history.history['loss'],
    history.history['val_acc'],
    history.history['val_loss'],
)

# Results:
"""
GPU - training custom classifier on top of VGG16 conv base:
Epoch 30/30
100/100 [==============================] - 7s 71ms/step - loss: 0.2300 - acc: 0.9171 - val_loss: 0.3025 - val_acc: 0.8810
It took 216.08081030845642 sec

Fine-turning (joint training with the top conv base layers)
Epoch 29/30
100/100 [==============================] - 7s 71ms/step - loss: 0.1027 - acc: 0.9669 - val_loss: 0.3053 - val_acc: 0.9420
It took 216.03741216659546 sec
"""
