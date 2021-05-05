# Utilities for handling tansorflow datasets

import py, sys
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from utils.files import make_dir


def datasets_list():
    """Returns a list of all available TF datasets"""
    return tfds.list_builders()


def getFileName(elem):
    """Returns file name of a dataset element"""
    return elem['image/filename'].numpy().decode()


def getImage(elem):
    """Returns image from a dataset element"""
    return elem['image']


def labelToText(label, ds_info):
    """Converts a numerical label into a human0readable text"""
    label_info = ds_info.features["label"]
    return label_info.int2str(label)

def getNumClasses(ds_info):
    """Returns a number of classes in a dataset"""
    return ds_info.features["label"].num_classes


def viewImage(elem):
    """Plots an image of a dataset element"""
    plt.title(getFileName(elem))
    plt.imshow(getImage(elem))
    plt.show()


def getCorruptFiles(dataset):
    """Rerurns a list of corrupt files, and corresponding error messages.
        Dataset should be loaded wit flag 'as_supervised'=False"""
    capture = py.io.StdCaptureFD()  # redirect I/O
    corrupt_files= []
    errMsg = []
    for x in dataset:
        out, err = capture.readouterr()
        if err:
            corrupt_files.append(getFileName(x))
            errMsg.append(err)
    capture.done()  # restore I/O
    return corrupt_files, errMsg


def buildSplitStrings(trainSz, validSz=None, testSz=None, is_percentage=True, splitName='train'):
    """Returns sub-split strings"""
    # Slicing datasets: https://www.tensorflow.org/datasets/splits
    trainValidSz = trainSz + validSz
    allSz = trainValidSz + testSz
    if is_percentage and allSz > 100:
        raise ValueError('Sum of train, validation, and test sets should be <= 100%, it is {}%'.format(allSz))
    unit = '%' if is_percentage else ""
    res = ['{}[:{}{}]'.format(splitName, trainSz, unit)]
    if validSz:
        res.append('{}[{}{}:{}{}]'.format(splitName, trainSz, unit, trainValidSz, unit))
    if testSz:
        res.append('{}[{}{}:{}{}]'.format(splitName, trainValidSz, unit, allSz, unit))
    return res


def copyDatasetFiles_JPG(ds, ds_info, rootDir, splitName='train', keyData='image', keyFileName='image/filename', keyLabel='label'):
    """Copies data from a dataset to local folders.
        Dataset should be a dict (no 'as_supervised' flag).
        Under rootDir, there will be created the following directories structure:
          splitName directory
            label 1 directory
            ..
            label n directory
        Returns a table of class directorits with a number of copied files,
        plus a total of copied files"""
    stats = {}
    def add(lbl):
        if lbl in stats:
            stats[lbl] += 1
        else:
            stats[lbl] = 1

    rDir = make_dir(rootDir, '')
    splitDir = make_dir(rDir, splitName)
    for x in ds:
        # Figure out label dir
        lbl = x[keyLabel]
        label = labelToText(x['label'], ds_info)
        lblDir = make_dir(splitDir, label)
        # Figure out file name
        fName = x[keyFileName]
        fName = fName.numpy().decode().split('/')[-1]
        filePath = os.path.join(lblDir, fName)
        # Convert image and write it to disk
        data = x[keyData]
        out = tf.cast(data, tf.uint8)
        out = tf.image.encode_jpeg(out)
        tf.io.write_file(filePath, out)
        # Gather stats
        add(label)
    total = 0
    for n in stats.values():
        total += n
    return stats, total
"""
An example of using this method:

"# as_supervised = False
trainSz = 60
validSz = 20
testSz = 20
trainStr, validStr, testStr = buildSplitStrings(trainSz, validSz, testSz)

root_path = os.path.join(dataDir, 'cats_vs_dogs/downloads/copy')
(ds_train1, ds_valid1, ds_test1), ds_info1 = tfds.load(dataset_name, split=[trainStr, validStr, testStr], with_info=True, as_supervised=False, shuffle_files=True)
print('train: ', copyDatasetFiles_JPG(ds_train1, ds_info1, root_path, splitName='train') )
print('valid: ', copyDatasetFiles_JPG(ds_valid1, ds_info1, root_path, splitName='validation') )
print('test: ', copyDatasetFiles_JPG(ds_test1, ds_info1, root_path, splitName='test') )
"""


def findMislabeledImages(model, dataset, showImages=False, printMislabeled=False):
    """Returns a list of dataset images misclassified by a model (fileName, true label, prediction).
        If showImages is True, all misclassified images are shown one by one, as found.
        If printMislabeled is True, mislabeled file names will be printed out, as found"""
    t1 = time.time()
    err = 0
    res = []
    for x in dataset:
        pred = model.predict(x['image'])
        label = x['label'].numpy()
        for pr, lbl, fn, img in zip(pred, label, x['image/filename'], x['image']):
            predDig = np.argmax(pr)
            if predDig != lbl:
                fName = fn.numpy().decode()
                prob = pr[predDig]
                res.append((fName, lbl, prob))
                err += 1
                if printMislabeled:
                    print(fName, lbl, prob)
                if showImages:
                    plt.title(fName)
                    plt.imshow(img.numpy().astype("uint8"))
                    plt.show()
    t2 = time.time()
    print('It took {} sec to find misclassified images'.format(t2 - t1))
    return res, err


#########  FILTERS  ##########

def skipFiles(x, filesToSkip):
    """Skips files from dataset"""
    fName = x['image/filename']
    broadcase_eq = tf.not_equal(fName, filesToSkip)  # broadcasts and compares element-wise
    return tf.reduce_all(broadcase_eq)  # reducing to a single bool. True if all are not equal


def onlySkippedFiles(x, filesToSkip):
    """Returns True only for files that have to be skipped"""
    return tf.equal(skipFiles(x, filesToSkip), False)

#########  END OF FILTERS  ##########