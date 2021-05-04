# Utilities for handling tansorflow datasets

import tensorflow_datasets as tfds
import tensorflow as tf
import py, sys
import matplotlib.pyplot as plt


def getFileName(elem):
    """Returns file name of a dataset element"""
    return elem['image/filename'].numpy().decode()


def getImage(elem):
    """Returns image from a dataset element"""
    return elem['image']


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