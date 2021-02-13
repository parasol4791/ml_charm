import os
import sys

from PIL import Image


def imageDirStats(directory):
    """Returns a dictionary for files in a directory with:
    nImages - total number of images,
    minH - image size with a minimal height,
    minW - image size with a minimal width,
    maxH - image size with a max height,
    maxW - image size with a max width"""
    nmax = sys.maxsize
    n_images = 0
    maxHeight = maxWidth = (-1, -1)
    minHeight = minWidth = (nmax, nmax)
    for fName in os.listdir(directory):
        if '.jpg' in fName or '.png' in fName:
            n_images += 1
            sz = Image.open(directory + fName).size
            if sz[0] < minHeight[0]:
                minHeight = sz
            if sz[1] < minWidth[1]:
                minWidth = sz
            if sz[0] > maxHeight[0]:
                maxHeight = sz
            if sz[1] > maxWidth[1]:
                maxWidth = sz
    return {'nImages': n_images, 'minH': minHeight, 'minW': minWidth, 'maxH': maxHeight, 'maxW': maxWidth}
