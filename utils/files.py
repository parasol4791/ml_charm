import os, sys
from PIL import Image

def imageDirStats(dir):
    '''Returns a dictionary for files in a directory with:
    nImages - total number of images,
    minH - image size with a minimal height,
    minW - image size with a minimal width,
    maxH - image size with a max height,
    maxW - image size with a max width '''
    nmax = sys.maxsize
    nImages = 0
    maxHeight = maxWidth = (-1,-1)
    minHeight = minWidth = (nmax,nmax)
    for fName in os.listdir(dir):
        if '.jpg' in fName or '.png' in fName:
            nImages += 1
            sz = Image.open(dir + fName).size
            if sz[0] < minHeight[0]:
                minHeight = sz
            if sz[1] < minWidth[1]:
                minWidth = sz
            if sz[0] > maxHeight[0]:
                maxHeight = sz
            if sz[1] > maxWidth[1]:
                maxWidth = sz
    return {'nImages' : nImages, 'minH' : minHeight, 'minW' : minWidth, 'maxH' : maxHeight, 'maxW' : maxWidth }
