import os
import numpy as np
import matplotlib.pyplot as plt
from utils import imread
from detectBlobs import detectBlobs
from drawBlobs import drawBlobs
import cv2
from skimage.color import rgb2gray as rgb2gray
# Evaluation code for blob detection
# Your goal is to implement scale space blob detection using LoG  
#
# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji

imageName = 'butterfly_new.jpg'
numBlobsToDraw = 1000
imName = imageName.split('.')[0]

datadir = os.path.join('..', 'data', 'blobs')
x = os.path.join(datadir, imageName)
im = imread(os.path.join(datadir, imageName))

#im = cv2.imread(os.path.join(datadir, imageName))
#im =  rgb2gray(im)
#out = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

blobs = detectBlobs(im)  # dummy placeholder

drawBlobs(im, blobs, numBlobsToDraw)

