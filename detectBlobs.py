# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import numpy as np
import cv2
from scipy.ndimage.filters import convolve
from utils import loG
from scipy.ndimage.filters import generic_filter 
from scipy.ndimage.filters import gaussian_laplace
from skimage.color import rgb2gray as rgb2gray

def func(values):
    return (values.argmax() == 13)
def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 5 array with blob in each row in (x, y, radius, angle, score)
    #
    # Dummy - returns a blob at the center of the image
    
    im =  rgb2gray(im)
    numLevels = 15
    k = 1.5
    sigma = 1.2
    threshold = 0.01
    blob = []
    scaleSigma = [(k**i)*sigma for i in range(15)]
    #print(scaleSigma[14])
    scaleSpace = np.zeros((im.shape[0], im.shape[1], numLevels))
    for key, val in enumerate(scaleSigma):
        #print(val)
        #kernel = loG(3*val,sigma=val)
        #print(kernel.shape)
        #squaredResponse = convolve(im,kernel,mode='constant', cval=0.0)**2
        squaredResponse = ((val**2)*gaussian_laplace(im,sigma=val,mode='constant', cval=0.0))**2
        scaleSpace[:,:,key] = squaredResponse
    out = generic_filter(scaleSpace, func, footprint=np.ones((3,3,3)), mode='constant', cval=0.0)
    max_ind = np.where(out)
    #print(max_ind[0].shape)
    #indexs = zip(max_ind[0], max_ind[1], max_ind[2])
    #print(indexs)
    temp = scaleSpace[max_ind] > threshold
    print(len(temp))
    #print(temp)
    for key,val in enumerate(temp):
        if val:
            x = max_ind[0][key]
            y = max_ind[1][key]
            n = max_ind[2][key]
            #print(n)
            blob.append((y, x, np.sqrt(2)*scaleSigma[n], 0, scaleSpace[x,y,n]))

    return np.asarray(blob)
