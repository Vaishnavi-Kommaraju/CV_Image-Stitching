import numpy as np
import matplotlib.pyplot as plt
import os
from utils import imread
from utils import showMatches
from detectBlobs import detectBlobs
from compute_sift import compute_sift
from computeMatches import computeMatches
from ransac import ransac
from mergeImages import mergeImages
from skimage.color import rgb2gray as rgb2gray
import pdb
#Image directory
dataDir = os.path.join('..', 'data', 'stitching')

#Read input images
testExamples = ['hill', 'field', 'ledge', 'pier', 'river', 'roofs', 'building', 'uttower']
exampleIndex = 5
imageName1 = '{}1_r.jpg'.format(testExamples[exampleIndex])
imageName2 = '{}2_r.jpg'.format(testExamples[exampleIndex])

im1 = imread(os.path.join(dataDir, imageName1))
im2 = imread(os.path.join(dataDir, imageName2))
print(im1.shape)
print(im2.shape)
#im1 =  rgb2gray(im1)
#im2 = rgb2gray(im2)

#Detect keypoints
blobs1 = detectBlobs(im1)
blobs2 = detectBlobs(im2)
print(blobs1.shape)
print(blobs2.shape)

#Compute SIFT features
sift1 = compute_sift(im1, blobs1[:, 0:4])
sift2 = compute_sift(im2, blobs2[:, 0:4])
print(sift1.shape, sift2.shape)

#Find the matching between features
matches = computeMatches(sift1, sift2)
showMatches(im1, im2, blobs1, blobs2, matches)
#Ransac to find correct matches and compute transformation
inliers, transf = ransac(matches, blobs1, blobs2)
print(inliers.shape)
print(transf)
goodMatches = np.ones_like(matches)*(-1)
goodMatches[inliers] = matches[inliers]
#pdb.set_trace()
showMatches(im1, im2, blobs1, blobs2, goodMatches)

#Merge two images and display the output
stitchIm = mergeImages(im1, im2, transf)
plt.figure()
plt.imshow(stitchIm)
plt.title('stitched image: {}'.format(testExamples[exampleIndex]))
