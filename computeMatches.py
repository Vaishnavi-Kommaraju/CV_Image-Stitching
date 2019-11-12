import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 4

def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """
    # implement this
    print(f1.shape, f2.shape)
    #f1 = np.transpose(f1)
    #f2 = np.transpose(f2)
    #print(f1.shape, f2.shape)
    threshold = 0.8
    matches = np.ones(f1.shape[0])*(-1)
    cnt = 0
    for i in range(f1.shape[0]):
        ssd = np.mean((f1[i,:] - f2 )**2, axis = 1)
        ssd_sorted = ssd.argsort()
        ratio = np.linalg.norm(f1[i,:] - f2[ssd_sorted[0],:])/np.linalg.norm(f1[i,:] - f2[ssd_sorted[1],:])
        #print(ratio)
        if ratio > threshold:
            cnt += 1
        if ratio <= threshold:
            matches[i] = ssd_sorted[0]
    #print(matches)
    print(np.sum(matches>=0))
    print(cnt)
    
    return matches
