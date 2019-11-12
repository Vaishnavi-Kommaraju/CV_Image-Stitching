import numpy as np
import cv2
import time
import pdb
# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2019
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Mini-project 4

def ransac(matches, blobs1, blobs2):
    # implement this
    #pass
    numIter = 100
    ind = np.asarray(np.where(matches>=0)[0])

    samples_2 = blobs1[ind]
    samples_1 = blobs2[matches[ind].astype(int)]
    prev_max = 0
    cnt = 0
    for i in range(numIter):
        A = []
        B = []
        three_samples = np.random.choice(len(samples_1),3)
        img1_points = samples_1[three_samples]
        img2_points = samples_2[three_samples]
        for i in range(len(img1_points)):
            A.append([img1_points[i][0], img1_points[i][1], 0, 0, 1, 0])
            A.append([0, 0, img1_points[i][0], img1_points[i][1], 0, 1])
            B.append(img2_points[i][0])
            B.append(img2_points[i][1])
        matA = np.asarray(A)
        matB = np.asarray(B)
        #print(matA)
        #print(matB)
        result = np.linalg.lstsq(matA, matB)[0]
        #result = np.linalg.solve(matA, matB)
        #print(result)
        #print(np.dot(matA, result))
        #print(matB)
        err = []
        for i in range(len(samples_1)):
            points = np.asarray([[samples_1[i][0], samples_1[i][1], 0, 0, 1, 0], 
                                 [0, 0, samples_1[i][0], samples_1[i][1], 0, 1]])
            out = np.dot(points, result)
            #pdb.set_trace()
            #print(out)
            err.append(np.sqrt((out[0]-samples_2[i][0])**2 + (out[1]-samples_2[i][1])**2))
        
        norm_err = err/np.mean(err)
        #print('len',len(norm_err))
        #print(norm_err)
        count = np.sum(np.asarray(norm_err)<0.15)
        indices_ = ind[np.where(np.asarray(norm_err)<0.15)[0]]
        #pdb.set_trace()
        #print(count)
        if count > prev_max:
            cnt += 1
            prev_max = count
            best_model = result.copy()
            indices = indices_.copy()
    print(cnt)   
    return indices, np.asarray([[best_model[0], best_model[1], best_model[4]],[best_model[2], best_model[3], best_model[5]]])
        