#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:23:11 2017

@author: fredlu
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from os.path import isfile
from sklearn.externals import joblib
from time import time

#change this address into "your/path/to/FlowerDetector/FlowerDetector/trainedModel/kmeans_"
address = "/Users/fredlu/Developer/ComputerVision/FlowerDetector/FlowerDetector/trainedModel/kmeans_"
def bag_of_words_rec(image, k):
    clf_k = joblib.load(address+str(k)+".m")
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des= sift.detectAndCompute(image, None)
    clusters = clf_k.predict(des)
    histogram = np.array( [ 0 for i in range(k)])
    for iterm in clusters:
        histogram[iterm] += 1
    return histogram
    
    
    
    
    