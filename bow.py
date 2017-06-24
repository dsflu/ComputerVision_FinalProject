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

def bag_of_words(images, k, n_images):
    print ("enter sift")
    clf_k = KMeans(n_clusters = k)
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()

    if not isfile("desdata/descriptors.npy"):
        for pic in images:
            kp, des= sift.detectAndCompute(pic, None)
            descriptor_list.append(des)
        np.save(file="desdata/descriptors", arr=descriptor_list)
    else:
        descriptor_list = np.load("desdata/descriptors.npy")
    print ("finish computing sift for all the images")
    print ("enter kmeans")
    time1 = time()
    vStack = np.array(descriptor_list[0])
    for remaining in descriptor_list[1:]:
        vStack = np.vstack((vStack, remaining))
    descriptor_vstack = vStack.copy()
    kmeans_ret = clf_k.fit_predict(descriptor_vstack)
    time2 = time()
    print ("finishing kmeans")
    print("Kmeans time: ", time2 - time1, ".s")
    joblib.dump(clf_k, "trainedModel/kmeans_"+str(k)+".m")
    
    print ("enter computing histogram")
    
    mega_histogram = np.array([np.zeros(k) for i in range(n_images)])
    old_count = 0
    for i in range(n_images):
        l = len(descriptor_list[i])
        for j in range(l):
            idx = kmeans_ret[old_count+j]
            mega_histogram[i][idx] += 1
        old_count += l
    print "Vocabulary Histogram Generated"
    
    return mega_histogram, vStack

def bag_of_words_test(images, k):
    clf_k = joblib.load("trainedModel/kmeans_"+str(k)+".m")
    n_images = len(images)
    print ("test: enter sift")
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    if not isfile("desdata/descriptors_test_300.npy"):
        for pic in images:
            kp, des= sift.detectAndCompute(pic, None)
            descriptor_list.append(des)
        np.save(file="desdata/descriptors_test_300", arr=descriptor_list)
    else:
        descriptor_list = np.load("desdata/descriptors_test_300.npy")
    print ("test: finish computing sift for all the images")
    print ("test: enter kmeans")
    time1 = time()
    vStack = np.array(descriptor_list[0])
    for remaining in descriptor_list[1:]:
        vStack = np.vstack((vStack, remaining))
    descriptor_vstack = vStack.copy()
    print ("test: start kmeans prediction")
    kmeans_ret = clf_k.predict(descriptor_vstack)
    time2 = time()
    print ("test: finish kmeans")
    print("test: Kmeans time: ", time2 - time1, ".s")
    print ("test: enter computing histogram")
    
    mega_histogram = np.array([np.zeros(k) for i in range(n_images)])
    old_count = 0
    for i in range(n_images):
        l = len(descriptor_list[i])
        for j in range(l):
            idx = kmeans_ret[old_count+j]
            mega_histogram[i][idx] += 1
        old_count += l
    print "test: Vocabulary Histogram Generated"
    
    return mega_histogram, vStack

def bag_of_words_rec(image, k):
    clf_k = joblib.load("trainedModel/kmeans_"+str(k)+".m")
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des= sift.detectAndCompute(image, None)
    kmeans_ret = clf_k.predict(des)
    mega_histogram = np.array( [ 0 for i in range(k)])
    for iterm in kmeans_ret:
        mega_histogram[iterm] += 1
    return mega_histogram
    
    
    
    
    