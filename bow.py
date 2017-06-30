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

'''
compute bag of words based on the CV lab assignment
While instead of using ramdon patches, we use the sift descriptors to
construct the codebook.
'''
def bag_of_words(images, k, n_images):
    print ("enter sift")
    clf_k = KMeans(n_clusters = k)
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    
    # compute the sift descriptors
    if not isfile("desdata/descriptors_hsv.npy"):
        for pic in images:
            kp, des= sift.detectAndCompute(pic, None)
            descriptor_list.append(des)
        np.save(file="desdata/descriptors_hsv", arr=descriptor_list)
    else:
        descriptor_list = np.load("desdata/descriptors_hsv.npy")
    print ("finish computing sift for all the images")
    # do the k-means clustering to get the codebook
    print ("enter kmeans")
    time1 = time()
    np_des = np.array(descriptor_list[0])
    for remaining in descriptor_list[1:]:
        np_des = np.vstack((np_des, remaining))
    descriptor_np_des = np_des.copy()
    clusters = clf_k.fit_predict(descriptor_np_des)
    time2 = time()
    print ("finishing kmeans")
    print("Kmeans time: ", time2 - time1, ".s")
    joblib.dump(clf_k, "trainedModel/kmeans_hsv_"+str(k)+".m")
    
    # recomputer the feature histogram for each image based on the codebook
    print ("enter computing histogram")
    
    histogram = np.array([np.zeros(k) for i in range(n_images)])
    index = 0
    for i in range(n_images):
        l = len(descriptor_list[i])
        for j in range(l):
            index2 = clusters[index+j]
            histogram[i][index2] += 1
        index += l
    print "Histogram finished"
    
    return histogram, np_des

def bag_of_words_test(images, k):
    clf_k = joblib.load("trainedModel/kmeans_hsv_"+str(k)+".m")
    n_images = len(images)
    print ("test: enter sift")
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    if not isfile("desdata/descriptors_test_hsv.npy"):
        for pic in images:
            kp, des= sift.detectAndCompute(pic, None)
            descriptor_list.append(des)
#        np.save(file="desdata/descriptors_test", arr=descriptor_list)
    else:
        descriptor_list = np.load("desdata/descriptors_test_hsv.npy")
    print ("test: finish computing sift for all the images")
    print ("test: enter kmeans")
    time1 = time()
    np_des = np.array(descriptor_list[0])
    for remaining in descriptor_list[1:]:
        np_des = np.vstack((np_des, remaining))
    descriptor_np_des = np_des.copy()
    print ("test: start kmeans prediction")
    clusters = clf_k.predict(descriptor_np_des)
    time2 = time()
    print ("test: finish kmeans")
    print("test: Kmeans time: ", time2 - time1, ".s")
    print ("test: enter computing histogram")
    
    histogram = np.array([np.zeros(k) for i in range(n_images)])
    index = 0
    for i in range(n_images):
        l = len(descriptor_list[i])
        for j in range(l):
            index2 = clusters[index+j]
            histogram[i][index2] += 1
        index += l
    print "test: Histogram finished"
    
    return histogram, np_des

def bag_of_words_rec(image, k):
    clf_k = joblib.load("trainedModel/kmeans_hsv_"+str(k)+".m")
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des= sift.detectAndCompute(image, None)
    clusters = clf_k.predict(des)
    histogram = np.array( [ 0 for i in range(k)])
    for iterm in clusters:
        histogram[iterm] += 1
    return histogram
    
    
    
    
    