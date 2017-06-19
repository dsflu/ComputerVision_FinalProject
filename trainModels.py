#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:32:09 2017

@author: fredlu
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



def standardization(mega_histogram):
    scale = StandardScaler().fit(mega_histogram)
    mega_histogram = scale.transform(mega_histogram)
    return mega_histogram

def SVM(mega_histogram, labels):
    print "Using SVM as classification"
    classifier = SVC()
    classifier.fit(mega_histogram, labels)
    print "SVM finished"
    return classifier