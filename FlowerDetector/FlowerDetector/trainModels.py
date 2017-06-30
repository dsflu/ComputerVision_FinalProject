#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:32:09 2017

@author: fredlu
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def standardization(histogram):
    scale = StandardScaler().fit(histogram)
    histogram = scale.transform(histogram)
    return histogram
def recognize(histogram,clf):
    clf_pred = clf.predict(histogram)
    cfl_pred2 = clf.predict_proba(histogram)
    return clf_pred,cfl_pred2
    
    
    
    