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

def standardization(mega_histogram):
    scale = StandardScaler().fit(mega_histogram)
    mega_histogram = scale.transform(mega_histogram)
    return mega_histogram


def Logistic(mega_histogram, labels,k_fold):
    log = LogisticRegression()
    print("Logistic cross validation accuracy:")
    score_log = cross_val_score(log, mega_histogram, labels, cv=k_fold)
    print("\t\tBest: %0.2f"%score_log.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (score_log.mean(), score_log.std() * 2))
    return log, np.median(score_log)

def Bagging(mega_histogram, labels,k_fold):
    bag = BaggingClassifier()
    print("BaggingClassifier cross validation accuracy:")
    score_bag = cross_val_score(bag, mega_histogram, labels, cv=k_fold)
    print("\t\tBest: %0.2f"%score_bag.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (score_bag.mean(), score_bag.std() * 2))
    return bag, np.median(score_bag)
    

def MLP(mega_histogram, labels,k_fold):
    mlp = MLPClassifier(hidden_layer_sizes=(256,1024,512,128,))
    print("MLP cross validation accuracy:")
    score_mlp = cross_val_score(mlp, mega_histogram, labels, cv=k_fold)
    print("\t\tBest: %0.2f"%score_mlp.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (score_mlp.mean(), score_mlp.std() * 2))
    return mlp, np.median(score_mlp)

    
def AdaBoost(mega_histogram, labels,k_fold):
    ada = AdaBoostClassifier(learning_rate=0.01)
    print("AdaBoost cross validation accuracy:")
    score_ada = cross_val_score(ada, mega_histogram, labels, cv=k_fold)
    print("\t\tBest: %0.2f"%score_ada.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (score_ada.mean(), score_ada.std() * 2))
    return ada, np.median(score_ada)
    

def SVM(mega_histogram, labels,k_fold):
    svm_onevsall = SVC(probability=True)
#    svm_onevsall = SVC(cache_size=200, C=180, gamma=0.5, tol=1e-7, shrinking=False, decision_function_shape='ovr')
    svm_onevsone = SVC(cache_size=200, C=180, gamma=0.5, tol=1e-7, shrinking=False, decision_function_shape='ovo')
    print("SVM cross validation accuracy:")
    scores_onevsall = cross_val_score(svm_onevsall, mega_histogram, labels, cv=k_fold)
    print("\tSVM one vs all:")
    print("\t\tBest: %0.2f"%scores_onevsall.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (scores_onevsall.mean(), scores_onevsall.std() * 2))
    scores_onevsone = cross_val_score(svm_onevsone, mega_histogram, labels, cv=k_fold)
    print("\tSVM one vs one:")
    print("\t\tBest: %0.2f"%scores_onevsone.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (scores_onevsone.mean(), scores_onevsone.std() * 2))
    return svm_onevsall, np.median(scores_onevsall),svm_onevsone,np.median(scores_onevsone)

def RF(mega_histogram, labels,k_fold):
    rfb = RandomForestClassifier(n_estimators=60, criterion="entropy", oob_score=True, n_jobs=-1)
    rfn = RandomForestClassifier(n_estimators=60, criterion="entropy", oob_score=False, n_jobs=-1, bootstrap=False)
    scoresrfb = cross_val_score(rfb, mega_histogram, labels, cv=k_fold)
    scoresrfn = cross_val_score(rfn, mega_histogram, labels, cv=k_fold)
    print("Random Forest cross validation accuracy:")
    print("\tWith boosting")
    print("\t\tBest: %0.2f" % scoresrfb.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (scoresrfb.mean(), scoresrfb.std() * 2))
    print("\tWithout boosting")
    print("\t\tBest: %0.2f" % scoresrfn.max())
    print("\t\tAccuracy: %0.2f (+/- %0.2f)" % (scoresrfn.mean(), scoresrfn.std() * 2))
    return rfb,np.median(scoresrfb), rfn,np.median(scoresrfn)


def fit_clf_Data(clf1,clf2,clf3,mega_histogram, labels):
    print "fitting data to the selected models"
#    clf1.fit(mega_histogram, labels)
#    clf2.fit(mega_histogram, labels)
#    clf3.fit(mega_histogram, labels)
    print "Ensembling"
    eclf1 = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2),('clf3', clf3)], voting='soft')
    eclf1 = eclf1.fit(mega_histogram, labels)
    eclf1_pred = eclf1.predict(mega_histogram)
    print "ensemble acc: ", accuracy_score(labels, eclf1_pred)
    return eclf1, accuracy_score(labels, eclf1_pred),eclf1_pred

def test_clf(clf,mega_histogram,labels):
    print "test data to the model"
    clf_pred = clf.predict(mega_histogram)
    print "test acc: ", accuracy_score(labels, clf_pred)
    return accuracy_score(labels, clf_pred),clf_pred

def recognize(mega_histogram,clf):
    clf_pred = clf.predict(mega_histogram)
    cfl_pred2 = clf.predict_proba(mega_histogram)
    return clf_pred,cfl_pred2
    
    
    
    