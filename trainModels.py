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

'''
model seetings are based on sklearn tutorial
'''
def standardization(histogram):
    scale = StandardScaler().fit(histogram)
    histogram = scale.transform(histogram)
    return histogram


def Logistic(histogram, labels,k_fold):
    log = LogisticRegression()
    print("Logistic cross validation Mean accuracy:")
    score_log = cross_val_score(log, histogram, labels, cv=k_fold)
    print("\t\tBest Accuracy: %0.2f"%score_log.max())
    print("\t\tMean Accuracy: %0.2f (+/- %0.2f)" % (score_log.mean(), score_log.std() * 2))
    return log, np.mean(score_log)

def Bagging(histogram, labels,k_fold):
    bag = BaggingClassifier()
    print("BaggingClassifier cross validation Mean accuracy:")
    score_bag = cross_val_score(bag, histogram, labels, cv=k_fold)
    print("\t\tBest Accuracy: %0.2f"%score_bag.max())
    print("\t\tMean Accuracy: %0.2f (+/- %0.2f)" % (score_bag.mean(), score_bag.std() * 2))
    return bag, np.mean(score_bag)
    

def MLP(histogram, labels,k_fold):
    mlp = MLPClassifier(hidden_layer_sizes=(256,1024,512,128,))
    print("MLP cross validation Mean accuracy:")
    score_mlp = cross_val_score(mlp, histogram, labels, cv=k_fold)
    print("\t\tBest Accuracy: %0.2f"%score_mlp.max())
    print("\t\tMean Accuracy: %0.2f (+/- %0.2f)" % (score_mlp.mean(), score_mlp.std() * 2))
    return mlp, np.mean(score_mlp)

    
def AdaBoost(histogram, labels,k_fold):
    ada = AdaBoostClassifier(learning_rate=0.01)
    print("AdaBoost cross validation Mean accuracy:")
    score_ada = cross_val_score(ada, histogram, labels, cv=k_fold)
    print("\t\tBest Accuracy: %0.2f"%score_ada.max())
    print("\t\tMean Accuracy: %0.2f (+/- %0.2f)" % (score_ada.mean(), score_ada.std() * 2))
    return ada, np.mean(score_ada)
    

def SVM(histogram, labels,k_fold):
    svm = SVC(probability=True)
    print("SVM cross validation Mean accuracy:")
    score_svm = cross_val_score(svm, histogram, labels, cv=k_fold)
    print("\t\tBest Accuracy: %0.2f"%score_svm.max())
    print("\t\tMean Accuracy: %0.2f (+/- %0.2f)" % (score_svm.mean(), score_svm.std() * 2))
    return svm, np.mean(score_svm)

def RF(histogram, labels,k_fold):
    rf1 = RandomForestClassifier(n_estimators=60, criterion="entropy", oob_score=True, n_jobs=-1)
    rf2 = RandomForestClassifier(n_estimators=60, criterion="entropy", oob_score=False, n_jobs=-1, bootstrap=False)
    score_rf1 = cross_val_score(rf1, histogram, labels, cv=k_fold)
    score_rf2 = cross_val_score(rf2, histogram, labels, cv=k_fold)
    print("Random Forest cross validation Mean accuracy:")
    print("\tWith boosting")
    print("\t\tBest Accuracy: %0.2f" % score_rf1.max())
    print("\t\tMean Accuracy: %0.2f (+/- %0.2f)" % (score_rf1.mean(), score_rf1.std() * 2))
    print("\tWithout boosting")
    print("\t\tBest Accuracy: %0.2f" % score_rf2.max())
    print("\t\tMean Accuracy: %0.2f (+/- %0.2f)" % (score_rf2.mean(), score_rf2.std() * 2))
    return rf1,np.mean(score_rf1), rf2,np.mean(score_rf2)


def fit_clf_Data(clf1,clf2,clf3,histogram, labels):
    print "fitting data to the selected models"
    print "Ensembling"
    eclf1 = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3)],voting='soft')
    eclf1 = eclf1.fit(histogram, labels)
    eclf1_pred = eclf1.predict(histogram)
    print "ensemble acc: ", accuracy_score(labels, eclf1_pred)
    return eclf1, accuracy_score(labels, eclf1_pred),eclf1_pred

def test_clf(clf,histogram,labels):
    print "test data to the model"
    clf_pred = clf.predict(histogram)
    print "test acc: ", accuracy_score(labels, clf_pred)
    return accuracy_score(labels, clf_pred),clf_pred

def recognize(histogram,clf):
    clf_pred = clf.predict(histogram)
    cfl_pred2 = clf.predict_proba(histogram)
    return clf_pred,cfl_pred2
    
    
    
    