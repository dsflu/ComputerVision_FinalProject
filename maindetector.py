#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:05:32 2017

@author: fredlu
"""

import cv2
import sys
from time import time
from os.path import isfile
from glob import glob
import numpy as np
import pandas as pd
import bow as bow
import trainModels as tm
from sklearn.externals import joblib
'''
Todo:
    1. training data
    2. google cloud


'''

def train_Model(images, nlabels, bowdata, k_size, n_images):
    Scores = {}
    k_fold = 10
    if not isfile(bowdata+".npy"):
        time1 = time()
        mega_histogram, v = bow.bag_of_words(images, k_size,n_images)
        np.save(file=bowdata, arr=mega_histogram)
        time2 = time()
        print("Create BOW: ", time2 - time1, ".s")
    else:
        mega_histogram = np.load(bowdata+".npy")
        print(".s")
    print ("enter classification")
    mega_histogram = tm.standardization(mega_histogram)
    clf1, score1, clf2, score2 = tm.SVM(mega_histogram, nlabels,k_fold)
    clf3, score3, clf4, score4 = tm.RF(mega_histogram, nlabels,k_fold)
    clf5, score5 = tm.AdaBoost(mega_histogram, nlabels,k_fold)
    clf6, score6 = tm.MLP(mega_histogram, nlabels,k_fold)
    clf7, score7 = tm.Bagging(mega_histogram, nlabels,k_fold)
    print ("end classification")
    print ("enter ensemble")
    Scores[clf1] = score1
    Scores[clf2] = score2
    Scores[clf3] = score3
    Scores[clf4] = score4
    Scores[clf5] = score5
    Scores[clf6] = score6
    Scores[clf7] = score7

    S = sorted(Scores.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    
    eclf1 = S[0][0]
    eclf2 = S[1][0]
    eclf3 = S[2][0]
    
    eclf,score,pred = tm.fit_clf_Data(eclf1,eclf2,eclf3,mega_histogram, labels)
    print ("end ensemble")
    
    return eclf,score,pred
       
def test_Model(images,nlabels,bowdata, k_size, clf):
    if not isfile(bowdata+".npy"):
        x = time()
        mega_histogram, v = bow.bag_of_words_test(images, k_size)
        np.save(file=bowdata, arr=mega_histogram)
        y = time()
        print("Create BOW: ", y - x, ".s")
    else:
        mega_histogram = np.load(bowdata+".npy")
        print(".s")
    mega_histogram = tm.standardization(mega_histogram)
    print ("enter clf test")
    score,pred = tm.test_clf(clf,mega_histogram, labels)
    return pred

def recognizeImage(image, k_size,clf):
    mega_histogram = bow.bag_of_words_rec(image, k_size)
    mega_histogram = tm.standardization(mega_histogram)
    pred = tm.recognize(mega_histogram,clf)
    return pred

if __name__ == '__main__':
    
    flag_train_test = 0
    
    if flag_train_test:
        images = []
        count = 0
        path = './training_data/*.jpg'
        train_id = pd.read_csv('csv/train_id.csv',header=None)
        test_id = pd.read_csv('csv/validation_id.csv',header=None)
        train_list = train_id[0].tolist()
        test_list = test_id[0].tolist()
        images = [cv2.imread('training_data/image_' + '%0*d' % (5, i) + '.jpg',
                             flags=cv2.IMREAD_COLOR) for i in train_list]
        test_images = [cv2.imread('training_data/image_' + '%0*d' % (5, i) + '.jpg',
                             flags=cv2.IMREAD_COLOR) for i in test_list]
        
        image_rec = cv2.imread('recognize/image_00001.jpg')
        
    #    for img in glob(path):
    #        im = cv2.imread(img, flags=cv2.IMREAD_COLOR)
    #        images.append(im)
    #        count +=1 
        labels_data = pd.read_csv('csv/label.csv',header=None)
        total_labels = labels_data[0].tolist()
        labels =[]
        test_labels = []
        labels = [total_labels[i-1] for i in train_list]
        test_labels = [total_labels[i-1] for i in test_list]
        count = len(images)
                 
        k = 50
        print ('phase 1 done')
        
        print("\n\nK = " + str(k))
        bowdata = "bowdata/bow_"+"k_"+str(k)
        test_bowdata = "bowdata/bow_test_"+"k_"+str(k)
    #        rfilename = "doc/img/shape"+"sift"+"_"+str(k)
        eclf,score,train_pred = train_Model(images=images, nlabels=labels, bowdata=bowdata, k_size=k, n_images = count)
        test_pred = test_Model(images=test_images,nlabels=test_labels,bowdata=test_bowdata, k_size=k, clf = eclf)
        joblib.dump(eclf, "trainedModel/eclf_"+str(k)+".m")
        
    else:
        
        k = 50
        eclf = joblib.load("trainedModel/eclf_"+str(k)+".m")
        pred,pred2 = recognizeImage(image_rec, k, eclf)
        print ('\n\nThe predicted label is: '+str(pred)+'\n'+'Confidence is: '+str(pred2[0][pred[0]-1])+'\n\n')
        
#    fileObject1 = open('True_Labels.txt', 'w') 
#    for ip in labels:  
#         fileObject1.write(str(ip))
#         fileObject1.write('\n')
#    fileObject1.close()
#    
#    fileObject = open('Predicted_Lables.txt', 'w') 
#    for ip in test_pred:  
#         fileObject.write(str(ip))
#         fileObject.write('\n')
#    fileObject.close()
              
        
        
        
        
        
        