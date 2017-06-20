import cv2
import sys
from time import time
from os.path import isfile
from glob import glob
import numpy as np
import pandas as pd
import bow as bow
import trainModels as tm
'''
Todo:
    1. training data
    2. google cloud


'''

def train_Model(images, nlabels, bowdata, k_size, n_images):
    Scores = {}
    if not isfile(bowdata+".npy"):
        x = time()
        mega_histogram, v = bow.bag_of_words(images, k_size,n_images)
        np.save(file=bowdata, arr=mega_histogram)
        y = time()
        print("Create BOW: ", y - x, ".s")
    else:
        mega_histogram = np.load(bowdata+".npy")
        print(".s")
    print ("enter classification")
    mega_histogram = tm.standardization(mega_histogram)
    clf1, score1, clf2, score2 = tm.SVM(mega_histogram, nlabels,5)
    clf3, score3, clf4, score4 = tm.RF(mega_histogram, nlabels,5)
    print ("end classification")
    print ("enter ensemble")
    Scores[clf1] = score1
    Scores[clf2] = score2
    Scores[clf3] = score3
    Scores[clf4] = score4

    S = sorted(Scores.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    
    eclf1 = S[0][0]
    eclf2 = S[1][0]
    
    eclf,err,pred = tm.fit_clf_Data(eclf1,eclf2,mega_histogram, labels)
    print ("end ensemble")
    
    return eclf,err,pred
    
#def recognize_Pic(self,test_img, test_image_path=None):
#    ''' '''
#    
#def test_Model(classifier):
    

if __name__ == '__main__':
    
    images = []
    count = 0
    path = './training_data/*.jpg'
    train_id = pd.read_csv('csv/train_id.csv',header=None)
    train_list = train_id[0].tolist()
    images = [cv2.imread('training_data/image_' + '%0*d' % (5, i) + '.jpg',
                         flags=cv2.IMREAD_COLOR) for i in train_list]
    
    
#    for img in glob(path):
#        im = cv2.imread(img, flags=cv2.IMREAD_COLOR)
#        images.append(im)
#        count +=1 
    labels_data = pd.read_csv('csv/label.csv',header=None)
    total_labels = labels_data[0].tolist()
    labels =[]
    labels = [total_labels[i-1] for i in train_list]
    count = len(images)
             
    k_list = [3]
    print ('phase 1 done')
    for k in k_list:
        print("\n\nK = " + str(k))
        bowdata = "bowdata/bow_"+"k_"+str(k)
#        rfilename = "doc/img/shape"+"sift"+"_"+str(k)
        eclf,err,pred = train_Model(images=images, nlabels=labels, bowdata=bowdata, k_size=k, n_images = count)
#        test_Model(clf)
    fileObject1 = open('truelabels.txt', 'w') 
    for ip in labels:  
         fileObject1.write(str(ip))
         fileObject1.write('\n')
    fileObject1.close()
    
    fileObject = open('sampleList.txt', 'w') 
    for ip in pred:  
         fileObject.write(str(ip))
         fileObject.write('\n')
    fileObject.close()
              
        
        
        
        
        
        