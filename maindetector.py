import cv2
import sys
from time import time
from os.path import isfile
from glob import glob
import numpy as np
import pandas as pd
import bow as bow
import trainModels as tm


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
        
    mega_histogram = tm.standardization(mega_histogram)
    clf1, score1, clf2, score2 = tm.SVM(mega_histogram, nlabels,5)
    clf3, score3, clf4, score4 = tm.RF(mega_histogram, nlabels,5)
    
    Scores[clf1] = score1
    Scores[clf2] = score2
    Scores[clf3] = score3
    Scores[clf4] = score4

    S = sorted(Scores.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    
    eclf1 = S[0][0]
    eclf2 = S[1][0]
    
    eclf,err = tm.fit_clf_Data(eclf1,eclf2,mega_histogram, labels)
    
    return eclf,err
    
#def recognize_Pic(self,test_img, test_image_path=None):
#    ''' '''
#    
#def test_Model(classifier):
    

if __name__ == '__main__':
    
    images = []
    count = 0
    path = './test/*.jpg'
    for img in glob(path):
        im = cv2.imread(img, flags=cv2.IMREAD_COLOR)
        images.append(im)
        count +=1 
    labels_data = pd.read_csv('csv/fakelabel.csv',header=None)
    labels = labels_data[0].tolist()
             
    k_list = [3]
    print ('phase 1 done')
    for k in k_list:
        print("\n\nK = " + str(ks))
        bowdata = "bowdata/bow_"+"k_"+str(k)
#        rfilename = "doc/img/shape"+"sift"+"_"+str(k)
        eclf,err = train_Model(images=images, nlabels=labels, bowdata=bowdata, k_size=k, n_images = count)
#        test_Model(clf)
        
        
        
        
        
        