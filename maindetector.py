import cv2
import sys
from time import time
from os.path import isfile
from glob import glob
import numpy as np
import pandas as pd
import bow as bow


def train_Model(images, nlabels, bowdata, k_size, n_images):
    if not isfile(bowdata+".npy"):
        x = time()
        mega_histogram = bow.bag_of_words(images, k_size,n_images)
        y = time()
        print("Create BOW: ", y - x, ".s")
#        w = time()
        return mega_histogram
#def recognize_Pic(self,test_img, test_image_path=None):
#    ''' '''
#    
#def test_Model(self):

if __name__ == '__main__':
    
    images = []
    count = 0
    path = './test/*.jpg'
    for img in glob(path):
        im = cv2.imread(img, flags=cv2.IMREAD_COLOR)
        images.append(im)
        count +=1 
    labels_data = pd.read_csv('csv/label.csv',header=None)
    labels = labels_data[0].tolist()
             
    k_list = [3]
    print ('phase 1 done')
    for k in k_list:
        print("\n\nK = " + str(ks))
        bowdata = "bowdata/bow_"+"k_"+str(k)
#        rfilename = "doc/img/shape"+"sift"+"_"+str(k)
        mega_histogram = train_Model(images=images, nlabels=labels, bowdata=bowdata, k_size=k, n_images = count)