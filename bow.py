import cv2
import numpy as np
from sklearn.cluster import KMeans

def bag_of_words(images, k, n_images):
    print ("enter sift")
    # create the codebook
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
#    BOW = cv2.BOWKMeansTrainer(k)
#    kp = []
#    des_list = []
    for pic in images:
        kp, des= sift.detectAndCompute(pic, None)
        descriptor_list.append(des)
        
    vStack = np.array(descriptor_list[0])
    for remaining in descriptor_list[1:]:
        vStack = np.vstack((vStack, remaining))
    descriptor_vstack = vStack.copy()
    kmeans_ret = KMeans(n_clusters = k).fit_predict(descriptor_vstack)
    
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
        