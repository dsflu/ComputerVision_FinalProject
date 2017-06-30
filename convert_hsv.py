#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:47:50 2017

@author: fredlu
"""

import cv2
import numpy as np

'''
this method is based on 
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
'''
def convert_hsv(images, K=16, criteria=(cv2.TERM_CRITERIA_EPS +
                                          cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)):
    hsv_stack = []
    count=1
    for image in images:
        data = cv2.cvtColor(src=image,code=cv2.COLOR_RGB2HSV).reshape(-1,3)
        data = np.float32(data)
        ret, label, center = cv2.kmeans(data, K, None, criteria, 10,
                                        cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        hsv_image = res.reshape((image.shape))
        hsv_stack.append(hsv_image)
        cv2.imwrite("training_hsv/image_"+ '%0*d' % (5, count) + '.jpg', hsv_image)
#        print 'finish converting '+str(i)+' th pic'
        count = count+1
        
    print 'finish converting'
    return np.array(hsv_stack)