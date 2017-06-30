# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:50:37 2017

@author: Xiangwei Shi
"""
from sklearn.externals import joblib
import cv2
from FlowerDetector.maindetector import recognizeImage

address = "/Users/fredlu/Developer/ComputerVision/FlowerDetector/FlowerDetector/trainedModel"

def detection(img_path):
	
	k = 200
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	eclf = joblib.load(address+"/eclf_"+str(k)+".m")
	pred,pred2 = recognizeImage(img,k,eclf)

	return pred, str(pred2[0][pred[0]-1])