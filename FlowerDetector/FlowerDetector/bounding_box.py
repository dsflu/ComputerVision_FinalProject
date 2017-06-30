# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:50:37 2017

@author: Xiangwei Shi
"""
import cv2
import numpy as np
import os

def bounding_box(file_path, file_name):
    path = os.path.join(file_path,file_name)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh,None,iterations=2)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = 1
    area = {}
    for c in contours:
        area[i] = cv2.contourArea(c)
        i = i+1
    index = sorted(area.items(),key=lambda item:item[1], reverse= True)
    for key in index:
        max_key = key[0]
        break
    i = 1
    for c in contours:
        if i==max_key:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(img,(x, y), (x + w, y + h), (0, 0,255), 2)
            break
        else:
            i = i+1
    path = os.path.join(file_path,'example.jpg')
    cv2.imwrite(path,img)