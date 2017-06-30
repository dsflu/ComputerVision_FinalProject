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
import FlowerDetector.bow as bow
import FlowerDetector.trainModels as tm
from sklearn.externals import joblib
import random

def recognizeImage(image, k_size,clf):
    histogram = bow.bag_of_words_rec(image, k_size)
    histogram = tm.standardization(histogram)
    pred = tm.recognize(histogram,clf)
    return pred

