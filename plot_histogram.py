#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 19:15:19 2017

@author: fredlu
"""

from matplotlib import pyplot as plt
import numpy as np 

def plot(k,histogram):
    
		x_scalar = np.arange(k)
		y_scalar = np.array([abs(np.sum(histogram[:,h], dtype=np.int32)) for h in range(k)])

		print y_scalar

		plt.bar(x_scalar, y_scalar)
		plt.xlabel("Visual Word Index")
		plt.ylabel("Frequency")
		plt.title("Complete Vocabulary Generated")
		plt.xticks(x_scalar + 0.4, x_scalar)
		plt.show()

if __name__ == '__main__':
    
    k = 50
    bowdata = "bowdata/bow_"+ str(k)
    histogram = np.load(bowdata+".npy")
    plot(k,histogram)