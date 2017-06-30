#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 21:24:59 2017

@author: fredlu
"""

import pandas as pd

if __name__ == '__main__':
    labels_data = pd.read_csv('csv/label.csv',header=None)
    label = labels_data[labels_data[0] == 8]
    