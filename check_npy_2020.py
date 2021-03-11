# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:27:21 2021

@author: Pierre
"""

import numpy as np

train_lips = np.load("../data/train_test_npy_2020/train_lips.npy")
train_tongue = np.load("../data/train_test_npy_2020/train_tongue.npy")
train_label = np.load("../data/train_test_npy_2020/train_label.npy")
print(train_lips.shape)
print(train_tongue.shape)
print(train_label.shape)