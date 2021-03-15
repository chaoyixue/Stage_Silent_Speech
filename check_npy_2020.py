# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:27:21 2021

@author: Pierre
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
train_lips = np.load("levre.npy")
train_tongue = np.load("langue.npy")
print(train_lips.shape)
print(train_tongue.shape)

plt.figure()
for i in tqdm(range(train_lips.shape[0])):
    plt.imshow(train_lips[i, :, :])
    plt.pause(1)



