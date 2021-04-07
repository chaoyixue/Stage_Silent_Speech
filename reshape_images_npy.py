"""
This file is used to test the reshaping process of the images npy
"""


import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import cv2

if __name__ == "__main__":
    test = np.load("tongues_all_chapiters.npy")
    print(test.shape)
    result = np.zeros((16935, 5, 64, 64, 1))
    for i in range(len(result)):
        result[i, :, :, :, 0] = test[i*5:i*5+5, :, :, 0]
    print(result.shape)
    np.save("five_tongues_all_chapiter.npy", result)
