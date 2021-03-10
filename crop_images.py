# -*- coding: utf-8 -*-
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def crop_levre_images(filepath, longeur):
    pass 


def crop_langue_images(filepath, longeur):
    pass
    
    

if __name__ == "__main__":
    img = mpimg.imread("../data_2021/ch1_en/levre/8081.bmp")
    plt.figure()
    plt.imshow(img)
    plt.show() 
    