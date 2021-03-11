# -*- coding: utf-8 -*-
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm


def crop_levre_images(filepath, longeur,height_bot, height_top, width_left, width_right, output_name):
    """

    :param width_right:
    :param width_left:
    :param height_top:
    :param height_bot:
    :param filepath:
    :param longeur:
    :param output_name:
    :return:
    """
    data_npy_list = []
    for index_image in tqdm(range(longeur)):
        mylevre = mpimg.imread(filepath + str(index_image+1) + ".bmp")
        mylevre = mylevre[:]
    pass


def crop_langue_images(filepath, longeur, output_name):
    """

    :param filepath:
    :param longeur:
    :param output_name:
    :return:
    """
    pass
    

if __name__ == "__main__":
    img = mpimg.imread("../data_2021/ch1_en/levre/8081.bmp")
    print(img.shape)
    plt.figure()
    plt.imshow(img)
    plt.show() 
    