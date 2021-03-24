# -*- coding: utf-8 -*-
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import cv2


def crop_images(filepath, longeur, height_bot, height_top, width_left, width_right, output_name):
    
    """
    crop all the images in a folder and save the images as npy file

    :param width_right:
    :param width_left:
    :param height_top:
    :param height_bot:
    :param filepath: the folder that contains images
    :param longeur: the number of images in the folder
    :param output_name: the name of the output file npy
    :return:
    """
    
    data_npy_list = []
    for index_image in tqdm(range(longeur)):
        myimg = mpimg.imread(filepath + "/" + str(index_image+1) + ".bmp")
        myimg = myimg[height_bot:height_top, width_left:width_right]
        myimg = cv2.resize(myimg, (64, 64), interpolation=cv2.INTER_AREA)
        data_npy_list.append(myimg)
    img_npy = np.array(data_npy_list)
    np.save(output_name, img_npy)
    return img_npy


if __name__ == "__main__":
    # crop the levre images
    levre_npy = crop_images("../data_2021/ch7_en/levre", 15951, 0, 450, 100, 650, "lip_ch7.npy")
    print(levre_npy.shape)
    
    # crop the langue images
    langue_npy = crop_images("../data_2021/ch7_en/langue", 15951, 10, 220, 10, 280, "tongue_ch7.npy")
    print(langue_npy.shape)
