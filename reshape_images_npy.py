"""
This file is used to test the reshaping process of the images npy to shape (N', 5, 64, 64,1)
"""


import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import cv2


def reshape_npy_to_recurrent(npy_chapitre, longueur_original, longueur_sequence):
    """
    reshape the original array from  N,64,64,1 to N-4, 5, 64, 64 ,1

    for example
    10 images N = 10 10,64,64,1
    longeur sequence = 5
    reshape to 6,5,64,64,1
    12345
    23456
    34567
    45678
    56789
    678910     ->  10-5+1, 5, 64, 64, 1

    :param npy_chapitre: the numpy array with shape N,64,64,1
    :param longueur_original: N
    :param longueur_sequence: the number of images in one sequence
    :return: a np array with shape (longeur_original-longeur_sequence+1,longueur_sequence,64,64,1)
    """
    result = np.zeros((longueur_original-longueur_sequence+1, longueur_sequence, 64, 64, 1))
    for i in range(len(result)):
        result[i, :, :, :] = npy_chapitre[i:i+longueur_sequence, :, :, :]
    print(result.shape)
    return result


if __name__ == "__main__":
    test = np.load("../data_npy_one_image/tongues_all_chapiters.npy")
    chapitre_1 = test[:10054, :, :, :]
    result_chapitre_1 = reshape_npy_to_recurrent(chapitre_1, 10054, 5)
    np.save("../five_recurrent_image_npy/tongues/ch1.npy", result_chapitre_1)

    chapitre_2 = test[10054:10054+14441, :, :, :]
    result_chapitre_2 = reshape_npy_to_recurrent(chapitre_2, 14441, 5)
    np.save("../five_recurrent_image_npy/tongues/ch2.npy", result_chapitre_2)

    chapitre_3 = test[24495:24495+8885, :, :, :]
    result_chapitre_3 = reshape_npy_to_recurrent(chapitre_3, 8885, 5)
    np.save("../five_recurrent_image_npy/tongues/ch3.npy", result_chapitre_3)

    chapitre_4 = test[33380:33380+15621, :, :, :]
    result_chapitre_4 = reshape_npy_to_recurrent(chapitre_4, 15621, 5)
    np.save("../five_recurrent_image_npy/tongues/ch4.npy", result_chapitre_4)

    chapitre_5 = test[49001:49001+14553, :, :, :]
    result_chapitre_5 = reshape_npy_to_recurrent(chapitre_5, 14553, 5)
    np.save("../five_recurrent_image_npy/tongues/ch5.npy", result_chapitre_5)

    chapitre_6 = test[63554:63554+5174, :, :, :]
    result_chapitre_6 = reshape_npy_to_recurrent(chapitre_6, 5174, 5)
    np.save("../five_recurrent_image_npy/tongues/ch6.npy", result_chapitre_6)

    chapitre_7 = test[68728:68728+15951, :, :, :]
    result_chapitre_7 = reshape_npy_to_recurrent(chapitre_7, 15951, 5)
    np.save("../five_recurrent_image_npy/tongues/ch7.npy", result_chapitre_7)

    all_chapitre = np.concatenate((result_chapitre_1, result_chapitre_2, result_chapitre_3,
                                   result_chapitre_4, result_chapitre_5, result_chapitre_6,
                                   result_chapitre_7), axis=0)
    print(all_chapitre.shape)
    np.save("../five_recurrent_image_npy/tongues/tongues_recurrent_5images_all_chapitres.npy", all_chapitre)