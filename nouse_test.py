import librosa
from tensorflow import keras
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import matplotlib.image as mpimg


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
    # load data
    X_lips = np.load("../data_npy_one_image/lips_all_chapiters.npy")
    X_tongues = np.load("../data_npy_one_image/tongues_all_chapiters.npy")
    Y = np.load("../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    print("aaa")

    #
    lips_ch7 = X_lips[-15951:, :, :, :]
    tongues_ch7 = X_tongues[-15951:, :, :, :]
    Y_ch7 = Y[:, -15951:]

    # recurrent ch7
    lips_recurrent_ch7 = reshape_npy_to_recurrent(lips_ch7, 15951, 5)
    tongue_recurrent_ch7 = reshape_npy_to_recurrent(tongues_ch7, 15951, 5)
    validation_spectre = Y_ch7[:, 2:-2]

    np.save("lips_validation_ch7.npy", lips_recurrent_ch7)
    np.save("tongues_validation_ch7.npy", tongue_recurrent_ch7)
    np.save("spectrum_validation.npy", validation_spectre)
