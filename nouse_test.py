import librosa
from tensorflow import keras
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import matplotlib.image as mpimg

if __name__ == "__main__":
    # load data
    X_lips = np.load("../data_five_recurrent/lips_recurrent_5images_all_chapitres.npy")
    X_tongues = np.load("../data_five_recurrent/tongues_recurrent_5images_all_chapitres.npy")
    Y = np.load("../data_five_recurrent/spectrum_recurrent_all.npy")

    # validation data ch7
    validation_lips = X_lips[-15947:, :, :, :, 0]
    validation_tongues = X_lips[-15947, :, :, :, 0]
