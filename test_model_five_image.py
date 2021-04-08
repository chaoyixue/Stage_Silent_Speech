"""
This file is used to test models which uses 5images as input

"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import librosa
import soundfile as sf
import librosa.display
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # load data
    X_lips = np.load("../data_five_images_0408/five_lips_all_chapiter.npy")
    X_tongues = np.load("../data_five_images_0408/five_tongues_all_chapiter.npy")
    Y = np.load("../data_five_images_0408/spectrogrammes_5_images_all_chapitre.npy")

    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_spectrum = np.max(Y)
    Y = Y / max_spectrum

    # split train set data
    lips_x_train = X_lips[:-3190]
    lips_x_test = X_lips[-3190:]
    tongues_x_train = X_tongues[:-3190]
    tongues_x_test = X_tongues[-3190:]
    y_train = Y[:, :-3190]
    y_test = Y[:, -3190:]
    y_train = np.transpose(y_train)
    y_test = np.transpose(y_test)

    model = keras.models.load_model("../ssi_model7-36-0.00005596.h5")
    model.summary()
    test_result = model.predict([lips_x_test, tongues_x_test])
    result = np.transpose(test_result)

    result = result * max_spectrum

    fig, ax = plt.subplots(nrows=2)
    img = librosa.display.specshow(librosa.amplitude_to_db(np.transpose(y_test),
                                                           ref=np.max), sr=44100, hop_length=3675,
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('original spectrum')
    librosa.display.specshow(librosa.amplitude_to_db(result, ref=np.max), sr=44100, hop_length=3675,
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()