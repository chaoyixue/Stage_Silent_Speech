"""
This file is used to test the autoencoder model trained to check out if it generates a spectrum correct.
By using griffin lim, test if  the wav file reconstructed is clair.

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
    X = np.load("spectrogrammes_all_chapitre.npy")
    max_value = np.max(X)
    print(max_value)
    # normalisation
    X = X / max_value
    print(X.max())
    print(X.min())

    # split train test data
    x_train = np.transpose(X[:, :84776 - 15951])
    x_test = np.transpose(X[:, -15951:])  # changed to shape (15951, 736) for the training process
    print(x_train.shape)
    print(x_test.shape)
    model = keras.models.load_model("../model3_adam/weights-improvement-199-0.00000230.h5")
    model.summary()
    test_result = model.predict(x_test)
    # changed to shape (736, 15951) for the figure of spectrum and the reconstruction
    test_result = np.transpose(test_result)
    print(test_result.shape)
    x_test = np.transpose(x_test)

    ####################################################################################################################
    """
    show the spectrum original and the spectrum learned
    """

    """
    fig, ax = plt.subplots(nrows=2)

    img = librosa.display.specshow(librosa.amplitude_to_db(x_test,
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('Power spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(test_result, ref=np.max), sr=44100, hop_length=735,
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    """

    ####################################################################################################################
    """
    reconstruction of the signal and save it into a wav file
    """
    # denormalisation
    test_result = test_result * max_value
    # reconstruction using griffin lim
    test_reconstruit = librosa.griffinlim(test_result, hop_length=735, win_length=735 * 2)

    sf.write("ch7_reconstruit.wav", test_reconstruit, 44100)

    ####################################################################################################################
    """
    show and compare two wav files in the domaine of time
    """

    sound_original = librosa.load("../data/20200617_153719_RecFile_1_bruce_ch7"
                                  "/RecFile_1_20200617_153719_Sound_Capture_DShow_5_monoOutput1.wav", sr=44100)
    fig, ax = plt.subplots(nrows=2)
    librosa.display.waveplot(sound_original[0], sr=44100, color='b', ax=ax[0])
    ax[0].set(title='Original', xlabel=None)
    ax[0].label_outer()
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("Amplitude V")
    librosa.display.waveplot(test_reconstruit, sr=44100, color='r', ax=ax[1])
    ax[1].set(title='reconstruction', xlabel=None)
    ax[1].label_outer()
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Amplitude V")
    plt.show()


