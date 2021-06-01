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
    X_lips = np.load("../data_npy_one_image/lips_all_chapiters.npy")
    X_tongues = np.load("../data_npy_one_image/tongues_all_chapiters.npy")
    Y = np.load("../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")

    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_spectrum = np.max(Y)
    Y = Y / max_spectrum

    # split train set data
    lips_x_train = X_lips[:-15951]
    lips_x_test = X_lips[-15951:]
    tongues_x_train = X_tongues[:-15951]
    tongues_x_test = X_tongues[-15951:]
    y_train = Y[:, :-15951]
    y_test = Y[:, -15951:]
    y_train = np.matrix.transpose(y_train)
    y_test = np.matrix.transpose(y_test)

    model = keras.models.load_model("../ssi_model1_val_loss-0.00004934.h5")
    test_result = model.predict([lips_x_test, tongues_x_test])
    result = np.matrix.transpose(test_result)

    result = result * max_spectrum
    """
    test_reconstruit = librosa.griffinlim(result, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_by_images.wav", test_reconstruit, 44100)
    """

    fig, ax = plt.subplots(nrows=2)
    img = librosa.display.specshow(librosa.amplitude_to_db(np.matrix.transpose(y_test),
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='linear', x_axis='time', ax=ax[0])
    ax[0].set_title('original spectrum')
    librosa.display.specshow(librosa.amplitude_to_db(result, ref=np.max), sr=44100, hop_length=735,
                             y_axis='linear', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

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
    """
