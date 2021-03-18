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
    x_test = np.transpose(X[:, -15951:])
    print(x_train.shape)
    print(x_test.shape)
    model = keras.models.load_model("../autoencoder_model_0316/weights-improvement-50-0.00.h5")
    model.summary()
    test_result = model.predict(x_test)
    test_result = np.transpose(test_result)
    print(test_result.shape)
    # show the spectrum original and the spectrum learned
    fig, ax = plt.subplots(nrows=2, sharex="True", sharey="True")
    x_test = np.transpose(x_test)
    print(test_result[:20, 5000])
    print(test_result[:20, 4000])
    print(x_test[:20, 5000])
    img = librosa.display.specshow(librosa.amplitude_to_db(x_test,
                                                           ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('Power spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(test_result, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    # test_reconstruit = librosa.griffinlim(test_result, hop_length=735, win_length=735 * 2)

    # sf.write("ch7_reconstruit.wav", test_reconstruit, 44100)

    """
    sound_original = librosa.load("../data/20200617_153719_RecFile_1_bruce_ch7"
                                  "/RecFile_1_20200617_153719_Sound_Capture_DShow_5_monoOutput1.wav", sr=44100)
    fig, ax = plt.subplots(nrows=2, sharex=False, sharey=False)
    librosa.display.waveplot(sound_original[0], sr=44100, color='b', ax=ax[0])
    ax[0].set(title='Original', xlabel=None)
    ax[0].label_outer()
    librosa.display.waveplot(test_reconstruit, sr=44100, color='r', ax=ax[1])
    ax[1].set(title='reconstruction', xlabel=None)
    ax[1].label_outer()
    plt.show()
    """

