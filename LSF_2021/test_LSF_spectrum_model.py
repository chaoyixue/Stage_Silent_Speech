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
    X_lsf = np.load("../../LSF_data/lsp_all_chapiter.npy")
    X_f0 = np.load("../../LSF_data/f0_all_chapiter.npy")
    X_uv = np.load("../../LSF_data/uv_all_chapiter.npy")
    spectrum = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")

    # X_lsf = X_lsf.reshape(len(X_lsf), X_lsf.shape[1], 1)
    # X_f0 = X_f0.reshape(len(X_f0), 1, 1)
    # X_uv = X_uv.reshape(len(X_uv), 1, 1)

    # normalisation
    max_spectrum = np.max(spectrum)
    spectrum = spectrum / max_spectrum
    spectrum = np.matrix.transpose(spectrum)

    # split train test data
    lsf_train = X_lsf[:-15951]
    lsf_test = X_lsf[-15951:]
    f0_train = X_f0[:-15951]
    f0_test = X_f0[-15951:]
    uv_train = X_uv[:-15951]
    uv_test = X_uv[-15951:]

    y_train = spectrum[:-15951]
    y_test = spectrum[-15951:]

    mymodel = keras.models.load_model("../../results/week_0602/lsf_spectrum_model8-565-0.00001415.h5")
    test_result = mymodel.predict([lsf_test, f0_test, uv_test])
    result = np.matrix.transpose(test_result)
    result = result * max_spectrum

    # reconstruct the wav file
    test_reconstruit = librosa.griffinlim(result, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_lsf_f0_uv_model8.wav", test_reconstruit, 44100)

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


