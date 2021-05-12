import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten
from tensorflow.keras.models import Model
import librosa
import soundfile as sf
import librosa.display

if __name__ == "__main__":
    # load data
    X_lips = np.load("../../data_npy_one_image/lips_all_chapiters.npy")
    X_tongues = np.load("../../data_npy_one_image/tongues_all_chapiters.npy")
    # load the module matrix of the spectrogram
    Y_module = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    # load the real part of the spectrogram
    y_real = np.load("../../labels_two_parts/real_parts_spectrograms.npy")
    # load the imaginary part of the spectrogram
    y_imag = np.load("../../labels_two_parts/imag_parts_spectrograms.npy")
    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_spectrum = np.max(Y_module)
    y_real /= max_spectrum
    y_imag /= max_spectrum

    # split train set data
    lips_x_train = X_lips[:-15951]
    lips_x_test = X_lips[-15951:]
    tongues_x_train = X_tongues[:-15951]
    tongues_x_test = X_tongues[-15951:]
    # train test split for the real parts
    y_real = np.transpose(y_real)
    y_train_real = y_real[:-15951, :]
    y_test_real = y_real[-15951:, :]
    # train test split for the imaginary parts
    y_imag = np.transpose(y_imag)
    y_train_imag = y_imag[:-15951, :]
    y_test_imag = y_imag[-15951:, :]

    model = keras.models.load_model("../../results/two_parts_ssi_model1-45-0.00007206.h5")
    model.summary()
    [test_real, test_imag] = model.predict([lips_x_test, tongues_x_test])
    reconstructed_spectrogram = (test_real + 1j * test_imag) * max_spectrum
    reconstructed_spectrogram = np.transpose(reconstructed_spectrogram)
    # module of the reconstructed spectrogram
    module_reconstructed = np.abs(reconstructed_spectrogram)

    fig, ax = plt.subplots(nrows=2)

    img = librosa.display.specshow(librosa.amplitude_to_db(Y_module[:, -15951:],
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('Power spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(module_reconstructed, ref=np.max), sr=44100, hop_length=735,
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    reconstructed_wav = librosa.istft(reconstructed_spectrogram, hop_length=735, win_length=735*2)
    sf.write("reconstruction_by_istft_two_parts_ssi_model1-45-0.00007206.wav", reconstructed_wav, samplerate=44100)
    print("aaa")