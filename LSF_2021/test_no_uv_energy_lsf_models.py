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
    X_f0 = np.load("../../LSF_data/f0_all_chapiter_without_threshold.npy")
    spectrum = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    energy = np.load("../../LSF_data/energy_all_chapiters.npy")
    # reshape the energy matrix to (84679,1)
    energy = energy.reshape((energy.shape[1], 1))
    print(energy.shape)

    # normalisation
    max_spectrum = np.max(spectrum)
    spectrum = spectrum / max_spectrum
    spectrum = np.matrix.transpose(spectrum)

    # split train test data
    lsf_train = X_lsf[:-15951]
    lsf_test = X_lsf[-15951:]
    f0_train = X_f0[:-15951]
    f0_test = X_f0[-15951:]
    energy_train = energy[:-15951]
    energy_test = energy[-15951:]

    y_train = spectrum[:-15951]
    y_test = spectrum[-15951:]

    mymodel = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/"
                                      "results/week_0614/no_uv_energy_lsf_model2-1908-0.00000847.h5")
    test_result = mymodel.predict([lsf_test, f0_test, energy_test])
    result = np.matrix.transpose(test_result)
    result = result * max_spectrum

    # reconstruct the wave file

    test_reconstruit = librosa.griffinlim(result, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_no_uv_lsf_model2.wav", test_reconstruit, 44100)

    # load the wave file produced by griffin-lim
    wav_produced, _ = librosa.load("ch7_reconstructed_no_uv_lsf_model2.wav", sr=44100)
    spectrogram_produced_griffin = np.abs(librosa.stft(wav_produced, n_fft=735*2, hop_length=735, win_length=735*2))

    fig, ax = plt.subplots(nrows=3)
    img = librosa.display.specshow(librosa.amplitude_to_db(np.matrix.transpose(y_test),
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='linear', x_axis='time', ax=ax[0])
    ax[0].set_title('original spectrum')
    librosa.display.specshow(librosa.amplitude_to_db(result, ref=np.max), sr=44100, hop_length=735,
                             y_axis='linear', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram_produced_griffin, ref=np.max),
                             sr=44100, hop_length=735, y_axis='linear', x_axis='time', ax=ax[2])
    ax[2].set_title('spectrum reproduced griffinlim')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()




