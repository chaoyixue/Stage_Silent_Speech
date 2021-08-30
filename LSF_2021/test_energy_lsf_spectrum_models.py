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
    # load data coupe
    X_lsf = np.load("../../LSF_data_coupe/lsp_cut_all.npy")
    X_f0 = np.load("../../LSF_data_coupe/f0_cut_all.npy")
    X_uv = np.load("../../LSF_data_coupe/uv_cut_all.npy")
    spectrum = np.load("../../data_coupe/spectrogrammes_all_chapitres_coupe.npy")
    energy = np.load("../../LSF_data_coupe/energy_cut_all.npy")

    # normalisation
    max_spectrum = np.max(spectrum)
    spectrum = spectrum / max_spectrum

    # split train test data
    lsf_train = X_lsf[:-15951]
    lsf_test = X_lsf[-15951:]
    f0_train = X_f0[:-15951]
    f0_test = X_f0[-15951:]
    uv_train = X_uv[:-15951]
    uv_test = X_uv[-15951:]
    energy_train = energy[:-15951]
    energy_test = energy[-15951:]

    y_train = spectrum[:-15951]
    y_test = spectrum[-15951:]

    mymodel = keras.models.load_model("../../results/week_0823/energy_lsf_spectrum_data_coupe_model2-678-0.00000796.h5")
    test_result = mymodel.predict([lsf_test, f0_test, uv_test, energy_test]) # 15951,736
    result = np.matrix.transpose(test_result) # 736, 15951
    result = result * max_spectrum

    # reconstruct the wave file

    test_reconstruit = librosa.griffinlim(result, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_energy_lsf_spectrum_datacoupe_model1.wav", test_reconstruit, 44100)

    # load the wave file produced by griffin-lim
    wav_produced, _ = librosa.load("ch7_reconstructed_energy_lsf_spectrum_datacoupe_model1.wav", sr=44100)
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




