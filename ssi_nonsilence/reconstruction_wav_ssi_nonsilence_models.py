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
    # initial spectrum 15951,736
    spectrum_reconstructed = np.zeros((15951, 736))

    # load data
    X_lip_nonsilence_all_chapiter = np.load("../../non_silence_LSF_data/lip_nonsilence_all_chapiter.npy")
    X_tongue_nonsilence_all_chapiter = np.load("../../non_silence_LSF_data/tongue_nonsilence_all_chapiter.npy")
    Y_nonsilence = np.load("../../non_silence_LSF_data/spectrum_nonsilence_all_chapiter.npy")

    # normalisation
    X_lip_nonsilence_all_chapiter = X_lip_nonsilence_all_chapiter / 255.0
    X_tongue_nonsilence_all_chapiter = X_tongue_nonsilence_all_chapiter / 255.0
    max_spectrum = np.max(Y_nonsilence)
    Y_nonsilence = Y_nonsilence / max_spectrum

    # split train set data
    nb_frames_nonsilence_ch7 = 10114
    lips_x_train = X_lip_nonsilence_all_chapiter[:-nb_frames_nonsilence_ch7]
    lips_x_test = X_lip_nonsilence_all_chapiter[-nb_frames_nonsilence_ch7:]
    tongues_x_train = X_tongue_nonsilence_all_chapiter[:-nb_frames_nonsilence_ch7]
    tongues_x_test = X_tongue_nonsilence_all_chapiter[-nb_frames_nonsilence_ch7:]
    y_train = Y_nonsilence[:-nb_frames_nonsilence_ch7]
    y_test = Y_nonsilence[-nb_frames_nonsilence_ch7:]

    ssi_nonsilence_model = keras.models.load_model("../../results/week_0805/ssi_nonsilence_model1-15-0.00007264.h5")
    spectrum_nonsilence_ch7 = ssi_nonsilence_model.predict([lips_x_test, tongues_x_test])  # (10114,736)
    # index_nonsilence and index_silences to reconstruct the whole spectrum for ch7
    ch7_index_speech = np.load("../../non_silence_LSF_data/index_non_silence_chapiter_7.npy")
    ch7_index_silence = np.load("../../non_silence_LSF_data/index_silence_chapiter_7.npy")
    spectrum_reconstructed[ch7_index_speech] = spectrum_nonsilence_ch7
    print("spectrum_reconstructed_shape : ", spectrum_nonsilence_ch7.shape)

    spectrum_griffin_lim = np.matrix.transpose(spectrum_reconstructed)  # (736, 15951)

    # compare the spectrum predicted and the original spectrum
    spectrum_original = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy") # (736, 84679)
    spectrum_original_ch7 = spectrum_original[:, -15951:] / max_spectrum
    mse = tf.keras.losses.MeanSquaredError()
    error = mse(spectrum_original_ch7, spectrum_griffin_lim).numpy()
    print("mean squared error between the spectrum predicted and the original spectrum : %8f" % error)
    # denormalisation
    spectrum_griffin_lim = spectrum_griffin_lim * max_spectrum

    # griffin lim
    test_reconstruit = librosa.griffinlim(spectrum_griffin_lim, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_by_ssi_nonsilence_model1_0805.wav", test_reconstruit, 44100)

    # load the wave file produced by griffin-lim
    wav_produced, _ = librosa.load("ch7_reconstructed_by_ssi_nonsilence_model1_0805.wav", sr=44100)
    spectrogram_produced_griffin = np.abs(librosa.stft(wav_produced, n_fft=735 * 2, hop_length=735, win_length=735 * 2))

    fig, ax = plt.subplots(nrows=3)
    img = librosa.display.specshow(librosa.amplitude_to_db(spectrum_original_ch7,
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='linear', x_axis='time', ax=ax[0])
    ax[0].set_title('original spectrum')
    librosa.display.specshow(librosa.amplitude_to_db(spectrum_griffin_lim, ref=np.max), sr=44100, hop_length=735,
                             y_axis='linear', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram_produced_griffin, ref=np.max),
                             sr=44100, hop_length=735, y_axis='linear', x_axis='time', ax=ax[2])
    ax[2].set_title('spectrum reproduced griffinlim')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


