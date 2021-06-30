"""
This file is used to reconstruct the spectrum using the pretrained models energy_lsf_spectrum models
The inputs are results predicted by transfer_autoencoder_models (lsf, f0, u/v flags, energy)

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


def do_the_prediction_of_image2grandeurs(image2lsf_model, image2f0_model, image2uv_model, image2energy_model):
    # load images inputs
    X_lip = np.load("../../data_npy_one_image/lips_all_chapiters.npy")
    X_tongue = np.load("../../data_npy_one_image/tongues_all_chapiters.npy")

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0

    # split the testing part(ch7)
    nb_image_chapiter7 = 15951
    X_lip_test = X_lip[-nb_image_chapiter7:, :]
    X_tongue_test = X_tongue[-nb_image_chapiter7:, :]

    # do the prediction
    lsf_ch7_predicted = image2lsf_model.predict([X_lip_test, X_tongue_test])
    f0_ch7_predicted = image2f0_model.predict([X_lip_test, X_tongue_test])
    uv_ch7_predicted = image2uv_model.predict([X_lip_test, X_tongue_test])
    energy_ch7_predicted = image2energy_model.predict([X_lip_test, X_tongue_test])

    return lsf_ch7_predicted, f0_ch7_predicted, uv_ch7_predicted, energy_ch7_predicted


def reconstruction_spectrum_by_four_inputs_predicted():
    tr_ae_lsf_model = keras.models.load_model("../../results/week_0622/"
                                              "transfer_autoencoder_lsf_model1-12-0.00502079.h5")
    tr_ae_f0_model = keras.models.load_model("../../results/week_0622/transfer_autoencoder_f0_model1-10-0.03161320.h5")
    tr_ae_uv_model = keras.models.load_model("../../results/week_0622/transfer_autoencoder_uv_model1-09-0.865.h5")
    tr_ae_energy_model = keras.models.load_model("../../results/week_0622/"
                                                 "transfer_autoencoder_energy_model6_dr03-06-0.01141086.h5")

    # predict the four variables using the pretrained models
    lsf_predicted, f0_prediected, uv_predicted, energy_predicted = \
        do_the_prediction_of_image2grandeurs(tr_ae_lsf_model, tr_ae_f0_model, tr_ae_uv_model, tr_ae_energy_model)

    print("aaa")
    print(lsf_predicted.shape)
    print(f0_prediected.shape)
    print(uv_predicted.shape)
    print(energy_predicted.shape)

    X_f0 = np.load("../../LSF_data/f0_all_chapiter.npy")
    energy = np.load("../../LSF_data/energy_all_chapiters.npy")
    spectrum = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    max_spectrum = np.max(spectrum)
    spectrum = spectrum / max_spectrum
    spectrum = np.matrix.transpose(spectrum)
    y_test = spectrum[-15951:]

    # calculate the maximum value of the original data to be used during denormalisation
    max_f0 = np.max(X_f0)
    max_energy = np.max(energy)

    # denormalisation
    f0_prediected = f0_prediected * max_f0
    energy_predicted = energy_predicted * max_energy

    # load the energy_lsf_spectrum model used
    mymodel = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/week_0607/"
                                      "energy_lsf_spectrum_model2-667-0.00000845.h5")

    test_result = mymodel.predict([lsf_predicted, f0_prediected, uv_predicted, energy_predicted])
    result = np.matrix.transpose(test_result)
    mse = tf.keras.losses.MeanSquaredError()
    error = mse(y_test, test_result).numpy()
    print("mean squared error between the spectrum predicted and the original spectrum : %8f" % error)
    result = result * max_spectrum

    # reconstruct the wave file
    test_reconstruit = librosa.griffinlim(result, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_total_model_lsf.wav", test_reconstruit, 44100)

    # load the wave file produced by griffin-lim
    wav_produced, _ = librosa.load("ch7_reconstructed_total_model_lsf.wav", sr=44100)
    spectrogram_produced_griffin = np.abs(librosa.stft(wav_produced, n_fft=735 * 2, hop_length=735, win_length=735 * 2))

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


def reconstruction_spectrum_original_lsf():
    """
    This function is used to test if we use the original lsf values and other variables predicted by models could
    improve the result of the prediction.
    If it's true, that means the problem is from the wrong prediction of lsf values.
    This test is used to localize the problem.
    :return:
    """
    tr_ae_lsf_model = keras.models.load_model("../../results/week_0622/"
                                              "transfer_autoencoder_lsf_model1-12-0.00502079.h5")
    tr_ae_f0_model = keras.models.load_model("../../results/week_0622/transfer_autoencoder_f0_model1-10-0.03161320.h5")
    tr_ae_uv_model = keras.models.load_model("../../results/week_0622/transfer_autoencoder_uv_model1-09-0.865.h5")
    tr_ae_energy_model = keras.models.load_model("../../results/week_0622/"
                                                 "transfer_autoencoder_energy_model6_dr03-06-0.01141086.h5")

    # predict the four variables using the pretrained models
    _, f0_prediected, uv_predicted, energy_predicted = \
        do_the_prediction_of_image2grandeurs(tr_ae_lsf_model, tr_ae_f0_model, tr_ae_uv_model, tr_ae_energy_model)
    # the lsf predicted in this case is replaced by the original lsf values
    lsf_original = np.load("../../LSF_data/lsp_all_chapiter.npy")
    lsf_test = lsf_original[-15951:, :]
    print("aaa")
    print(lsf_original.shape)
    print(f0_prediected.shape)
    print(uv_predicted.shape)
    print(energy_predicted.shape)

    X_f0 = np.load("../../LSF_data/f0_all_chapiter.npy")
    energy = np.load("../../LSF_data/energy_all_chapiters.npy")
    spectrum = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    max_spectrum = np.max(spectrum)
    spectrum = spectrum / max_spectrum
    spectrum = np.matrix.transpose(spectrum)
    y_test = spectrum[-15951:]

    # calculate the maximum value of the original data to be used during denormalisation
    max_f0 = np.max(X_f0)
    max_energy = np.max(energy)

    # denormalisation
    f0_prediected = f0_prediected * max_f0
    energy_predicted = energy_predicted * max_energy

    # load the energy_lsf_spectrum model used
    mymodel = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/week_0607/"
                                      "energy_lsf_spectrum_model2-667-0.00000845.h5")

    test_result = mymodel.predict([lsf_test, f0_prediected, uv_predicted, energy_predicted])
    result = np.matrix.transpose(test_result)
    mse = tf.keras.losses.MeanSquaredError()
    error = mse(y_test, test_result).numpy()
    print("mean squared error between the spectrum predicted and the original spectrum : %8f" % error)
    result = result * max_spectrum

    # reconstruct the wave file
    test_reconstruit = librosa.griffinlim(result, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_total_model_lsf.wav", test_reconstruit, 44100)

    # load the wave file produced by griffin-lim
    wav_produced, _ = librosa.load("ch7_reconstructed_total_model_lsf.wav", sr=44100)
    spectrogram_produced_griffin = np.abs(librosa.stft(wav_produced, n_fft=735 * 2, hop_length=735, win_length=735 * 2))

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


def reconstruction_spectrum_original_lsf_f0():
    tr_ae_lsf_model = keras.models.load_model("../../results/week_0622/"
                                              "transfer_autoencoder_lsf_model1-12-0.00502079.h5")
    tr_ae_f0_model = keras.models.load_model("../../results/week_0622/transfer_autoencoder_f0_model1-10-0.03161320.h5")
    tr_ae_uv_model = keras.models.load_model("../../results/week_0622/transfer_autoencoder_uv_model1-09-0.865.h5")
    tr_ae_energy_model = keras.models.load_model("../../results/week_0622/"
                                                 "transfer_autoencoder_energy_model6_dr03-06-0.01141086.h5")

    # predict the four variables using the pretrained models
    _, _, uv_predicted, energy_predicted = \
        do_the_prediction_of_image2grandeurs(tr_ae_lsf_model, tr_ae_f0_model, tr_ae_uv_model, tr_ae_energy_model)
    # the lsf predicted in this case is replaced by the original lsf values
    lsf_original = np.load("../../LSF_data/lsp_all_chapiter.npy")
    lsf_test = lsf_original[-15951:, :]
    f0_original = np.load("../../LSF_data/f0_all_chapiter.npy")
    f0_test = f0_original[-15951:]
    print("aaa")
    print(lsf_original.shape)
    print(f0_test.shape)
    print(uv_predicted.shape)
    print(energy_predicted.shape)

    X_f0 = np.load("../../LSF_data/f0_all_chapiter.npy")
    energy = np.load("../../LSF_data/energy_all_chapiters.npy")
    spectrum = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    max_spectrum = np.max(spectrum)
    spectrum = spectrum / max_spectrum
    spectrum = np.matrix.transpose(spectrum)
    y_test = spectrum[-15951:]

    # calculate the maximum value of the original data to be used during denormalisation
    max_f0 = np.max(X_f0)
    max_energy = np.max(energy)

    # denormalisation
    energy_predicted = energy_predicted * max_energy

    # load the energy_lsf_spectrum model used
    mymodel = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/week_0607/"
                                      "energy_lsf_spectrum_model2-667-0.00000845.h5")

    test_result = mymodel.predict([lsf_test, f0_test, uv_predicted, energy_predicted])
    result = np.matrix.transpose(test_result)
    mse = tf.keras.losses.MeanSquaredError()
    error = mse(y_test, test_result).numpy()
    print("mean squared error between the spectrum predicted and the original spectrum : %8f" % error)
    result = result * max_spectrum

    # reconstruct the wave file
    test_reconstruit = librosa.griffinlim(result, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_original_lsf_f0.wav", test_reconstruit, 44100)

    # load the wave file produced by griffin-lim
    wav_produced, _ = librosa.load("ch7_reconstructed_original_lsf_f0.wav", sr=44100)
    spectrogram_produced_griffin = np.abs(librosa.stft(wav_produced, n_fft=735 * 2, hop_length=735, win_length=735 * 2))

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


if __name__ == "__main__":
    reconstruction_spectrum_original_lsf_f0()

