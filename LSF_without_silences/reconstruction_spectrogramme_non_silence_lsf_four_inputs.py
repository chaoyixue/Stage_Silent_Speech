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
from LSF_2021.test_spectrum_reconstruction_by_four_inputs_predicted import do_the_prediction_of_image2grandeurs


def reconstruction_non_silence_lsf_and_other_grandeurs(index_speech_ch7, index_silence_ch7):
    tr_ae_lsf_model = keras.models.load_model("../../results/week_0705/image2lsf_model7_dr03-36-0.00491458.h5")
    tr_ae_f0_model = keras.models.load_model("../../results/week_0622/transfer_autoencoder_f0_model1-10-0.03161320.h5")
    tr_ae_uv_model = keras.models.load_model("../../results/week_0622/transfer_autoencoder_uv_model1-09-0.865.h5")
    tr_ae_energy_model = keras.models.load_model("../../results/week_0622/"
                                                 "transfer_autoencoder_energy_model6_dr03-06-0.01141086.h5")

    # predict the four variables using the pretrained models
    # the silence parts of lsf_predicted will then be used as the silence parts
    lsf_predicted_used_for_silences, f0_prediected, uv_predicted, energy_predicted = \
        do_the_prediction_of_image2grandeurs(tr_ae_lsf_model, tr_ae_f0_model, tr_ae_uv_model, tr_ae_energy_model)

    print(f0_prediected.shape)
    print(uv_predicted.shape)
    print(energy_predicted.shape)

    # use a threshold of 0.5 to reset the uv predicted to 0 or 1
    uv_predicted[uv_predicted >= 0.5] = 1
    uv_predicted[uv_predicted < 0.5] = 0

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

    lsf_combined = np.zeros((15951, 13))
    predicted_lsf_silence_part_ch7 = lsf_predicted_used_for_silences[index_silence_ch7]  # (5837,13)

    # predict the speech part use the model nonsilence and the cleaned data
    X_lip_cleaned = np.load("../../non_silence_LSF_data/lip_nonsilence_all_chapiter.npy")
    X_tongue_cleaned = np.load("../../non_silence_LSF_data/tongue_nonsilence_all_chapiter.npy")
    nonsilence_lsf_model = keras.models.load_model("../../results/week_0726/"
                                                   "nonsilence_image2lsf_model7_dr30-56-0.00537513.h5")
    # normalisation
    X_lip_cleaned = X_lip_cleaned/255.0
    X_tongue_cleaned = X_tongue_cleaned/255.0
    X_lip_cleaned_ch7 = X_lip_cleaned[-10114:]
    X_tongue_cleaned_ch7 = X_tongue_cleaned[-10114:]

    predicted_lsf_sppech_part_ch7 = nonsilence_lsf_model.predict([X_lip_cleaned_ch7, X_tongue_cleaned_ch7])  # (10114,13)
    lsf_combined[index_speech_ch7] = predicted_lsf_sppech_part_ch7
    lsf_combined[index_silence_ch7] = predicted_lsf_silence_part_ch7
    # the last coefficient set à 0
    lsf_combined[:, 12] = 0

    # load the energy_lsf_spectrum model used
    mymodel = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/week_0607/"
                                      "energy_lsf_spectrum_model2-667-0.00000845.h5")

    test_result = mymodel.predict([lsf_combined, f0_prediected, uv_predicted, energy_predicted])
    result = np.matrix.transpose(test_result)
    mse = tf.keras.losses.MeanSquaredError()
    error = mse(y_test, test_result).numpy()
    print("mean squared error between the spectrum predicted and the original spectrum : %8f" % error)
    result = result * max_spectrum

    # reconstruct the wave file
    test_reconstruit = librosa.griffinlim(result, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reconstructed_total_model_nonsilence_lsf_0729.wav", test_reconstruit, 44100)

    # load the wave file produced by griffin-lim
    wav_produced, _ = librosa.load("ch7_reconstructed_total_model_nonsilence_lsf_0729.wav", sr=44100)
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
    ch7index_speech = np.load("index_non_silence_chapiter_7.npy")
    ch7index_silence = np.load("index_silence_chapiter_7.npy")
    reconstruction_non_silence_lsf_and_other_grandeurs(ch7index_speech, ch7index_silence)