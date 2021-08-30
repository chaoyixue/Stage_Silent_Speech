"""
This file is used to calculate the root-mean-square (RMS) energy for each frame and save them into a npy file
"""

import librosa
import numpy as np


def convert_wav_to_energy(file_path, sample_rate=44100, window_length=735*2, hop_length=735):
    """

    :param file_path:
    :param sample_rate:
    :param window_length:
    :param hop_length:
    :return:
    """
    y, sr = librosa.load(file_path, sr=sample_rate)  # load the wav file with the sample rate chosen
    energy_matrix = librosa.feature.rms(y, frame_length=window_length, hop_length=hop_length)
    return energy_matrix


def all_chapiter_wav_to_energy():
    filepath_ch1 = "../../wav_files/chapiter1.wav"
    spect_ch1 = convert_wav_to_energy(filepath_ch1)
    spect_ch1 = spect_ch1[:, :10054]
    filepath_ch2 = "../../wav_files/chapiter2.wav"
    spect_ch2 = convert_wav_to_energy(filepath_ch2)
    spect_ch2 = spect_ch2[:, :14441]
    filepath_ch3 = "../../wav_files/chapiter3.wav"
    spect_ch3 = convert_wav_to_energy(filepath_ch3)
    spect_ch3 = spect_ch3[:, :8885]

    filepath_ch4 = "../../wav_files/chapiter4.wav"
    spect_ch4 = convert_wav_to_energy(filepath_ch4)
    spect_ch4 = spect_ch4[:, :15621]

    filepath_ch5 = "../../wav_files/chapiter5.wav"
    spect_ch5 = convert_wav_to_energy(filepath_ch5)
    spect_ch5 = spect_ch5[:, :14553]

    filepath_ch6 = "../../wav_files/chapiter6.wav"
    spect_ch6 = convert_wav_to_energy(filepath_ch6)
    spect_ch6 = spect_ch6[:, :5174]

    filepath_ch7 = "../../wav_files/chapiter7.wav"
    spect_ch7 = convert_wav_to_energy(filepath_ch7)
    spect_ch7 = spect_ch7[:, :15951]

    result = np.concatenate((spect_ch1, spect_ch2, spect_ch3, spect_ch4, spect_ch5, spect_ch6, spect_ch7), axis=1)
    print(result.shape)
    # axis 0 is the frequency axis corresponding to 736 bins of frequency. axis 1 is the time
    # axis corresponding to seconds
    np.save("matrix_energy_all_chapiters.npy", result)


if __name__ == "__main__":
    # calculation matrice d'énergie à partir des fichiers audio coupé
    filepath_ch1 = "../../wav_files_coupe/ch1_coupe.wav"
    spect_ch1 = convert_wav_to_energy(filepath_ch1)[:, :-1]

    filepath_ch2 = "../../wav_files_coupe/ch2_coupe.wav"
    spect_ch2 = convert_wav_to_energy(filepath_ch2)[:, :-1]

    filepath_ch3 = "../../wav_files_coupe/ch3_coupe.wav"
    spect_ch3 = convert_wav_to_energy(filepath_ch3)[:, :-1]

    filepath_ch4 = "../../wav_files_coupe/ch4_coupe.wav"
    spect_ch4 = convert_wav_to_energy(filepath_ch4)[:, :-1]

    filepath_ch5 = "../../wav_files_coupe/ch5_coupe.wav"
    spect_ch5 = convert_wav_to_energy(filepath_ch5)[:, :-1]

    filepath_ch6 = "../../wav_files_coupe/ch6_coupe.wav"
    spect_ch6 = convert_wav_to_energy(filepath_ch6)[:, :-1]

    filepath_ch7 = "../../wav_files_coupe/ch7_coupe.wav"
    spect_ch7 = convert_wav_to_energy(filepath_ch7)[:, :-1]

    result = np.concatenate((spect_ch1, spect_ch2, spect_ch3, spect_ch4, spect_ch5, spect_ch6, spect_ch7), axis=1)
    result = np.transpose(result)
    print(result.shape)
    # axis 0 is the frequency axis corresponding to 736 bins of frequency. axis 1 is the time
    # axis corresponding to seconds
    np.save("../../LSF_data_coupe/energy_cut_all.npy", result)
