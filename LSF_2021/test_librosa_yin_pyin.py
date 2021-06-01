"""
This file is used to test the function yin and pyin in librosa which is used to calculate the fundamental frequency
"""

import librosa
from matplotlib import pyplot as plt
import numpy as np


def calculate_f0(file_path, fmin=65, fmax=2093, sample_rate=44100, frame_length=2048, win_length=735*2,
                 hop_length=735, threshold=0.1, seuil_voice=300):
    """
    :param seuil_voice:
    :param threshold:
    :param file_path: the path of the wav file .wav
    :param fmin:
    :param fmax:
    :param sample_rate: the sample rate of the wav file
    :param frame_length:
    :param win_length:
    :param hop_length:
    :return: a numpy array which contains the f0 at every moment
    """

    wav_file, _ = librosa.load(file_path, sr=sample_rate)
    result_f0 = librosa.yin(wav_file, fmin=fmin, fmax=fmax, sr=sample_rate, frame_length=frame_length,
                            win_length=win_length, hop_length=hop_length, trough_threshold=threshold)
    # choose a threshold to distinguish voiced and unvoiced part of the waveform
    # for the unvoiced parts, we consider the F0 = 0
    result_f0[result_f0 >= seuil_voice] = 0
    # initialize a vector to save the voiced unvoiced flags
    uv_flags = np.zeros(result_f0.shape)
    uv_flags[result_f0 > 0] = 1
    return result_f0, uv_flags


if __name__ == "__main__":
    f0_chapiter1, uv_ch1 = calculate_f0("../../wav_files/chapiter1.wav")
    f0_chapiter2, uv_ch2 = calculate_f0("../../wav_files/chapiter2.wav")
    f0_chapiter3, uv_ch3 = calculate_f0("../../wav_files/chapiter3.wav")
    f0_chapiter4, uv_ch4 = calculate_f0("../../wav_files/chapiter4.wav")
    f0_chapiter5, uv_ch5 = calculate_f0("../../wav_files/chapiter5.wav")
    f0_chapiter6, uv_ch6 = calculate_f0("../../wav_files/chapiter6.wav")
    f0_chapiter7, uv_ch7 = calculate_f0("../../wav_files/chapiter7.wav")
    # to correspond to the number of images lips and tongues
    f0_chapiter1 = f0_chapiter1[:10054]
    f0_chapiter2 = f0_chapiter2[:14441]
    f0_chapiter3 = f0_chapiter3[:8885]
    f0_chapiter4 = f0_chapiter4[:15621]
    f0_chapiter5 = f0_chapiter5[:14553]
    f0_chapiter6 = f0_chapiter6[:5174]
    f0_chapiter7 = f0_chapiter7[:15951]
    uv_ch1 = uv_ch1[:10054]
    uv_ch2 = uv_ch2[:14441]
    uv_ch3 = uv_ch3[:8885]
    uv_ch4 = uv_ch4[:15621]
    uv_ch5 = uv_ch5[:14553]
    uv_ch6 = uv_ch6[:5174]
    uv_ch7 = uv_ch7[:15951]
    f0_all_chapiters = np.concatenate((f0_chapiter1, f0_chapiter2, f0_chapiter3,
                                      f0_chapiter4, f0_chapiter5, f0_chapiter6, f0_chapiter7), axis=0)
    uv_all_chapiters = np.concatenate((uv_ch1, uv_ch2, uv_ch3, uv_ch4, uv_ch5, uv_ch6, uv_ch7), axis=0)
    print(f0_all_chapiters.shape)
    print(uv_all_chapiters.shape)
    np.save("f0_all_chapiter.npy", f0_all_chapiters)
    np.save("uv_all_chapiter.npy", uv_all_chapiters)
    """
    x1 = np.arange(0, len(f0_chapiter1)) * 0.016  # 16ms between two values
    x2 = np.arange(0, len(f0_chapiter2))
    x3 = np.arange(0, len(f0_chapiter3))
    x4 = np.arange(0, len(f0_chapiter4))
    x5 = np.arange(0, len(f0_chapiter5))
    x6 = np.arange(0, len(f0_chapiter6))
    """
    x7 = np.arange(0, len(f0_chapiter7)) * 0.016

    plt.figure()
    plt.plot(x7, f0_chapiter7, 'r-', label="chapiter 1")
    plt.legend()
    plt.title("fundamental frequency")
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    plt.show()


