"""
This file is used to test the function yin and pyin in librosa which is used to calculate the fundamental frequency
"""

import librosa
from matplotlib import pyplot as plt
import numpy as np


def calculate_f0(file_path, fmin=65, fmax=2093, sample_rate=44100, frame_length=2048, win_length=735*2, hop_length=735):
    """
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
                            win_length=win_length, hop_length=hop_length)
    return result_f0


if __name__ == "__main__":
    f0_chapiter1 = calculate_f0("../../wav_files/chapiter1.wav")
    f0_chapiter2 = calculate_f0("../../wav_files/chapiter2.wav")
    f0_chapiter3 = calculate_f0("../../wav_files/chapiter3.wav")
    f0_chapiter4 = calculate_f0("../../wav_files/chapiter4.wav")
    f0_chapiter5 = calculate_f0("../../wav_files/chapiter5.wav")
    f0_chapiter6 = calculate_f0("../../wav_files/chapiter6.wav")
    f0_chapiter7 = calculate_f0("../../wav_files/chapiter7.wav")
    x1 = np.arange(0, len(f0_chapiter1)) * 0.016  # 16ms between two values
    x2 = np.arange(0, len(f0_chapiter2))
    x3 = np.arange(0, len(f0_chapiter3))
    x4 = np.arange(0, len(f0_chapiter4))
    x5 = np.arange(0, len(f0_chapiter5))
    x6 = np.arange(0, len(f0_chapiter6))
    x7 = np.arange(0, len(f0_chapiter7))
    plt.figure()
    plt.plot(x1, f0_chapiter1, 'r-', label="chapiter 1")
    plt.legend()
    plt.title("fundamental frequency")
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    plt.show()
    print("aaa")