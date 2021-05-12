"""
This file is used to generate two parts of labels for the real and imaginary parts of the spectrogram
"""

import numpy as np
import librosa


def convert_wav_to_two_part_spectrum(file_path, sample_rate=44100, nfft=735*2, window_length=735*2, hop_length=735):
    """
    :param file_path:
    :param sample_rate:
    :param nfft:
    :param window_length:
    :param hop_length:
    :return:
    """
    y, sr = librosa.load(file_path, sr=sample_rate)  # load the wav file with the sample rate chosen
    spectrogram_y = librosa.stft(y, n_fft=nfft, hop_length=hop_length, win_length=window_length)
    real_part = spectrogram_y.real
    imag_part = spectrogram_y.imag
    # shape : (736, N)
    return real_part, imag_part


if __name__ == "__main__":
    filepath_ch1 = "../../wav_files/chapiter1.wav"
    real_ch1, imag_ch1 = convert_wav_to_two_part_spectrum(filepath_ch1)
    real_ch1 = real_ch1[:, :10054]
    imag_ch1 = imag_ch1[:, :10054]

    filepath_ch2 = "../../wav_files/chapiter2.wav"
    real_ch2, imag_ch2 = convert_wav_to_two_part_spectrum(filepath_ch2)
    real_ch2 = real_ch2[:, :14441]
    imag_ch2 = imag_ch2[:, :14441]

    filepath_ch3 = "../../wav_files/chapiter3.wav"
    real_ch3, imag_ch3 = convert_wav_to_two_part_spectrum(filepath_ch3)
    real_ch3 = real_ch3[:, :8885]
    imag_ch3 = imag_ch3[:, :8885]

    filepath_ch4 = "../../wav_files/chapiter4.wav"
    real_ch4, imag_ch4 = convert_wav_to_two_part_spectrum(filepath_ch4)
    real_ch4 = real_ch4[:, :15621]
    imag_ch4 = imag_ch4[:, :15621]

    filepath_ch5 = "../../wav_files/chapiter5.wav"
    real_ch5, imag_ch5 = convert_wav_to_two_part_spectrum(filepath_ch5)
    real_ch5 = real_ch5[:, :14553]
    imag_ch5 = imag_ch5[:, :14553]

    filepath_ch6 = "../../wav_files/chapiter6.wav"
    real_ch6, imag_ch6 = convert_wav_to_two_part_spectrum(filepath_ch6)
    real_ch6 = real_ch6[:, :5174]
    imag_ch6 = imag_ch6[:, :5174]

    filepath_ch7 = "../../wav_files/chapiter7.wav"
    real_ch7, imag_ch7 = convert_wav_to_two_part_spectrum(filepath_ch7)
    real_ch7 = real_ch7[:, :15951]
    imag_ch7 = imag_ch7[:, :15951]

    real_parts = np.concatenate((real_ch1, real_ch2, real_ch3, real_ch4, real_ch5, real_ch6, real_ch7), axis=1)
    imag_parts = np.concatenate((imag_ch1, imag_ch2, imag_ch3, imag_ch4, imag_ch5, imag_ch6, imag_ch7), axis=1)
    # axis 0 is the frequency axis corresponding to 736 bins of frequency. axis 1 is the time
    # axis corresponding to seconds
    np.save("real_parts_spectrograms.npy", real_parts)
    np.save("imag_parts_spectrograms.npy", imag_parts)
