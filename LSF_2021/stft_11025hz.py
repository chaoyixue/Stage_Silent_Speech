"""
This file is used to convert wav files to spectrograms then concatenate and save them
into a npy file
The wav file will first be downsampled to 11025hz
"""

import numpy as np
import librosa


def downsample_convert_wav_to_spectrum(file_path, sample_rate=44100, target_rate=11025,
                                       nfft=735*2, window_length=735*2, hop_length=735):
    y, sr = librosa.load(file_path, sr=sample_rate)  # load the wav file with the sample rate chosen
    print(y.shape)
    # downsample the waveform to 11025hz
    y_11025 = librosa.resample(y, sr, target_sr=target_rate)
    # calculate the module spectrum of the waveform downsampled
    spectrogram_y = np.abs(librosa.stft(y_11025, n_fft=nfft, hop_length=hop_length, win_length=window_length))
    return spectrogram_y


if __name__ == "__main__":
    filepath_ch1 = "../../wav_files/chapiter1.wav"
    spectrum_ch1 = downsample_convert_wav_to_spectrum(filepath_ch1)
    print("aaa")
