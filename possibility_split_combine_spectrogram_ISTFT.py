"""
This file is used to test if we can split the spectrogram into real and imaginary parts and combine them to reconstruct
the original wave file using ISTFT
"""

import numpy as np
import librosa
import soundfile as sf


if __name__ == "__main__":
    # load the original wav
    test_wave, _ = librosa.load("../RecFile_1_20200617_153719_Sound_Capture_DShow_5_monoOutput1.wav", sr=44100)

    # calculate the complex spectrogram stft
    spectrogram_test_wav = librosa.stft(test_wave, n_fft=735*2, win_length=735*2, hop_length=735)

    # calculate the real part of the spectrogram
    real_spectrogram = spectrogram_test_wav.real
    # calculate the imaginary part of the spectrogram
    imaginary_spectrogram = spectrogram_test_wav.imag

    # combine these two parts
    reconstruction_spectrogram = real_spectrogram + 1j * imaginary_spectrogram
    print(np.array_equal(spectrogram_test_wav, reconstruction_spectrogram))

    # reform the wavfile using ISTFT
    reconstruction_wav = librosa.istft(reconstruction_spectrogram, hop_length=735, win_length=735*2)
    sf.write("reconstruction_by_istft.wav", reconstruction_wav, samplerate=44100)
