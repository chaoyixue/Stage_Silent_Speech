"""
This file is used to test the function yin and pyin in librosa which is used to calculate the fundamental frequency
"""

import librosa


if __name__ == "__main__":
    test_wav, _ = librosa.load("../../wav_files/chapiter7.wav", sr=44100)
    fundamental_vector = librosa.yin(test_wav, fmin=65, fmax=2093, sr=44100, frame_length=2048,
                                     win_length=735*2, hop_length=735)
    stft_wav = librosa.stft(test_wav, n_fft=735*2, hop_length=735, win_length=735*2)
    print("aaa")
