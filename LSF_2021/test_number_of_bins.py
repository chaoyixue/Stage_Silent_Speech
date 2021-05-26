import librosa
import numpy as np
from wavefile_preprocessing import convert_wav_to_spectrum


if __name__ == "__main__":
    test_wav = librosa.load("../../wav_files/chapiter1.wav", sr=44100)
