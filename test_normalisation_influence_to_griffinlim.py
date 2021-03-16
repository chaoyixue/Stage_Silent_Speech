"""
This file is used to test the influence of normalisation to the griffinlim algorithm
we found out that even normalized data can be reconstructed using griffin lim algorithm
the only difference is that when using normalized spectrum data as input,the output wavfile will have a very low
volumn.
"""

import numpy as np
import librosa
from matplotlib import pyplot as plt
import librosa.display
import soundfile as sf

if __name__ == "__main__":
    # the original wav file
    original_wav, sr_original = librosa.load("../data/20200617_153719_RecFile_1_bruce_ch7"
                                             "/RecFile_1_20200617_153719_Sound_Capture_DShow_5_monoOutput1.wav",
                                             sr=44100)
    # the spectrum calculated with the original file
    spectrum_original_wav = np.abs(librosa.stft(original_wav, n_fft=1470, hop_length=735))

    # the spectrum saved in the npy file
    spectrum_npy_all_chapitre = np.load("spectrogrammes_all_chapitre.npy")
    spectrum_test = spectrum_npy_all_chapitre[:, -15951:]
    y_reconstruit = librosa.griffinlim(spectrum_test, hop_length=735, win_length=735*2)
    sf.write("reconstruction_npy_ch7.wav", y_reconstruit, samplerate=44100)

    # the spectrum normalized in the npy file
    spectrum_normalized = spectrum_test/np.max(spectrum_test)
    y_normalized_reconstruit = librosa.griffinlim(spectrum_normalized, hop_length=735, win_length=735*2)
    sf.write("reconstruction_ch7_normalized.wav", y_normalized_reconstruit, samplerate=44100)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img = librosa.display.specshow(librosa.amplitude_to_db(spectrum_original_wav,
                                                           ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('Power spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(spectrum_test, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrogram npy')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


