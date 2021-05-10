"""
This file is used to generate two parts of labels for the real and imaginary parts of the spectrogram
"""

import numpy as np

if __name__ == "__main__":
    spectrogram_labels = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    print(spectrogram_labels.shape)
    real_spectrogram = spectrogram_labels.real
    imaginary_spectrogram = spectrogram_labels.imag
    print(real_spectrogram.shape)
    print(imaginary_spectrogram.shape)
    np.save("real_spectrograme.npy", real_spectrogram)
    np.save("imag_spectrograme.npy", imaginary_spectrogram)