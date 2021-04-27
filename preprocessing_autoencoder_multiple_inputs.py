"""
This file is used to do preprocessing for the spectrums to be used as inputs and outputs of the autoencoder
"""

import numpy as np


def reshape_to_recurrent_spectrums(npy_array_spectrogramme, longeur_sequence):
    """

    :param npy_array_spectrogramme: spectrogramme array  of shape (bins_frequency, shape_time) for example (736, 84776)
    :param longeur_sequence: the number of vecteurs needed to be rejoint for example 5
    :return: a npy array of which the shape is (nb_groupes_vecteur, length_vector, bin_frequency) for example
                (shape_time - longeur_sequence + 1, longeur_sequence,736)
    """
    npy_array_spectrogramme = np.transpose(npy_array_spectrogramme)
    original_length = np.shape(npy_array_spectrogramme)[0]
    bins_frequency = np.shape(npy_array_spectrogramme)[1]
    result_npy_spectrogramme = np.zeros((original_length - longeur_sequence + 1, longeur_sequence, bins_frequency))
    for i in range(len(result_npy_spectrogramme)):
        result_npy_spectrogramme[i, :, :] = npy_array_spectrogramme[i:i+longeur_sequence, :]
    return result_npy_spectrogramme


if __name__ == "__main__":
    test = np.load("../autoencoder_data/spectrogrammes_all_chapitre.npy")
    result = reshape_to_recurrent_spectrums(test, 5)
    print(test.shape)
    np.save("spectrogramme_recurrent_all_chapitre.npy", result)
