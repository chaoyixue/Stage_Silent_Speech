import numpy as np

if __name__ == "__main__":
    training_labels = np.load("../labels_generated_autoencoder_30values/training_labels_30_neurons.npy")
    testing_labels = np.load("../labels_generated_autoencoder_30values/validation_labels_30_neurons.npy")
    max_30 = np.max(training_labels)
    min_30 = np.min(training_labels)
    training_labels /= max_30
    testing_labels /= max_30
    spectrum_original = np.load("../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    max_spectrum = np.max(spectrum_original)
    min_spectrum = np.min(spectrum_original)
    spectrum_original /= max_spectrum
    print("aaa")