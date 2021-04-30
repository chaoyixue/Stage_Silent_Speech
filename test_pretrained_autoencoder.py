"""
This file is used to visulize the center layer which contains 30 neurons
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import librosa
import soundfile as sf
import librosa.display
from matplotlib import pyplot as plt


def visualize_couche_centrale():
    # load data
    X = np.load("../autoencoder_data/spectrogrammes_all_chapitre.npy")
    max_value = np.max(X)
    print(max_value)
    # normalisation
    X = X / max_value
    print(X.max())
    print(X.min())

    # split train test data
    x_train = np.transpose(X[:, :84776 - 15951])
    x_test = np.transpose(X[:, -15951:])  # changed to shape (15951, 736) for the training process
    print(x_train.shape)
    print(x_test.shape)
    model = keras.models.load_model("model_autoencoder_trained_0.00000230.h5")
    model.summary()

    # the output is the central layer of the autoencoder
    last_layer = keras.models.Model(inputs=model.input, outputs=model.get_layer('dense_4').output)
    last_layer.summary()
    test_result = last_layer.predict(x_test)
    # changed to shape (736, 15951) for the figure of spectrum and the reconstruction
    test_result = np.transpose(test_result)
    print(test_result.shape)
    x_test = np.transpose(x_test)

    plt.imshow(test_result[:, :1000], cmap='hot')
    plt.xlabel("axis time")
    plt.ylabel("neurons in the center layer")
    plt.title("The values in the center layer")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    """
    use the spectrogramme all chapitre corresponding to construct the 30 neurons of the central layer
    """
    """
    spectrogramme_all_corresponding = np.load("../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    max_value = np.max(spectrogramme_all_corresponding)
    print(max_value)
    # normalisation
    spectrogramme_all_corresponding = spectrogramme_all_corresponding / max_value
    print(spectrogramme_all_corresponding.max())
    print(spectrogramme_all_corresponding.min())

    training_spectrogrammes = np.transpose(spectrogramme_all_corresponding[:, :84679-15951])
    testing_spectrogrammes = np.transpose(spectrogramme_all_corresponding[:, -15951:])

    # load the pretrained model
    model = keras.models.load_model("model_autoencoder_trained_0.00000230.h5")
    model.summary()
    center_layer = keras.models.Model(inputs=model.input, outputs=model.get_layer('dense_4').output)
    center_layer.summary()
    training_30_neurons = center_layer.predict(training_spectrogrammes)
    testing_30_neurons = center_layer.predict(testing_spectrogrammes)
    np.save("training_labels_30_neurons.npy", training_30_neurons)
    np.save("validation_labels_30_neurons.npy", testing_30_neurons)
    """
    visualize_couche_centrale()