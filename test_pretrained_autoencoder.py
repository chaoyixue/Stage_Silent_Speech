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

if __name__ == "__main__":
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

    last_layer = keras.models.Model(inputs=model.input, outputs=model.get_layer('dense_4').output)
    last_layer.summary()
    test_result = last_layer.predict(x_test)
    # changed to shape (736, 15951) for the figure of spectrum and the reconstruction
    test_result = np.transpose(test_result)
    print(test_result.shape)
    x_test = np.transpose(x_test)

    plt.imshow(test_result[:, :100], cmap='hot')
    plt.xlabel("axis time")
    plt.ylabel("neurons in the center layer")
    plt.title("The values in the center layer")
    plt.colorbar()
    plt.show()

    """
    fig, ax = plt.subplots(nrows=2)

    img = librosa.display.specshow(librosa.amplitude_to_db(x_test,
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('Power spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(test_result, ref=np.max), sr=44100, hop_length=735,
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    """