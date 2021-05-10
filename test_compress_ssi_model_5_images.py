from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import librosa
import soundfile as sf
import librosa.display
from matplotlib import pyplot as plt
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten


def prediction_compress_ssi_recurrent_model():
    # load data
    X_lips = np.load("../validation_data/lips_validation_ch7.npy")
    X_tongues = np.load("../validation_data/tongues_validation_ch7.npy")
    validation_30_values = np.load("../labels_generated_autoencoder_30values/validation_labels_30_neurons.npy")

    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_30 = np.max(validation_30_values)
    validation_30_values /= max_30
    # choose the 15947 values in the middle
    validation_30_values = np.transpose(validation_30_values[2:-2])

    model = keras.models.load_model("../results/week_0426/compress_ssi_model1_bs128-24-0.00933496.h5")
    model.summary()
    test_result = model.predict([X_lips, X_tongues])
    result = np.transpose(test_result)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    im = ax[0].imshow(validation_30_values[:, 500:1500], cmap="hot")
    ax[0].set_title("labels_30_values")
    ax[1].imshow(result[:, 500:1500], cmap="hot")
    ax[1].set_title("results_30_values")
    fig.colorbar(im, ax=ax)
    plt.show()


def prediction_autoencoder_input_couche_centre(autoencoder_model, input_npy):
    """
    This function is used to predict the spectrogramme using the decoder part of autoencoder
    The input is 30 values generated and the output is the spectrogrammes

    :return: a numpy array spectrogramme
    """
    autoencoder_model.summary()
    decoder_part_input = Input(shape=(30, ))
    decoder_1 = autoencoder_model.get_layer('dense_5')(decoder_part_input)
    decoder_2 = autoencoder_model.get_layer('dense_6')(decoder_1)
    decoder_3 = autoencoder_model.get_layer('dense_7')(decoder_2)
    decoder_4 = autoencoder_model.get_layer('dense_8')(decoder_3)
    decoder_5 = autoencoder_model.get_layer('dense_9')(decoder_4)
    decoder = Model(decoder_part_input, decoder_5)
    decoder.summary()
    result = decoder.predict(input_npy)
    return result


if __name__ == "__main__":
    # load data
    X_lips = np.load("../validation_data/lips_validation_ch7.npy")
    X_tongues = np.load("../validation_data/tongues_validation_ch7.npy")
    validation_30_values = np.load("../labels_generated_autoencoder_30values/validation_labels_30_neurons.npy")
    # shape (bins_frequency, nb_vecteurs) for example here (736,15951)
    validation_spectrogramme = np.load("../validation_data/spectrum_validation.npy")

    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_30 = np.max(validation_30_values)
    validation_30_values /= max_30
    max_spectrum = np.max(validation_spectrogramme)

    # choose the 15947 values in the middle
    validation_30_values = np.transpose(validation_30_values[2:-2])

    # load the compress_ssi_model line to change for the test process
    model = keras.models.load_model("../results/compress_ssi_model5_linear_bs128-41-0.00916323.h5")
    model.summary()
    # make the prediction to get the 30 neurons of central layer
    test_result = model.predict([X_lips, X_tongues])
    autoencoder = keras.models.load_model("model_autoencoder_trained_0.00000230.h5")
    # regenerate the spectrogramme using the pretrained decoder of the autoencoder and the 30 values
    spectrogramme_generated = prediction_autoencoder_input_couche_centre(autoencoder, test_result)
    spectrogramme_generated *= max_spectrum
    spectrogramme_generated = np.transpose(spectrogramme_generated)

    # compare the spectrogramme with the original spectrogramme
    fig, ax = plt.subplots(nrows=2)
    img = librosa.display.specshow(librosa.amplitude_to_db(validation_spectrogramme,
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('original spectrum')
    librosa.display.specshow(librosa.amplitude_to_db(spectrogramme_generated, ref=np.max), sr=44100, hop_length=735,
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # regenerate the wav file
    test_reconstruit = librosa.griffinlim(spectrogramme_generated, n_iter=128, hop_length=735, win_length=735 * 2)
    sf.write("ch7_0506_compress_ssi_model5_bs128_linear.wav", test_reconstruit, 44100)

    plt.show()

