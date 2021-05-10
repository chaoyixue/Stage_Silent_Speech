import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten
from tensorflow.keras.models import Model


def two_parts_ssi_model1():
    input_lips = Input(shape=(64, 64, 1))
    lips_conv1 = Conv2D(6, (5, 5), padding="same", activation="relu")(input_lips)
    lips_pooling1 = AveragePooling2D()(lips_conv1)
    lips_conv2 = Conv2D(16, (5, 5), padding="valid", activation="relu")(lips_pooling1)
    lips_pooling2 = AveragePooling2D()(lips_conv2)

    input_tongues = Input(shape=(64, 64, 1))
    tongues_conv1 = Conv2D(6, (5, 5), padding="same", activation="relu")(input_tongues)
    tongues_pooling1 = AveragePooling2D()(tongues_conv1)
    tongues_conv2 = Conv2D(16, (5, 5), padding="valid", activation="relu")(tongues_pooling1)
    tongues_pooling2 = AveragePooling2D()(tongues_conv2)

    cc = concatenate([lips_pooling2, tongues_pooling2])
    flat_layer = Flatten()(cc)
    fc_real = Dense(736, activation="linear", name="real_output")(flat_layer)
    fc_imag = Dense(736, activation="linear", name="imag_output")(flat_layer)

    mymodel = Model([input_lips, input_tongues], [fc_real, fc_imag])
    return mymodel


if __name__ == "__main__":

    # load data
    X_lips = np.load("lips_all_chapiters.npy")
    X_tongues = np.load("tongues_all_chapiters.npy")
    # load the module matrix of the spectrogram
    Y_module = np.load("spectrogrammes_all_chapitre_corresponding.npy")
    # load the real part of the spectrogram
    y_real = np.load("real_spectrograme.npy")
    # load the imaginary part of the spectrogram
    y_imag = np.load("imag_spectrograme.npy")
    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_spectrum = np.max(Y_module)
    y_real /= max_spectrum
    y_imag /= max_spectrum

    # split train set data
    lips_x_train = X_lips[:-15951]
    lips_x_test = X_lips[-15951:]
    tongues_x_train = X_tongues[:-15951]
    tongues_x_test = X_tongues[-15951:]
    # train test split for the real parts
    y_real = np.transpose(y_real)
    y_train_real = y_real[:-15951, :]
    y_test_real = y_real[-15951:, :]
    # train test split for the imaginary parts
    y_imag = np.transpose(y_imag)
    y_train_imag = y_imag[:-15951, :]
    y_test_imag = y_imag[-15951:, :]

    test_model = two_parts_ssi_model1()
    test_model.summary()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-7)
    test_model.compile(my_optimizer, loss={'real_output': tf.keras.losses.MeanSquaredError(),
                                           'imag_output': tf.keras.losses.MeanSquaredError()})

    filepath = "two_parts_ssi_model1/two_parts_ssi_model1-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[lips_x_train, tongues_x_train], y=[y_train_real, y_train_imag], batch_size=128,
                             epochs=100,
                             callbacks=callbacks_list,
                             validation_data=([lips_x_test, tongues_x_test], [y_test_real, y_test_imag]))
