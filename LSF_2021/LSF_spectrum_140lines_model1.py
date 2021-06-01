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


def lsp_spectrum_model1():
    input_lsp = Input(shape=(13,))
    lsp_fc1 = Dense(32, activation="relu")(input_lsp)
    lsp_fc2 = Dense(64, activation="relu")(lsp_fc1)
    lsp_fc3 = Dense(128, activation="relu")(lsp_fc2)

    input_f0 = Input(shape=(1, ))
    f0_fc1 = Dense(16, activation="relu")(input_f0)

    input_uv = Input(shape=(1, ))
    uv_fc1 = Dense(16, activation="relu")(input_uv)

    cc = concatenate([lsp_fc3, f0_fc1, uv_fc1])
    fc_result = Dense(140, activation="linear")(cc)
    mymodel = Model([input_lsp, input_f0, input_uv], fc_result)
    return mymodel


if __name__ == "__main__":
    # load data
    X_lsf = np.load("lsp_all_chapiter.npy")
    X_f0 = np.load("f0_all_chapiter.npy")
    X_uv = np.load("uv_all_chapiter.npy")
    spectrum = np.load("spectrogrammes_all_chapitre_corresponding.npy")

    # X_lsf = X_lsf.reshape(len(X_lsf), X_lsf.shape[1], 1)
    # X_f0 = X_f0.reshape(len(X_f0), 1, 1)
    # X_uv = X_uv.reshape(len(X_uv), 1, 1)

    # normalisation
    max_spectrum = np.max(spectrum)
    spectrum = spectrum / max_spectrum
    # only consider the low frequency
    spectrum = np.matrix.transpose(spectrum)[:, :140]

    # split train test data
    lsf_train = X_lsf[:-15951]
    lsf_test = X_lsf[-15951:]
    f0_train = X_f0[:-15951]
    f0_test = X_f0[-15951:]
    uv_train = X_uv[:-15951]
    uv_test = X_uv[-15951:]

    y_train = spectrum[:-15951]
    y_test = spectrum[-15951:]

    test_model = lsp_spectrum_model1()
    test_model.summary()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-7)
    test_model.compile(my_optimizer, loss=tf.keras.losses.MeanSquaredError())
    filepath = "lsf_spectrum_140lines_model1/lsf_spectrum_140lines_model1-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[lsf_train, f0_train, uv_train], y=y_train, batch_size=64, epochs=200,
                             callbacks=callbacks_list,
                             validation_data=([lsf_test, f0_test, uv_test], y_test))

