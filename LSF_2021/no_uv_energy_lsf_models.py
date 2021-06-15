import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten, Dropout
from tensorflow.keras.models import Model


def no_uv_energy_lsfmodel1():
    input_lsp = Input(shape=(13,))
    lsp_fc1 = Dense(32, activation="relu")(input_lsp)
    lsp_fc2 = Dense(64, activation="relu")(lsp_fc1)
    lsp_fc3 = Dense(128, activation="relu")(lsp_fc2)

    input_f0 = Input(shape=(1,))
    f0_fc1 = Dense(16, activation="relu")(input_f0)
    f0_fc2 = Dense(32, activation="relu")(f0_fc1)

    input_energy = Input(shape=(1,))
    energy_fc1 = Dense(16, activation="relu")(input_energy)

    cc = concatenate([lsp_fc3, f0_fc2, energy_fc1])
    fc_result_0 = Dense(128, activation="relu")(cc)
    fc_result_1 = Dense(256, activation="relu")(fc_result_0)
    fc_result_2 = Dense(512, activation="relu")(fc_result_1)
    fc_result_3 = Dense(736, activation="linear")(fc_result_2)
    mymodel = Model([input_lsp, input_f0, input_energy], fc_result_3)
    return mymodel


def no_uv_energy_lsfmodel2():
    input_lsp = Input(shape=(13,))
    lsp_fc1 = Dense(32, activation="relu")(input_lsp)
    lsp_fc2 = Dense(64, activation="relu")(lsp_fc1)
    lsp_fc3 = Dense(128, activation="relu")(lsp_fc2)

    input_f0 = Input(shape=(1,))
    f0_fc1 = Dense(16, activation="relu")(input_f0)
    f0_fc2 = Dense(32, activation="relu")(f0_fc1)

    input_energy = Input(shape=(1,))
    energy_fc1 = Dense(16, activation="relu")(input_energy)

    cc = concatenate([lsp_fc3, f0_fc2, energy_fc1])
    fc_result_1 = Dense(256, activation="relu")(cc)
    fc_result_2 = Dense(512, activation="relu")(fc_result_1)
    fc_result_3 = Dense(736, activation="linear")(fc_result_2)
    mymodel = Model([input_lsp, input_f0, input_energy], fc_result_3)
    return mymodel


if __name__ == "__main__":
    # load data
    X_lsf = np.load("lsp_all_chapiter.npy")
    X_f0 = np.load("f0_all_chapiter_without_threshold.npy")
    energy = np.load("energy_all_chapiters.npy")
    spectrum = np.load("spectrogrammes_all_chapitre_corresponding.npy")

    # reshape the energy matrix to (84679,1)
    energy = energy.reshape((energy.shape[1], 1))
    print(energy.shape)

    # normalisation
    max_spectrum = np.max(spectrum)
    spectrum = spectrum / max_spectrum
    spectrum = np.matrix.transpose(spectrum)

    # split train test data
    lsf_train = X_lsf[:-15951]
    lsf_test = X_lsf[-15951:]
    f0_train = X_f0[:-15951]
    f0_test = X_f0[-15951:]
    energy_train = energy[:-15951]
    energy_test = energy[-15951:]

    y_train = spectrum[:-15951]
    y_test = spectrum[-15951:]

    test_model = no_uv_energy_lsfmodel1()
    test_model.summary()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-7)
    test_model.compile(my_optimizer, loss=tf.keras.losses.MeanSquaredError())
    filepath = "no_uv_energy_lsf_model2/no_uv_energy_lsf_model2-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[lsf_train, f0_train, energy_train], y=y_train, batch_size=64, epochs=2000,
                             callbacks=callbacks_list,
                             validation_data=([lsf_test, f0_test, energy_test], y_test))

