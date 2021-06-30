import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, concatenate, Flatten, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model


def transfer_autoencoder_lsf_model1():
    autoencoder_lip = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/"
                                              "week_0614/autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/"
                                                 "week_0614/autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(cc)
    lsf_fc2 = Dense(13, activation='linear', name='lsf_fc2')(lsf_fc1)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], lsf_fc2, name='transfer_autoencoder_lsf')
    new_model.summary()
    return new_model


def transfer_autoencoder_lsf_model2():
    autoencoder_lip = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/"
                                              "week_0614/autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/"
                                                 "week_0614/autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(cc)
    lsf_fc2 = Dense(128, activation='relu', name='lsf_fc2')(lsf_fc1)
    lsf_fc3 = Dense(13, activation='linear', name='lsf_fc3')(lsf_fc2)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], lsf_fc3, name='transfer_autoencoder_lsf')
    new_model.summary()
    return new_model


def transfer_autoencoder_lsf_model3():
    autoencoder_lip = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/"
                                              "week_0614/autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("C:/Users/chaoy/Desktop/StageSilentSpeech/results/"
                                                 "week_0614/autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    lsf_fc1 = Dense(1024, activation='relu', name='lsf_fc1')(cc)
    lsf_fc2 = Dense(13, activation='relu', name='lsf_fc2')(lsf_fc1)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], lsf_fc2, name='transfer_autoencoder_lsf')
    new_model.summary()
    return new_model


if __name__ == "__main__":
    X_lip = np.load("../../data_npy_one_image/lips_all_chapiters.npy")
    X_tongue = np.load("../../data_npy_one_image/tongues_all_chapiters.npy")
    lsf_original = np.load("../../LSF_data/lsp_all_chapiter.npy")

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0

    # split train test data
    nb_image_chapiter7 = 15951
    X_lip_train = X_lip[:-nb_image_chapiter7, :]
    X_lip_test = X_lip[-nb_image_chapiter7:, :]
    X_tongue_train = X_tongue[:-nb_image_chapiter7, :]
    X_tongue_test = X_tongue[-nb_image_chapiter7:, :]
    lsf_train = lsf_original[:-nb_image_chapiter7, :]
    lsf_test = lsf_original[-nb_image_chapiter7:, :]

    test_model = transfer_autoencoder_lsf_model2()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    test_model.compile(optimizer=my_optimizer, loss=tf.keras.losses.MeanSquaredError())
    filepath = "transfer_autoencoder_lsf_model2/transfer_autoencoder_lsf_model2-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[X_lip_train, X_tongue_train], y=lsf_train, batch_size=64, epochs=1000,
                             callbacks=callbacks_list,
                             validation_data=([X_lip_test, X_tongue_test], lsf_test))
