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


def transfer_autoencoder_f0_model1():
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
    f0_fc1 = Dense(128, activation='relu')(cc)
    f0_fc2 = Dense(1, activation='linear')(f0_fc1)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], f0_fc2, name='transfer_autoencoder_f0')
    new_model.summary()
    return new_model


if __name__ == "__main__":
    X_lip = np.load("../../data_npy_one_image/lips_all_chapiters.npy")
    X_tongue = np.load("../../data_npy_one_image/tongues_all_chapiters.npy")
    f0_original = np.load("../../LSF_data/f0_all_chapiter.npy")

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0
    max_f0 = np.max(f0_original)
    f0_original = f0_original/max_f0
    print(max_f0)

    # split train test data
    nb_image_chapiter7 = 15951
    X_lip_train = X_lip[:-nb_image_chapiter7, :]
    X_lip_test = X_lip[-nb_image_chapiter7:, :]
    X_tongue_train = X_tongue[:-nb_image_chapiter7, :]
    X_tongue_test = X_tongue[-nb_image_chapiter7:, :]
    f0_train = f0_original[:-nb_image_chapiter7]
    f0_test = f0_original[-nb_image_chapiter7:]

    test_model = transfer_autoencoder_f0_model1()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    test_model.compile(optimizer=my_optimizer, loss=tf.keras.losses.MeanSquaredError())
    filepath = "transfer_autoencoder_f0_model1/transfer_autoencoder_f0_model1-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[X_lip_train, X_tongue_train], y=f0_train, batch_size=64, epochs=1000,
                             callbacks=callbacks_list,
                             validation_data=([X_lip_test, X_tongue_test], f0_test))
