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


def encoder_lsf_model1():
    input_lip = Input(shape=(64, 64, 1))
    # encoding
    lip_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_lip)
    lip_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(lip_conv1)
    lip_pooling1 = MaxPooling2D(pool_size=(2, 2))(lip_conv2)
    lip_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lip_pooling1)
    lip_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lip_conv3)
    lip_pooling2 = MaxPooling2D(pool_size=(2, 2))(lip_conv4)

    # decoding
    lip_conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lip_pooling2)
    lip_conv6 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lip_conv5)
    lip_upsample1 = UpSampling2D(size=(2, 2))(lip_conv6)
    lip_conv7 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(lip_upsample1)
    lip_conv8 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(lip_conv7)
    lip_upsample2 = UpSampling2D(size=(2, 2))(lip_conv8)
    lip_decoded = Conv2D(filters=1, kernel_size=(5, 5),
                         activation='sigmoid', padding='same', name='lip_output')(lip_upsample2)

    input_tongue = Input(shape=(64, 64, 1))
    # encoding
    tongue_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_tongue)
    tongue_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(tongue_conv1)
    tongue_pooling1 = MaxPooling2D(pool_size=(2, 2))(tongue_conv2)
    tongue_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_pooling1)
    tongue_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_conv3)
    tongue_pooling2 = MaxPooling2D(pool_size=(2, 2))(tongue_conv4)

    # decoding
    tongue_conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_pooling2)
    tongue_conv6 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_conv5)
    tongue_upsample1 = UpSampling2D(size=(2, 2))(tongue_conv6)
    tongue_conv7 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(tongue_upsample1)
    tongue_conv8 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(tongue_conv7)
    tongue_upsample2 = UpSampling2D(size=(2, 2))(tongue_conv8)
    tongue_decoded = Conv2D(filters=1, kernel_size=(5, 5),
                            activation='sigmoid', padding='same', name='tongue_output')(tongue_upsample2)
    # take the encoder part of the two image inputs
    cc = concatenate([lip_pooling2, tongue_pooling2])
    lsf_dense1 = Dense(256, activation='relu')(cc)
    lsf_coefficients = Dense(13, activation='sigmoid', name='lsf_output')(lsf_dense1)
    mymodel = Model([input_lip, input_tongue], [lip_decoded, tongue_decoded, lsf_coefficients])
    return mymodel


if __name__ == "__main__":
    X_lip = np.load("lips_all_chapiters.npy")
    X_tongue = np.load("tongues_all_chapiters.npy")
    lsf_original = np.load("lsp_all_chapiter.npy")

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

    test_model = encoder_lsf_model1()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    losses = {
        "lip_output": "binary_crossentropy",
        "tongue_output": "categorical_crossentropy",
        "lsf_output": "mean_squared_error"
    }
    test_model.compile(optimizer=my_optimizer, loss=losses)

    filepath = "encoder_lsf/encoder_lsf-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[X_lip_train, X_tongue_train], y=[X_lip_train, X_tongue_train, lsf_train],
                             batch_size=256, epochs=100, callbacks=callbacks_list,
                             validation_data=([X_lip_test, X_tongue_test], [X_lip_test, X_tongue_test, lsf_test]))
