import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Model


def ssi_nonsilence_model1():
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
    fc1 = Dense(1024, activation="relu")(flat_layer)
    ssi_dr1 = Dropout(0.5)(fc1)
    fc2 = Dense(736, activation="linear")(ssi_dr1)

    mymodel = Model([input_lips, input_tongues], fc2)
    return mymodel


def ssi_nonsilence_model2():
    input_lip = Input(shape=(64, 64, 1))
    # encoding
    lip_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_lip)
    lip_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(lip_conv1)
    lip_pooling1 = MaxPooling2D(pool_size=(2, 2))(lip_conv2)
    lip_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lip_pooling1)
    lip_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lip_conv3)
    lip_pooling2 = MaxPooling2D(pool_size=(2, 2))(lip_conv4)
    lip_conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(lip_pooling2)
    lip_conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(lip_conv5)
    lip_pooling3 = MaxPooling2D(pool_size=(2, 2))(lip_conv6)

    input_tongue = Input(shape=(64, 64, 1))
    # encoding
    tongue_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_tongue)
    tongue_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(tongue_conv1)
    tongue_pooling1 = MaxPooling2D(pool_size=(2, 2))(tongue_conv2)
    tongue_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_pooling1)
    tongue_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_conv3)
    tongue_pooling2 = MaxPooling2D(pool_size=(2, 2))(tongue_conv4)

    flat_lip = Flatten()(lip_pooling3)
    flat_tongue = Flatten()(tongue_pooling2)
    cc = concatenate([flat_lip, flat_tongue])
    ssi_fc1 = Dense(1024, activation='relu', name='ssi_fc1')(cc)
    ssi_dr1 = Dropout(0)(ssi_fc1)
    ssi_fc2 = Dense(736, activation='linear', name='ssi_fc2')(ssi_dr1)
    new_model = Model([input_lip, input_tongue], ssi_fc2, name='transfer_autoencoder_lsf')
    new_model.summary()
    return new_model


if __name__ == "__main__":
    # load data
    X_lips = np.load("lip_nonsilence_all_chapiter.npy")
    X_tongues = np.load("tongue_nonsilence_all_chapiter.npy")
    Y = np.load("spectrum_nonsilence_all_chapiter.npy")

    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_spectrum = np.max(Y)
    Y = Y / max_spectrum

    # split train test data
    nb_nonsilence_chapiter7 = 10114
    lips_x_train = X_lips[:-nb_nonsilence_chapiter7]
    lips_x_test = X_lips[-nb_nonsilence_chapiter7:]
    tongues_x_train = X_tongues[:-nb_nonsilence_chapiter7]
    tongues_x_test = X_tongues[-nb_nonsilence_chapiter7:]
    y_train = Y[:-nb_nonsilence_chapiter7]
    y_test = Y[-nb_nonsilence_chapiter7:]

    test_model = ssi_nonsilence_model1()
    test_model.summary()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-7)
    test_model.compile(my_optimizer, loss=tf.keras.losses.MeanSquaredError())

    filepath = "ssi_nonsilence_model1_dr50/ssi_nonsilence_model1_dr50-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[lips_x_train, tongues_x_train], y=y_train, batch_size=64, epochs=500,
                             callbacks=callbacks_list,
                             validation_data=([lips_x_test, tongues_x_test], y_test))
