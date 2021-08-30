import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, concatenate, Flatten, Conv2DTranspose, \
    UpSampling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model


def image2energy_data_coupe_model1():
    input_lip = Input(shape=(64, 64, 1))
    # encoding
    lip_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_lip)
    lip_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(lip_conv1)
    lip_pooling1 = MaxPooling2D(pool_size=(2, 2))(lip_conv2)
    lip_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lip_pooling1)
    lip_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lip_conv3)
    lip_pooling2 = MaxPooling2D(pool_size=(2, 2))(lip_conv4)

    input_tongue = Input(shape=(64, 64, 1))
    # encoding
    tongue_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_tongue)
    tongue_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(tongue_conv1)
    tongue_pooling1 = MaxPooling2D(pool_size=(2, 2))(tongue_conv2)
    tongue_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_pooling1)
    tongue_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_conv3)
    tongue_pooling2 = MaxPooling2D(pool_size=(2, 2))(tongue_conv4)

    flat_lip = Flatten()(lip_pooling2)
    flat_tongue = Flatten()(tongue_pooling2)
    cc = concatenate([flat_lip, flat_tongue])
    en_first = Dense(1024, activation='relu')(cc)
    en_dr1 = Dropout(0.5)(en_first)
    en_fc0 = Dense(256, activation='relu')(en_dr1)
    en_fc1 = Dense(128, activation='relu')(en_fc0)
    en_fc2 = Dense(32, activation='relu')(en_fc1)
    en_fc3 = Dense(1, activation='linear')(en_fc2)
    new_model = Model([input_lip, input_tongue], en_fc3, name='transfer_autoencoder_energy')
    new_model.summary()
    return new_model


if __name__ == "__main__":
    X_lip = np.load("lips_all_chapiters.npy")
    X_tongue = np.load("tongues_all_chapiters.npy")
    energy_original = np.load("energy_cut_all.npy")

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0
    max_energy = np.max(energy_original)
    energy_original = energy_original / max_energy

    # split train test data
    nb_image_chapiter7 = 15951
    X_lip_train = X_lip[:-nb_image_chapiter7]
    X_lip_test = X_lip[-nb_image_chapiter7:]
    X_tongue_train = X_tongue[:-nb_image_chapiter7]
    X_tongue_test = X_tongue[-nb_image_chapiter7:]
    energy_train = energy_original[:-15951]
    energy_test = energy_original[-15951:]

    test_model = image2energy_data_coupe_model1()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    test_model.compile(optimizer=my_optimizer, loss=tf.keras.losses.MeanSquaredError())
    filepath = "image2energy_data_coupe_model1/" \
               "image2energy_data_coupe_model1-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model
    callbacks_list = [checkpoint]
    history = test_model.fit(x=[X_lip_train, X_tongue_train], y=energy_train, batch_size=64, epochs=100,
                             callbacks=callbacks_list,
                             validation_data=([X_lip_test, X_tongue_test], energy_test))

