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


def image2uv_data_coupe_model1():
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
    f0_fc1 = Dense(128, activation='relu')(cc)
    f0_fc2 = Dense(1, activation='sigmoid')(f0_fc1)
    new_model = Model([input_lip, input_tongue], f0_fc2, name='image2uv_model')
    new_model.summary()
    return new_model


if __name__ == "__main__":
    X_lip = np.load("lips_all_chapiters.npy")
    X_tongue = np.load("tongues_all_chapiters.npy")
    uv_original = np.load("uv_cut_all.npy")

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0

    # split train test data
    nb_image_chapiter7 = 15951
    X_lip_train = X_lip[:-nb_image_chapiter7]
    X_lip_test = X_lip[-nb_image_chapiter7:]
    X_tongue_train = X_tongue[:-nb_image_chapiter7]
    X_tongue_test = X_tongue[-nb_image_chapiter7:]
    uv_train = uv_original[:-nb_image_chapiter7]
    uv_test = uv_original[-nb_image_chapiter7:]

    test_model = image2uv_data_coupe_model1()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    test_model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    filepath = "image2uv_data_coupe_model1/image2uv_data_coupe_model1-{epoch:02d}-{val_binary_accuracy:.5f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model
    callbacks_list = [checkpoint]
    history = test_model.fit(x=[X_lip_train, X_tongue_train], y=uv_train, batch_size=64, epochs=1000,
                             callbacks=callbacks_list,
                             validation_data=([X_lip_test, X_tongue_test], uv_test))
