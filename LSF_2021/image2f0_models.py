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


def image2f0_model1():
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
    f0_fc1 = Dense(256, activation='relu', name='f0_fc1')(cc)
    f0_dr1 = Dropout(0.5)(f0_fc1)
    f0_fc2 = Dense(1, activation='linear', name='f0_fc2')(f0_dr1)
    new_model = Model([input_lip, input_tongue], f0_fc2, name='image2f0_model')
    new_model.summary()
    return new_model


if __name__ == "__main__":
    X_lip = np.load("lips_all_chapiters.npy")
    X_tongue = np.load("tongues_all_chapiters.npy")
    f0_original = np.load("f0_all_chapiter.npy")

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0
    max_f0 = np.max(f0_original)
    f0_original = f0_original / max_f0
    print(max_f0)

    # split train test data
    nb_image_chapiter7 = 15951
    X_lip_train = X_lip[:-nb_image_chapiter7, :]
    X_lip_test = X_lip[-nb_image_chapiter7:, :]
    X_tongue_train = X_tongue[:-nb_image_chapiter7, :]
    X_tongue_test = X_tongue[-nb_image_chapiter7:, :]
    f0_train = f0_original[:-nb_image_chapiter7]
    f0_test = f0_original[-nb_image_chapiter7:]

    test_model = image2f0_model1()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    test_model.compile(optimizer=my_optimizer, loss=tf.keras.losses.MeanSquaredError())
    filepath = "image2f0_model1_dr50/image2f0_model1_dr50-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[X_lip_train, X_tongue_train], y=f0_train, batch_size=64, epochs=1000,
                             callbacks=callbacks_list,
                             validation_data=([X_lip_test, X_tongue_test], f0_test))
