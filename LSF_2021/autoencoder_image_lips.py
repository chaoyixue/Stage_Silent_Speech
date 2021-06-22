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


def autoencoder_lips():
    input_img = Input(shape=(64, 64, 1))
    # encoding
    conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lip_conv1')(input_img)
    conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lip_conv2')(conv1)
    pooling1 = MaxPooling2D(pool_size=(2, 2), name='lip_pooling1')(conv2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lip_conv3')(pooling1)
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lip_conv4')(conv3)
    pooling2 = MaxPooling2D(pool_size=(2, 2), name='lip_pooling2')(conv4)

    # decoding
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lip_conv5')(pooling2)
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lip_conv6')(conv5)
    upsample1 = UpSampling2D(size=(2, 2), name='lip_upsample1')(conv6)
    conv7 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lip_conv7')(upsample1)
    conv8 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lip_conv8')(conv7)
    upsample2 = UpSampling2D(size=(2, 2), name='lip_upsample2')(conv8)
    decoded = Conv2D(filters=1, kernel_size=(5, 5), activation='sigmoid', padding='same', name='lip_decoded')(upsample2)

    autoencoder_lip = Model(input_img, decoded, name='autoencoder_lips')
    autoencoder_lip.summary()
    return autoencoder_lip


if __name__ == "__main__":
    X = np.load("C:/Users/chaoy/Desktop/StageSilentSpeech/data_npy_one_image/lips_all_chapiters.npy")
    nb_images_chapiter7 = 15951

    # normalisation
    X = X/255.0

    # ch1-ch6
    X_train = X[:-15951, :]
    X_test = X[-15951:, :]

    model = autoencoder_lips()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    model.compile(optimizer=my_optimizer, loss='binary_crossentropy')

    filepath = "autoencoder_lips/autoencoder_lips-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = model.fit(x=X_train, y=X_train, batch_size=256, epochs=100, callbacks=callbacks_list,
                        validation_data=(X_test, X_test))
