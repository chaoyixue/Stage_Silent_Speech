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


def image2lsf_model1():
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
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(cc)
    lsf_fc2 = Dense(13, activation='linear', name='lsf_fc2')(lsf_fc1)
    new_model = Model([input_lip, input_tongue], lsf_fc2, name='transfer_autoencoder_lsf')
    new_model.summary()
    return new_model


def image2lsf_model2():
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
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(cc)
    lsf_fc2 = Dense(13, activation='linear', name='lsf_fc2')(lsf_fc1)
    new_model = Model([input_lip, input_tongue], lsf_fc2, name='transfer_autoencoder_lsf')
    new_model.summary()
    return new_model


def image2lsf_model3():
    input_lip = Input(shape=(64, 64, 1), name='lipinput')
    # encoding
    lip_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lipconv1')(input_lip)
    lip_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lipconv2')(lip_conv1)
    lip_pooling1 = MaxPooling2D(pool_size=(2, 2), name='lippool1')(lip_conv2)
    lip_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv3')(lip_pooling1)
    lip_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv4')(lip_conv3)
    lip_pooling2 = MaxPooling2D(pool_size=(2, 2), name='lippool2')(lip_conv4)
    lip_conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv5')(lip_pooling2)
    lip_conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv6')(lip_conv5)
    lip_pooling3 = MaxPooling2D(pool_size=(2, 2), name='lippool3')(lip_conv6)

    input_tongue = Input(shape=(64, 64, 1), name='tongueinput')
    # encoding
    tongue_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same',
                          name='tongueconv1')(input_tongue)
    tongue_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same',
                          name='tongueconv2')(tongue_conv1)
    tongue_pooling1 = MaxPooling2D(pool_size=(2, 2), name='tonguepool1')(tongue_conv2)
    tongue_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                          name='tongueconv3')(tongue_pooling1)
    tongue_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                          name='tongueconv4')(tongue_conv3)
    tongue_pooling2 = MaxPooling2D(pool_size=(2, 2), name='tonguepool2')(tongue_conv4)
    tongue_conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                          name='tongueconv5')(tongue_pooling2)
    tongue_conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                          name='tongueconv6')(tongue_conv5)
    tongue_pooling3 = MaxPooling2D(pool_size=(2, 2), name='tonguepool3')(tongue_conv6)

    flat_lip = Flatten()(lip_pooling3)
    flat_tongue = Flatten()(tongue_pooling3)
    cc = concatenate([flat_lip, flat_tongue])
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(cc)
    lsf_fc2 = Dense(13, activation='linear', name='lsf_fc2')(lsf_fc1)
    new_model = Model([input_lip, input_tongue], lsf_fc2, name='image2lsf_model3')
    new_model.summary()
    return new_model


def image2lsf_model4():
    input_lip = Input(shape=(64, 64, 1), name='lipinput')
    # encoding
    lip_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lipconv1')(input_lip)
    lip_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lipconv2')(lip_conv1)
    lip_pooling1 = MaxPooling2D(pool_size=(2, 2), name='lippool1')(lip_conv2)
    lip_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv3')(lip_pooling1)
    lip_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv4')(lip_conv3)
    lip_pooling2 = MaxPooling2D(pool_size=(2, 2), name='lippool2')(lip_conv4)
    lip_conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv5')(lip_pooling2)
    lip_conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv6')(lip_conv5)
    lip_pooling3 = MaxPooling2D(pool_size=(2, 2), name='lippool3')(lip_conv6)
    lip_conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                       name='lip_conv7')(lip_pooling3)
    lip_pooling4 = MaxPooling2D(pool_size=(2, 2), name='lippool4')(lip_conv7)

    input_tongue = Input(shape=(64, 64, 1), name='tongueinput')
    # encoding
    tongue_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same',
                          name='tongueconv1')(input_tongue)
    tongue_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same',
                          name='tongueconv2')(tongue_conv1)
    tongue_pooling1 = MaxPooling2D(pool_size=(2, 2), name='tonguepool1')(tongue_conv2)
    tongue_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                          name='tongueconv3')(tongue_pooling1)
    tongue_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                          name='tongueconv4')(tongue_conv3)
    tongue_pooling2 = MaxPooling2D(pool_size=(2, 2), name='tonguepool2')(tongue_conv4)

    flat_lip = Flatten()(lip_pooling4)
    flat_tongue = Flatten()(tongue_pooling2)
    cc = concatenate([flat_lip, flat_tongue])
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(cc)
    lsf_fc2 = Dense(13, activation='linear', name='lsf_fc2')(lsf_fc1)
    new_model = Model([input_lip, input_tongue], lsf_fc2, name='image2lsf_model3')
    new_model.summary()
    return new_model


def image2lsf_model5():
    input_lip = Input(shape=(64, 64, 1), name='lipinput')
    # encoding
    lip_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lipconv1')(input_lip)
    lip_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same', name='lipconv2')(lip_conv1)
    lip_pooling1 = MaxPooling2D(pool_size=(2, 2), name='lippool1')(lip_conv2)
    lip_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv3')(lip_pooling1)
    lip_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv4')(lip_conv3)
    lip_pooling2 = MaxPooling2D(pool_size=(2, 2), name='lippool2')(lip_conv4)
    lip_conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv5')(lip_pooling2)
    lip_conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='lipconv6')(lip_conv5)
    lip_pooling3 = MaxPooling2D(pool_size=(2, 2), name='lippool3')(lip_conv6)

    input_tongue = Input(shape=(64, 64, 1), name='tongueinput')
    # encoding
    tongue_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same',
                          name='tongueconv1')(input_tongue)
    tongue_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same',
                          name='tongueconv2')(tongue_conv1)
    tongue_pooling1 = MaxPooling2D(pool_size=(2, 2), name='tonguepool1')(tongue_conv2)
    tongue_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                          name='tongueconv3')(tongue_pooling1)
    tongue_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                          name='tongueconv4')(tongue_conv3)
    tongue_pooling2 = MaxPooling2D(pool_size=(2, 2), name='tonguepool2')(tongue_conv4)

    flat_lip = Flatten()(lip_pooling3)
    flat_tongue = Flatten()(tongue_pooling2)
    cc = concatenate([flat_lip, flat_tongue])
    lsf_fc0 = Dense(1024, activation='relu', name='lsf_fc0')(cc)
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(lsf_fc0)
    lsf_fc2 = Dense(13, activation='linear', name='lsf_fc2')(lsf_fc1)
    new_model = Model([input_lip, input_tongue], lsf_fc2, name='image2lsf_model5')
    new_model.summary()
    return new_model


def image2lsf_model6():
    input_lip = Input(shape=(64, 64, 1))
    # encoding
    lip_conv1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_lip)
    lip_conv2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(lip_conv1)
    lip_bn1 = BatchNormalization(name='lip_bn1')(lip_conv2)
    lip_pooling1 = MaxPooling2D(pool_size=(2, 2))(lip_bn1)
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
    tongue_bn1 = BatchNormalization(name='tongue_bn1')(tongue_conv2)
    tongue_pooling1 = MaxPooling2D(pool_size=(2, 2))(tongue_bn1)
    tongue_conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_pooling1)
    tongue_conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(tongue_conv3)
    tongue_pooling2 = MaxPooling2D(pool_size=(2, 2))(tongue_conv4)

    flat_lip = Flatten()(lip_pooling3)
    flat_tongue = Flatten()(tongue_pooling2)
    cc = concatenate([flat_lip, flat_tongue])
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(cc)
    lsf_fc2 = Dense(13, activation='linear', name='lsf_fc2')(lsf_fc1)
    new_model = Model([input_lip, input_tongue], lsf_fc2, name='transfer_autoencoder_lsf')
    new_model.summary()
    return new_model


def image2lsf_model7():
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
    lsf_fc1 = Dense(256, activation='relu', name='lsf_fc1')(cc)
    lsf_dr1 = Dropout(0.3)(lsf_fc1)
    lsf_fc2 = Dense(13, activation='linear', name='lsf_fc2')(lsf_dr1)
    new_model = Model([input_lip, input_tongue], lsf_fc2, name='transfer_autoencoder_lsf')
    new_model.summary()
    return new_model


if __name__ == "__main__":
    X_lip = np.load("lips_all_chapiters.npy")
    X_tongue = np.load("tongues_all_chapiters.npy")
    lsf_original = np.load("lsp_all_chapiter.npy")

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0

    # add the normalization of lsf values
    # max_lsf = np.max(lsf_original)
    # lsf_original = lsf_original/max_lsf
    # print("the maximum lsf value %f :" % max_lsf)
    # split train test data
    nb_image_chapiter7 = 15951
    X_lip_train = X_lip[:-nb_image_chapiter7, :]
    X_lip_test = X_lip[-nb_image_chapiter7:, :]
    X_tongue_train = X_tongue[:-nb_image_chapiter7, :]
    X_tongue_test = X_tongue[-nb_image_chapiter7:, :]
    lsf_train = lsf_original[:-nb_image_chapiter7, :]
    lsf_test = lsf_original[-nb_image_chapiter7:, :]

    test_model = image2lsf_model7()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    test_model.compile(optimizer=my_optimizer, loss=tf.keras.losses.MeanSquaredError())
    filepath = "image2lsf_model7_normalized_dr30/image2lsf_model7_normalized_dr30-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[X_lip_train, X_tongue_train], y=lsf_train, batch_size=64, epochs=200,
                             callbacks=callbacks_list,
                             validation_data=([X_lip_test, X_tongue_test], lsf_test))
