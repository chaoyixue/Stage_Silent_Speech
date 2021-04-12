import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten, TimeDistributed, LeakyReLU,\
    Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model


def ssi_model_8():
    input_lips = Input(shape=(5, 64, 64, 1))
    lips_conv1 = Conv2D(32, (5, 5), padding="same", activation="relu")
    lips_tm1 = TimeDistributed(lips_conv1)(input_lips)
    lips_pooling1 = TimeDistributed(MaxPooling2D((2, 2)))(lips_tm1)
    lips_dr1 = Dropout(0.5)(lips_pooling1)
    lips_bn1 = BatchNormalization()(lips_dr1)

    input_tongues = Input(shape=(5, 64, 64, 1))
    tongues_conv1 = Conv2D(32, (5, 5), padding="same", activation="relu")
    tongues_tm1 = TimeDistributed(tongues_conv1)(input_tongues)
    tongues_pooling1 = TimeDistributed(MaxPooling2D(2, 2))(tongues_tm1)
    tongues_dr1 = Dropout(0.5)(tongues_pooling1)
    tongues_bn1 = BatchNormalization()(tongues_dr1)

    cc = concatenate([lips_bn1, tongues_bn1])
    flat_layer = Flatten()(cc)
    fc1 = Dense(1024)(flat_layer)
    lk1 = LeakyReLU(0.3)(fc1)
    dr1 = Dropout(0.5)(lk1)
    fc2 = Dense(736)(dr1)
    lk2 = LeakyReLU(0.3)(fc2)

    mymodel = Model([input_lips, input_tongues], lk2)
    return mymodel


if __name__ == "__main__":
    # load data
    X_lips = np.load("../five_recurrent_image_npy/lips/lips_recurrent_5images_all_chapitres.npy")
    X_tongues = np.load("../five_recurrent_image_npy/tongues/tongues_recurrent_5images_all_chapitres.npy")
    Y = np.load("../five_recurrent_image_npy/spectrum_recurrent_all.npy")

    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_spectrum = np.max(Y)
    Y = Y / max_spectrum

    # split train set data
    lips_x_train = X_lips[:-15947]
    lips_x_test = X_lips[-15947:]
    tongues_x_train = X_tongues[:-15947]
    tongues_x_test = X_tongues[-15947:]
    y_train = Y[:, :-15947]
    y_test = Y[:, -15947:]
    y_train = np.transpose(y_train)
    y_test = np.transpose(y_test)

    test_model = ssi_model_8()
    test_model.summary()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-7)
    test_model.compile(my_optimizer, loss=tf.keras.losses.MeanSquaredError())

    filepath = "../ssi_model8_dr50/ssi_model8_dr50-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[lips_x_train, tongues_x_train], y=y_train, batch_size=64, epochs=200,
                             callbacks=callbacks_list,
                             validation_data=([lips_x_test, tongues_x_test], y_test))
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()