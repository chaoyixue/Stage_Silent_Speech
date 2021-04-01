import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten, BatchNormalization
from tensorflow.keras.models import Model


def two_input_one_output_model():
    input_lips = Input(shape=(64, 64, 1))
    lips_conv1 = Conv2D(6, (5, 5), padding="same", activation="relu")(input_lips)
    lips_pooling1 = AveragePooling2D()(lips_conv1)
    lips_bn1 = BatchNormalization()(lips_pooling1)
    lips_conv2 = Conv2D(16, (5, 5), padding="valid", activation="relu")(lips_bn1)
    lips_pooling2 = AveragePooling2D()(lips_conv2)
    lips_bn2 = BatchNormalization()(lips_pooling2)
    lips_conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(lips_bn2)
    lips_pooling3 = AveragePooling2D()(lips_conv3)
    lips_bn3 = BatchNormalization()(lips_pooling3)
    lips_conv4 = Conv2D(64, (3, 3), padding="same", activation="relu")(lips_bn3)
    lips_pooling4 = AveragePooling2D()(lips_conv4)

    input_tongues = Input(shape=(64, 64, 1))
    tongues_conv1 = Conv2D(6, (5, 5), padding="same", activation="relu")(input_tongues)
    tongues_pooling1 = AveragePooling2D()(tongues_conv1)
    tongues_bn1 = BatchNormalization()(tongues_pooling1)
    tongues_conv2 = Conv2D(16, (5, 5), padding="valid", activation="relu")(tongues_bn1)
    tongues_pooling2 = AveragePooling2D()(tongues_conv2)
    tongues_bn2 = BatchNormalization()(tongues_pooling2)
    tongues_conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(tongues_bn2)
    tongues_pooling3 = AveragePooling2D()(tongues_conv3)
    tongues_bn3 = BatchNormalization()(tongues_pooling3)
    tongues_conv4 = Conv2D(64, (3, 3), padding="same", activation="relu")(tongues_bn3)
    tongues_pooling4 = AveragePooling2D()(tongues_conv4)

    cc = concatenate([lips_pooling4, tongues_pooling4])
    flat_layer = Flatten()(cc)
    fc1 = Dense(1024, activation="relu")(flat_layer)
    fc2 = Dense(736, activation="linear")(fc1)

    mymodel = Model([input_lips, input_tongues], fc2)
    return mymodel


if __name__ == "__main__":
    # load data
    X_lips = np.load("lips_all_chapiters.npy")
    X_tongues = np.load("tongues_all_chapiters.npy")
    Y = np.load("spectrogrammes_all_chapitre_corresponding.npy")

    # normalisation
    X_lips = X_lips/255.0
    X_tongues = X_tongues/255.0
    max_spectrum = np.max(Y)
    Y = Y/max_spectrum

    # split train set data
    lips_x_train = X_lips[:-15951]
    lips_x_test = X_lips[-15951:]
    tongues_x_train = X_tongues[:-15951]
    tongues_x_test = X_tongues[-15951:]
    y_train = Y[:, :-15951]
    y_test = Y[:, -15951:]
    y_train = np.transpose(y_train)
    y_test = np.transpose(y_test)

    test_model = two_input_one_output_model()
    test_model.summary()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    test_model.compile(my_optimizer, loss=tf.keras.losses.MeanSquaredError())

    filepath = "ssi_model2/ssi_model2-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[lips_x_train, tongues_x_train], y=y_train, batch_size=64, epochs=200, callbacks=callbacks_list,
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