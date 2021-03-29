import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten, MaxPooling2D, \
    BatchNormalization, Dropout
from tensorflow.keras.models import Model


def two_input_one_output_model():
    input_lips = Input(shape=(64, 64, 1))
    lips_conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(input_lips)
    lips_conv2 = Conv2D(16, (3, 3), padding="same", activation="relu")(lips_conv1)
    lips_pooling1 = MaxPooling2D(pool_size=(2, 2))(lips_conv2)
    lips_bn1 = BatchNormalization()(lips_pooling1)
    lips_dr1 = Dropout(0.3)(lips_bn1)
    lips_conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(lips_dr1)
    lips_conv4 = Conv2D(32, (3, 3), padding="same", activation="relu")(lips_conv3)
    lips_pooling2 = MaxPooling2D(pool_size=(2, 2))(lips_conv4)
    lips_bn2 = BatchNormalization()(lips_pooling2)
    lips_dr2 = Dropout(0.3)(lips_bn2)
    lips_conv5 = Conv2D(64, (3, 3), padding="same", activation="relu")(lips_dr2)
    lips_conv6 = Conv2D(64, (3, 3), padding="same", activation="relu")(lips_conv5)
    lips_pooling3 = MaxPooling2D(pool_size=(2, 2))(lips_conv6)
    lips_bn3 = BatchNormalization()(lips_pooling3)
    lips_dr3 = Dropout(0.3)(lips_bn3)
    lips_conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(lips_dr3)
    lips_conv8 = Conv2D(128, (3, 3), padding="same", activation="relu")(lips_conv7)
    lips_pooling4 = MaxPooling2D(pool_size=(2, 2))(lips_conv8)
    lips_bn4 = BatchNormalization()(lips_pooling4)
    lips_dr4 = Dropout(0.3)(lips_bn4)
    lips_conv9 = Conv2D(256, (3, 3), padding="same", activation="relu")(lips_dr4)
    lips_conv10 = Conv2D(256, (3, 3), padding="same", activation="relu")(lips_conv9)
    lips_pooling5 = MaxPooling2D(pool_size=(2, 2))(lips_conv10)

    input_tongues = Input(shape=(64, 64, 1))
    tongues_conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(input_tongues)
    tongues_conv2 = Conv2D(16, (3, 3), padding="same", activation="relu")(tongues_conv1)
    tongues_pooling1 = MaxPooling2D(pool_size=(2, 2))(tongues_conv2)
    tongues_bn1 = BatchNormalization()(tongues_pooling1)
    tongues_dr1 = Dropout(0.3)(tongues_bn1)
    tongues_conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(tongues_dr1)
    tongues_conv4 = Conv2D(32, (3, 3), padding="same", activation="relu")(tongues_conv3)
    tongues_pooling2 = MaxPooling2D(pool_size=(2, 2))(tongues_conv4)
    tongues_bn2 = BatchNormalization()(tongues_pooling2)
    tongues_dr2 = Dropout(0.3)(tongues_bn2)
    tongues_conv5 = Conv2D(64, (3, 3), padding="same", activation="relu")(tongues_dr2)
    tongues_conv6 = Conv2D(64, (3, 3), padding="same", activation="relu")(tongues_conv5)
    tongues_pooling3 = MaxPooling2D(pool_size=(2, 2))(tongues_conv6)
    tongues_bn3 = BatchNormalization()(tongues_pooling3)
    tongues_dr3 = Dropout(0.3)(tongues_bn3)
    tongues_conv7 = Conv2D(128, (3, 3), padding="same", activation="relu")(tongues_dr3)
    tongues_conv8 = Conv2D(128, (3, 3), padding="same", activation="relu")(tongues_conv7)
    tongues_pooling4 = MaxPooling2D(pool_size=(2, 2))(tongues_conv8)
    tongues_bn4 = BatchNormalization()(tongues_pooling4)
    tongues_dr4 = Dropout(0.3)(tongues_bn4)
    tongues_conv9 = Conv2D(256, (3, 3), padding="same", activation="relu")(tongues_dr4)
    tongues_conv10 = Conv2D(256, (3, 3), padding="same", activation="relu")(tongues_conv9)
    tongues_pooling5 = MaxPooling2D(pool_size=(2, 2))(tongues_conv10)

    cc = concatenate([lips_pooling5, tongues_pooling5])
    flat_layer = Flatten()(cc)
    fc0 = Dense(2048, activation="relu")(flat_layer)
    fc1 = Dense(1024, activation="relu")(fc0)
    fc2 = Dense(736, activation="sigmoid")(fc1)

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

    filepath = "ssi_model4/weights-improvement-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=[lips_x_train, tongues_x_train], y=y_train, batch_size=64, epochs=50, callbacks=callbacks_list,
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