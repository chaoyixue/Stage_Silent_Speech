import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt


def model_autoencodeur(encoding_dim):
    myinput = keras.Input(shape=(736,))
    # encoding parts
    fc = layers.Dense(512, activation='relu')(myinput)
    fc1 = layers.Dense(368, activation='relu')(fc)
    fc2 = layers.Dense(189, activation='relu')(fc1)
    fc3 = layers.Dense(100, activation='relu')(fc2)
    encoded = layers.Dense(encoding_dim, activation='sigmoid')(fc3)

    # decoding parts
    fc4 = layers.Dense(100, activation='relu')(encoded)
    fc5 = layers.Dense(189, activation='relu')(fc4)
    fc6 = layers.Dense(368, activation='relu')(fc5)
    fc7 = layers.Dense(512, activation='relu')(fc6)
    decoded = layers.Dense(736, activation='sigmoid')(fc7)
    mymodel = keras.Model(myinput, decoded)
    return mymodel


if __name__ == "__main__":
    # load data
    X = np.load("spectrogrammes_all_chapitre.npy")

    # normalisation
    X = X / np.max(X)
    print(X.max())
    print(X.min())

    # split train test data
    x_train = np.transpose(X[:, :84776 - 15951])
    x_test = np.transpose(X[:, -15951:])
    print(x_train.shape)
    print(x_test.shape)

    test_model = model_autoencodeur(encoding_dim=30)
    test_model.summary()

    my_optimizer = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8)

    test_model.compile(optimizer=my_optimizer, loss=tf.keras.losses.MeanSquaredError())

    filepath = "four_layers_model/weights-improvement-{epoch:02d}-{val_loss:.5f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=x_train, y=x_train, batch_size=64, epochs=200, callbacks=callbacks_list,
                             validation_data=(x_test, x_test))
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
