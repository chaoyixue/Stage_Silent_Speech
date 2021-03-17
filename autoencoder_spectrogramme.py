import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt


def model_autoencodeur(encoding_dim):
    myinput = keras.Input(shape=(736, ))
    # encoding parts
    fc1 = layers.Dense(368, activation='relu')(myinput)
    fc2 = layers.Dense(189, activation='relu')(fc1)
    encoded = layers.Dense(encoding_dim, activation='sigmoid')(fc2)

    # decoding parts
    fc3 = layers.Dense(189, activation='relu')(encoded)
    fc4 = layers.Dense(368, activation='relu')(fc3)
    decoded = layers.Dense(736, activation='sigmoid')(fc4)

    mymodel = keras.Model(myinput, decoded)
    return mymodel


if __name__ == "__main__":
    # load data
    X = np.load("spectrogrammes_all_chapitre.npy")
    # normalisation
    X = X/np.max(X)
    print(X.max())
    print(X.min())
    
    # split train test data
    x_train = np.transpose(X[:, :84776 - 15951])
    x_test = np.transpose(X[:, -15951:])
    print(x_train.shape)
    print(x_test.shape)

    test_model = model_autoencodeur(encoding_dim=30)
    test_model.summary()

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.8)
    my_optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

    test_model.compile(optimizer=my_optimizer, loss=tf.keras.losses.MeanSquaredError())

    filepath = "../autoencoder_model_0317/weights-improvement-{epoch:02d}-{val_loss:.5f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = test_model.fit(x=x_train, y=x_train, batch_size=64, epochs=50, callbacks=callbacks_list,
                             validation_split=0.2)
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
