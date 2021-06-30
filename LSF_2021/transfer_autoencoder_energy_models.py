import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, concatenate, Flatten, Dropout
from tensorflow.keras.models import Model


def transfer_autoencoder_energy_model1():
    autoencoder_lip = keras.models.load_model("autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    f0_fc1 = Dense(128, activation='relu')(cc)
    f0_fc2 = Dense(1, activation='linear')(f0_fc1)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], f0_fc2, name='transfer_autoencoder_energy')
    new_model.summary()
    return new_model


def transfer_autoencoder_energy_model2():
    autoencoder_lip = keras.models.load_model("autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    f0_fc0 = Dense(256, activation='relu')(cc)
    f0_fc1 = Dense(128, activation='relu')(f0_fc0)
    f0_fc2 = Dense(1, activation='linear')(f0_fc1)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], f0_fc2, name='transfer_autoencoder_energy')
    new_model.summary()
    return new_model


def transfer_autoencoder_energy_model3():
    autoencoder_lip = keras.models.load_model("autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    f0_fc0 = Dense(256, activation='relu')(cc)
    f0_fc1 = Dense(128, activation='relu')(f0_fc0)
    f0_fc2 = Dense(32, activation='relu')(f0_fc1)
    f0_fc3 = Dense(1, activation='linear')(f0_fc2)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], f0_fc3, name='transfer_autoencoder_energy')
    new_model.summary()
    return new_model


def transfer_autoencoder_energy_model4():
    autoencoder_lip = keras.models.load_model("autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    f0_fc0 = Dense(256, activation='relu')(cc)
    f0_fc1 = Dense(128, activation='relu')(f0_fc0)
    f0_fc2 = Dense(64, activation='relu')(f0_fc1)
    f0_fc3 = Dense(32, activation='relu')(f0_fc2)
    f0_fc4 = Dense(1, activation='linear')(f0_fc3)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], f0_fc4, name='transfer_autoencoder_energy')
    new_model.summary()
    return new_model


def transfer_autoencoder_energy_model5():
    autoencoder_lip = keras.models.load_model("autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    f0_first = Dense(1024, activation='relu')(cc)
    f0_fc0 = Dense(256, activation='relu')(f0_first)
    f0_fc1 = Dense(128, activation='relu')(f0_fc0)
    f0_fc2 = Dense(32, activation='relu')(f0_fc1)
    f0_fc3 = Dense(1, activation='linear')(f0_fc2)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], f0_fc3, name='transfer_autoencoder_energy')
    new_model.summary()
    return new_model


def transfer_autoencoder_energy_model6():
    autoencoder_lip = keras.models.load_model("autoencoder_lips-100-0.48278144.h5")
    autoencoder_tongue = keras.models.load_model("autoencoder_tongues-100-0.06104492.h5")
    autoencoder_lip.summary()
    autoencoder_tongue.summary()
    output_lip = autoencoder_lip.get_layer("lip_pooling2").output
    output_tongue = autoencoder_tongue.get_layer("tongue_pooling2").output
    flat_lip = Flatten(name='flat_lip')(output_lip)
    flat_tongue = Flatten(name='flat_tongue')(output_tongue)
    cc = concatenate([flat_lip, flat_tongue])
    en_first = Dense(1024, activation='relu')(cc)
    en_dr1 = Dropout(0.5)(en_first)
    en_fc0 = Dense(256, activation='relu')(en_dr1)
    en_fc1 = Dense(128, activation='relu')(en_fc0)
    en_fc2 = Dense(32, activation='relu')(en_fc1)
    en_fc3 = Dense(1, activation='linear')(en_fc2)
    new_model = Model([autoencoder_lip.input, autoencoder_tongue.input], en_fc3, name='transfer_autoencoder_energy')
    new_model.summary()
    return new_model


if __name__ == "__main__":
    X_lip = np.load("lips_all_chapiters.npy")
    X_tongue = np.load("tongues_all_chapiters.npy")
    energy_original = np.load("energy_all_chapiters.npy")
    energy_original = energy_original.reshape((energy_original.shape[1], 1))

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0
    max_energy = np.max(energy_original)
    energy_original = energy_original/max_energy

    # split train test data
    nb_image_chapiter7 = 15951
    X_lip_train = X_lip[:-nb_image_chapiter7, :]
    X_lip_test = X_lip[-nb_image_chapiter7:, :]
    X_tongue_train = X_tongue[:-nb_image_chapiter7, :]
    X_tongue_test = X_tongue[-nb_image_chapiter7:, :]
    energy_train = energy_original[:-15951]
    energy_test = energy_original[-15951:]

    test_model = transfer_autoencoder_energy_model6()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    test_model.compile(optimizer=my_optimizer, loss=tf.keras.losses.MeanSquaredError())
    filepath = "transfer_autoencoder_energy_model6/" \
               "transfer_autoencoder_energy_model6-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model
    callbacks_list = [checkpoint]
    history = test_model.fit(x=[X_lip_train, X_tongue_train], y=energy_train, batch_size=64, epochs=100,
                             callbacks=callbacks_list,
                             validation_data=([X_lip_test, X_tongue_test], energy_test))
