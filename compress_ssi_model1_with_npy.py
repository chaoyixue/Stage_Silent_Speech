"""
This file use fit generator and use the 30 values of the neurons as the labels
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from keras.utils import Sequence
from sklearn.utils import shuffle
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, concatenate, Flatten, TimeDistributed, LeakyReLU,\
    Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow import keras


# Sequence subclass generators
class DataGenerator(Sequence):

    def __init__(self, input_sample_lips, input_sample_tongues, input_label, batch_size=64):
        self.input_sample_lips = input_sample_lips
        self.input_sample_tongues = input_sample_tongues
        self.input_label = input_label
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.input_sample_lips))

    def __len__(self):
        return int(len(self.input_sample_lips) - self.batch_size - 2)

    def __getitem__(self, idx):
        data_lips = self.input_sample_lips
        data_tongues = self.input_sample_tongues
        label = self.input_label
        # data_lips is the images npy and the batch_x_lips contains the lips images needed for each batch
        # so are the data_tongues and the  batch_x_tongues
        batch_x_lips = [i for i in data_lips[idx:(idx + 4 + self.batch_size)]]
        batch_x_tongues = [i for i in data_tongues[idx:(idx + 4 + self.batch_size)]]
        batch_y = label[idx+2:idx+self.batch_size+2, :]

        batch_image_lips = np.zeros((self.batch_size, 5, 64, 64, 1))
        for num_batch_lips in range(self.batch_size):
            # read five images
            for id_img_lips in range(5):

                lip_correspond = batch_x_lips[num_batch_lips+id_img_lips]
                batch_image_lips[num_batch_lips, id_img_lips, :, :, 0] = lip_correspond

        # same for the tongues images
        batch_image_tongues = np.zeros((self.batch_size, 5, 64, 64, 1))
        for num_batch_tongues in range(self.batch_size):
            # read five tongues images
            for id_img_tongues in range(5):
                tongue_correspond = batch_x_tongues[num_batch_tongues+id_img_tongues]
                batch_image_tongues[num_batch_tongues, id_img_tongues, :, :, 0] = tongue_correspond

        return [batch_image_lips, batch_image_tongues], batch_y


def compress_ssi_model1():
    input_lips = Input(shape=(5, 64, 64, 1))
    lips_conv1 = Conv2D(16, (5, 5), padding="same", activation="relu")
    lips_tm1 = TimeDistributed(lips_conv1)(input_lips)
    lips_pooling1 = TimeDistributed(AveragePooling2D())(lips_tm1)
    lips_conv2 = Conv2D(32, (5, 5), padding="valid", activation="relu")
    lips_tm2 = TimeDistributed(lips_conv2)(lips_pooling1)
    lips_pooling2 = TimeDistributed(AveragePooling2D())(lips_tm2)

    input_tongues = Input(shape=(5, 64, 64, 1))
    tongues_conv1 = Conv2D(16, (5, 5), padding="same", activation="relu")
    tongues_tm1 = TimeDistributed(tongues_conv1)(input_tongues)
    tongues_pooling1 = TimeDistributed(AveragePooling2D())(tongues_tm1)
    tongues_conv2 = Conv2D(32, (5, 5), padding="valid", activation="relu")
    tongues_tm2 = TimeDistributed(tongues_conv2)(tongues_pooling1)
    tongues_pooling2 = TimeDistributed(AveragePooling2D())(tongues_tm2)

    cc = concatenate([lips_pooling2, tongues_pooling2])
    flat_layer = Flatten()(cc)
    fc1 = Dense(30, activation="relu")(flat_layer)

    mymodel = Model([input_lips, input_tongues], fc1)
    return mymodel


if __name__ == "__main__":
    # organize the training data

    X_lips = np.load("../data_npy_one_image/lips_all_chapiters.npy")
    X_tongues = np.load("../data_npy_one_image/tongues_all_chapiters.npy")
    nb_training_images = 68728
    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    # images of ch1-ch6
    train_lips = X_lips[:nb_training_images, :, :, 0]
    train_tongues = X_lips[:nb_training_images, :, :, 0]

    training_labels = np.load("../labels_generated_autoencoder_30values/training_labels_30_neurons.npy")
    max_spectrum = np.max(training_labels)
    training_labels = training_labels / max_spectrum
########################################################################################################################
    # organize the validation data
    validation_lips = np.load("../validation_data/lips_validation_ch7.npy")
    validation_tongues = np.load("../validation_data/tongues_validation_ch7.npy")
    validation_lips /= 255.0
    validation_tongues /= 255.0
    validation_label = np.load("../labels_generated_autoencoder_30values/validation_labels_30_neurons.npy")
    validation_label /= max_spectrum
    validation_label = validation_label[2:-2]

########################################################################################################################
    # generate data
    my_batch_size = 64
    training_generator = DataGenerator(train_lips, train_tongues,
                                       training_labels, my_batch_size)

    # model
    model13 = compress_ssi_model1()
    model13.summary()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    model13.compile(my_optimizer, loss=tf.keras.losses.MeanSquaredError())

    filepath = "../compress_ssi_model1_relu_bs64/compress_ssi_model1_bs64-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = model13.fit_generator(training_generator,
                                    steps_per_epoch=int(nb_training_images // my_batch_size),
                                    epochs=50,
                                    validation_data=([validation_lips, validation_tongues], validation_label),
                                    callbacks=callbacks_list,
                                    use_multiprocessing=False,
                                    shuffle=True)





