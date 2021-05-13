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
        # data_lips is file_path list which contains the file paths of lips images for one batch
        batch_x_lips = [i for i in data_lips[idx:(idx + 4 + self.batch_size)]]
        batch_x_tongues = [i for i in data_tongues[idx:(idx + 4 + self.batch_size)]]
        batch_y = label[idx+2:idx+self.batch_size+2, :]

        batch_image_lips = np.zeros((self.batch_size, 5, 64, 64, 1))
        for num_batch_lips in range(self.batch_size):
            # read five images
            for id_img_lips in range(5):

                lip_correspond = np.array(Image.open(batch_x_lips[num_batch_lips+id_img_lips]))
                batch_image_lips[num_batch_lips, id_img_lips, :, :, 0] = lip_correspond

        # same for the tongues images
        batch_image_tongues = np.zeros((self.batch_size, 5, 64, 64, 1))
        for num_batch_tongues in range(self.batch_size):
            # read five tongues images
            for id_img_tongues in range(5):
                tongue_correspond = np.array(Image.open(batch_x_tongues[num_batch_tongues+id_img_tongues]))
                batch_image_tongues[num_batch_tongues, id_img_tongues, :, :, 0] = tongue_correspond

        # normalisation
        batch_image_lips = batch_image_lips/255.0
        batch_image_tongues = batch_image_tongues/255.0

        return [batch_image_lips, batch_image_tongues], batch_y


def phase_model1():
    input_lips = Input(shape=(5, 64, 64, 1))
    lips_conv1_1 = TimeDistributed(Conv2D(16, (3, 3), padding="same", activation="relu"))(input_lips)
    lips_conv1_2 = TimeDistributed(Conv2D(16, (3, 3), padding="same", activation="relu"))(lips_conv1_1)
    lips_pooling1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(lips_conv1_2)
    lips_conv2_1 = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(lips_pooling1)
    lips_conv2_2 = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(lips_conv2_1)
    lips_pooling2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(lips_conv2_2)
    lips_conv3_1 = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(lips_pooling2)
    lips_conv3_2 = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(lips_conv3_1)
    lips_pooling3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(lips_conv3_2)
    lips_conv4_1 = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu"))(lips_pooling3)
    lips_conv4_2 = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu"))(lips_conv4_1)
    lips_pooling4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(lips_conv4_2)

    input_tongues = Input(shape=(5, 64, 64, 1))
    tongues_conv1_1 = TimeDistributed(Conv2D(16, (3, 3), padding="same", activation="relu"))(input_tongues)
    tongues_conv1_2 = TimeDistributed(Conv2D(16, (3, 3), padding="same", activation="relu"))(tongues_conv1_1)
    tongues_pooling1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(tongues_conv1_2)
    tongues_conv2_1 = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(tongues_pooling1)
    tongues_conv2_2 = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="relu"))(tongues_conv2_1)
    tongues_pooling2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(tongues_conv2_2)
    tongues_conv3_1 = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(tongues_pooling2)
    tongues_conv3_2 = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="relu"))(tongues_conv3_1)
    tongues_pooling3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(tongues_conv3_2)
    tongues_conv4_1 = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu"))(tongues_pooling3)
    tongues_conv4_2 = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="relu"))(tongues_conv4_1)
    tongues_pooling4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(tongues_conv4_2)

    cc = concatenate([lips_pooling4, tongues_pooling4])
    flat_layer = Flatten()(cc)
    fc1 = Dense(736, activation="linear")(flat_layer)
    mymodel = Model([input_lips, input_tongues], fc1)
    return mymodel


if __name__ == "__main__":
    # organize the training data

    train_lips_filepath_list = []
    # the number of images used for training
    nb_training_images = 68728
    for nb_train_lips in range(nb_training_images):
        train_lips_filepath_list.append("lips_6464_all_chapiter/%d.bmp" % nb_train_lips)

    train_tongues_filepath_list = []
    for nb_train_tongues in range(nb_training_images):
        train_tongues_filepath_list.append("tongues_6464_all_chapiter/%d.bmp"
                                           % nb_train_tongues)

    Y = np.load("phase_spectrogrammes_all_chapitre_corresponding.npy")
    Y = np.transpose(Y)  # changed to (84679,736)

    # spectrum of ch1-ch6
    training_labels = Y[0:nb_training_images, :]
########################################################################################################################
    # organize the validation data
    validation_lips = np.load("lips_validation_ch7.npy")
    validation_tongues = np.load("tongues_validation_ch7.npy")
    validation_lips /= 255.0
    validation_tongues /= 255.0
    validation_label = Y[-15949:-2, :]

########################################################################################################################
    # generate data
    batch_size = 64
    training_generator = DataGenerator(train_lips_filepath_list, train_tongues_filepath_list,
                                       training_labels, batch_size)

    # model
    model11 = phase_model1()
    model11.summary()
    my_optimizer = keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
    model11.compile(my_optimizer, loss=tf.keras.losses.MeanSquaredError())

    filepath = "phase_model1/phase_model1-{epoch:02d}-{val_loss:.8f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto')  # only save improved accuracy model

    callbacks_list = [checkpoint]
    history = model11.fit_generator(training_generator,
                                    steps_per_epoch=int(nb_training_images // batch_size),
                                    epochs=100,
                                    validation_data=([validation_lips, validation_tongues], validation_label),
                                    callbacks=callbacks_list,
                                    use_multiprocessing=False,
                                    shuffle=True)
