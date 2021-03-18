"""
This program is used to find out if the error of the prediction is due to the predict function
in test_model.py we passed the whole spectrum to predict, in this program, we passed each time
one line of the spectrum then concatenate them.
But there is no difference between these two methods. So the spectrum not well generated is caused
by the model's underfitting but not caused by a bug when testing.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import librosa
import soundfile as sf
import librosa.display
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    # load data
    X = np.load("spectrogrammes_all_chapitre.npy")
    max_value = np.max(X)
    print(max_value)

    # normalisation
    X = X / max_value
    print(X.max())
    print(X.min())

    # split train test data
    x_train = np.transpose(X[:, :84776 - 15951])
    x_test = np.transpose(X[:, -15951:])
    print(x_train.shape)
    print(x_test.shape)
    model = keras.models.load_model("../weights-improvement-200-0.00011.h5")
    model.summary()

    for i in tqdm(range(x_test.shape[0])):
        new_time = x_test[i, :].reshape((1, 736))
        new_result = model.predict(new_time)
        if i == 0:
            result = new_result
        else:
            result = np.concatenate((result, new_result), axis=0)

    result = np.transpose(result)
    # show the spectrum original and the spectrum learned
    fig, ax = plt.subplots(nrows=2)
    x_test = np.transpose(x_test)  # x_test = 736 * N
    img = librosa.display.specshow(librosa.amplitude_to_db(x_test,
                                                           ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('Power spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(result, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()