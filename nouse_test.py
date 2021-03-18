import librosa
from tensorflow import keras
import numpy as np
import librosa.display
from matplotlib import pyplot as plt

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

    model = keras.models.load_model("../weights-improvement-200-0.00021.h5")
    test = x_test[0].reshape(1, 736)
    pred = model.predict(test)
    loss = np.square(test-pred).mean(axis=None)
    print(loss)