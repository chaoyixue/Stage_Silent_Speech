from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import librosa
import soundfile as sf
import librosa.display
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # load data
    X_lips = np.load("../validation_data/lips_validation_ch7.npy")
    X_tongues = np.load("../validation_data/tongues_validation_ch7.npy")
    Y = np.load("../validation_data/spectrum_validation.npy")

    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_spectrum = np.max(Y)
    Y = Y / max_spectrum

    y_test = np.transpose(Y)

    model = keras.models.load_model("../results/ssi_model12-23-0.00004926.h5")
    model.summary()
    test_result = model.predict([X_lips, X_tongues])

    result = np.transpose(test_result)
    print(result.shape)

    # denormalisation
    result = result * max_spectrum
    fig, ax = plt.subplots(nrows=2)
    img = librosa.display.specshow(librosa.amplitude_to_db(np.transpose(y_test),
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('original spectrum')
    librosa.display.specshow(librosa.amplitude_to_db(result, ref=np.max), sr=44100, hop_length=735,
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # reconstruction wav file
    test_reconstruit = librosa.griffinlim(result, n_iter=128, hop_length=735, win_length=735 * 2)
    sf.write("ch7_0421_model12_4926e5.wav", test_reconstruit, 44100)

    plt.show()
