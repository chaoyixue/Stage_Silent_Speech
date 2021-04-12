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
    X_lips = np.load("../five_recurrent_image_npy/lips/lips_recurrent_5images_all_chapitres.npy")
    X_tongues = np.load("../five_recurrent_image_npy/tongues/tongues_recurrent_5images_all_chapitres.npy")
    Y = np.load("../five_recurrent_image_npy/spectrum_recurrent_all.npy")

    # normalisation
    X_lips = X_lips / 255.0
    X_tongues = X_tongues / 255.0
    max_spectrum = np.max(Y)
    Y = Y / max_spectrum

    # split train set data
    lips_x_train = X_lips[:-15947]
    lips_x_test = X_lips[-15947:]
    tongues_x_train = X_tongues[:-15947]
    tongues_x_test = X_tongues[-15947:]
    y_train = Y[:, :-15947]
    y_test = Y[:, -15947:]
    y_train = np.transpose(y_train)
    y_test = np.transpose(y_test)

    model = keras.models.load_model("../ssi_model8-08-0.00005231.h5")
    model.summary()
    test_result = model.predict([lips_x_test, tongues_x_test])
    result = np.transpose(test_result)
    print(result.shape)

    # denormalisation
    result = result * max_spectrum

    """
    fig, ax = plt.subplots(nrows=2)
    img = librosa.display.specshow(librosa.amplitude_to_db(np.transpose(y_test),
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title('original spectrum')
    librosa.display.specshow(librosa.amplitude_to_db(result, ref=np.max), sr=44100, hop_length=735,
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum learned')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    """

    # reconstruction wav file
    test_reconstruit = librosa.griffinlim(result, n_iter=128, hop_length=735, win_length=735 * 2)
    sf.write("ch7_reccurent_5_images.wav", test_reconstruit, 44100)
