import librosa
from tensorflow import keras
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import matplotlib.image as mpimg

if __name__ == "__main__":
    test = np.load("../five_recurrent_image_npy/tongues/tongues_recurrent_5images_all_chapitres.npy")
    print(test.shape)
    plt.figure()
    plt.imshow(test[1000, 1, :, :, 0], cmap="gray")
    plt.show()