import librosa
from tensorflow import keras
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import matplotlib.image as mpimg

if __name__ == "__main__":
    test = np.load("../data_npy_one_image/tongues_all_chapiters.npy")
    print(test.shape)
    plt.figure()
    for i in tqdm(range(len(test))):
        plt.imshow(test[i,:, :, 0], cmap="gray")
        plt.pause(0.001)
