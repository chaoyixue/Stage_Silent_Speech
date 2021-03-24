import librosa
from tensorflow import keras
import numpy as np
import librosa.display
from matplotlib import pyplot as plt
from tqdm import tqdm
if __name__ == "__main__":
    test = np.load("spectrogrammes_all_chapitre.npy")
    print(test.shape)
