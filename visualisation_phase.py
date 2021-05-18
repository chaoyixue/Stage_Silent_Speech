from matplotlib import  pyplot as plt
import numpy as np

if __name__ == "__main__":
    test = np.load(
        "C:/Users/chaoy/Desktop/StageSilentSpeech/phase_spectrogram/phase_spectrogrammes_all_chapitre_corresponding.npy")
    plt.figure()
    plt.imshow(test, cmap="hot")
    plt.colorbar()
    plt.show()