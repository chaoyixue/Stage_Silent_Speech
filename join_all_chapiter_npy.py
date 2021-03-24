"""
This file is used to join all the npy files of seperated chapiters into one npy file which contains all the data of
7 chapiters
"""

import numpy as np

if __name__ == "__main__":
    tongue_ch1 = np.load("tongue_ch1.npy")
    tongue_ch2 = np.load("tongue_ch2.npy")
    tongue_ch3 = np.load("tongue_ch3.npy")
    tongue_ch4 = np.load("tongue_ch4.npy")
    tongue_ch5 = np.load("tongue_ch5.npy")
    tongue_ch6 = np.load("tongue_ch6.npy")
    tongue_ch7 = np.load("tongue_ch7.npy")

    whole_ch = np.concatenate((tongue_ch1, tongue_ch2, tongue_ch3, tongue_ch4, tongue_ch5, tongue_ch6, tongue_ch7), axis=0)
    print(whole_ch.shape)
    whole_ch = whole_ch.reshape((whole_ch.shape[0], whole_ch.shape[1], whole_ch.shape[2], 1))
    print(whole_ch.shape)
    np.save("tongues_all_chapiters.npy", whole_ch)

