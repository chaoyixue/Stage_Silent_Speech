"""
This file is used to test and verify the four variables predicted by the image2lsf models , image2f0 models etc.
"""
import numpy as np
from matplotlib import pyplot as plt
import keras.models


def test_lsf_values_from_the_model(ch7_predicted_lsf):
    original_lsf = np.load("../../LSF_data/lsp_all_chapiter.npy")  # (84679, 13)
    # choose the test data
    ch7_original_lsf = original_lsf[-15951:, :]
    x = np.arange(0, len(ch7_original_lsf))
    plt.figure()
    plt.plot(x, ch7_original_lsf[:, 12], 'b', label='original lsf 13th coefficient')
    plt.plot(x, ch7_predicted_lsf[:, 12], 'r', label='predicted lsf 13th coefficient')
    plt.legend()
    plt.grid()
    plt.show()


def do_prediction_image2lsf_model(path_image2lsf_model):
    image2lsf_model = keras.models.load_model(path_image2lsf_model)

    # load images inputs
    X_lip = np.load("../../data_npy_one_image/lips_all_chapiters.npy")
    X_tongue = np.load("../../data_npy_one_image/tongues_all_chapiters.npy")

    # normalisation
    X_lip = X_lip / 255.0
    X_tongue = X_tongue / 255.0

    # split the testing part(ch7)
    nb_image_chapiter7 = 15951
    X_lip_test = X_lip[-nb_image_chapiter7:, :]
    X_tongue_test = X_tongue[-nb_image_chapiter7:, :]

    lsf_ch7_predicted = image2lsf_model.predict([X_lip_test, X_tongue_test])
    return lsf_ch7_predicted


if __name__ == "__main__":
    path_image2lsf_model7 = "C:/Users/chaoy/Desktop/" \
                            "StageSilentSpeech/results/week_0705/image2lsf_model7_dr03-36-0.00491458.h5"
    ch7_lsf_predicted = do_prediction_image2lsf_model(path_image2lsf_model7)
    test_lsf_values_from_the_model(ch7_lsf_predicted)
