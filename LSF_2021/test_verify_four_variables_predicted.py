"""
This file is used to test and verify the four variables predicted by the image2lsf models , image2f0 models etc.
"""
import numpy as np
from matplotlib import pyplot as plt
import keras.models
import scipy.io as io
import os


def test_lsf_values_from_the_model(ch7_predicted_lsf, order):
    """
    This function plot the original lsf and the predicted lsf of certain order for the chapiter 7
    :param ch7_predicted_lsf: the matrix of the predicted lsf values
    :return:
    """
    original_lsf = np.load("../../LSF_data/lsp_all_chapiter.npy")  # (84679, 13)
    # choose the test data
    ch7_original_lsf = original_lsf[-15951:, :]
    x = np.arange(0, len(ch7_original_lsf))
    plt.figure()
    plt.plot(x, ch7_original_lsf[:, order-1], 'b', label='original lsf 13th coefficient')
    plt.plot(x, ch7_predicted_lsf[:, order-1], 'r', label='predicted lsf 13th coefficient')
    plt.legend()
    plt.grid()
    plt.show()


def plot_compare_f0_values(ch7_predicted_f0):
    # load the original F0 vector
    original_f0 = np.load("../../LSF_data/f0_all_chapiter.npy")
    # choose the f0 for chapiter 7
    ch7_original_f0 = original_f0[-15951:]
    x = np.arange(0, len(ch7_original_f0))
    plt.figure()
    plt.plot(x, ch7_original_f0[:], 'b', label='original f0 for chapiter 7')
    plt.plot(x, ch7_predicted_f0[:], 'r', label='predicted f0 for chapiter 7')
    plt.legend()
    plt.grid()
    plt.show()


def plot_compare_uv_values(ch7_predicted_uv):
    # load the original uv vector
    original_uv = np.load("../../LSF_data/uv_all_chapiter.npy")
    ch7_original_uv = original_uv[-15951:]
    x = np.arange(0, len(ch7_original_uv))
    plt.figure()
    plt.plot(x, ch7_original_uv[:], 'b', label='original uv for chapiter 7')
    plt.plot(x, ch7_predicted_uv[:], 'r', label='predicted uv for chapiter 7')
    plt.legend()
    plt.grid()
    plt.show()


def plot_compare_energy_values(ch7_predicted_energy):
    # load the original
    original_energy = np.load("../../LSF_data/energy_all_chapiters.npy")
    original_energy = np.matrix.transpose(original_energy)
    ch7_original_energy = original_energy[-15951:]
    print(len(ch7_original_energy))
    x = np.arange(0, len(ch7_original_energy))
    plt.figure()
    plt.plot(x, ch7_original_energy[:], 'b', label='original energy for chapiter 7')
    plt.plot(x, ch7_predicted_energy[:], 'r', label='predicted energy for chapiter 7')
    plt.legend()
    plt.grid()
    plt.show()


def do_prediction_image2f0_model(path_image2f0_model):
    image2f0_model = keras.models.load_model(path_image2f0_model)
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

    f0_chapiter7_predicted = image2f0_model.predict([X_lip_test, X_tongue_test])

    X_f0 = np.load("../../LSF_data/f0_all_chapiter.npy")
    max_f0 = np.max(X_f0)
    f0_chapiter7_predicted = f0_chapiter7_predicted * max_f0
    return f0_chapiter7_predicted


def do_prediction_image2energy_model(path_image2energy_model):
    image2energy_model = keras.models.load_model(path_image2energy_model)

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

    energy_chapiter7_predicted = image2energy_model.predict([X_lip_test, X_tongue_test])
    X_energy = np.load("../../LSF_data/energy_all_chapiters.npy")
    max_energy = np.max(X_energy)
    energy_chapiter7_predicted = energy_chapiter7_predicted * max_energy
    return energy_chapiter7_predicted


def do_prediction_image2lsf_model(path_image2lsf_model):
    """
    Predict the lsf matrix of the chapiter 7 by using the model with path_image2lsf_model
    :param path_image2lsf_model:
    :return:
    """
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


def do_prediction_image2uv_model(path_image2uv_model):
    image2uv_model = keras.models.load_model(path_image2uv_model)

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

    lsf_ch7_predicted = image2uv_model.predict([X_lip_test, X_tongue_test])
    return lsf_ch7_predicted


if __name__ == "__main__":

    path_image2lsf_model = "C:/Users/chaoy/Desktop/StageSilentSpeech/results/week_0705/" \
                           "models_used_for_the_reconstruction/image2lsf_model7_dr03-36-0.00491458.h5"
    lsf_npy = do_prediction_image2lsf_model(path_image2lsf_model)
    lsf_npy[:, 12] = 0
    io.savemat("lsf_predocted.mat", {'data': lsf_npy})
    
