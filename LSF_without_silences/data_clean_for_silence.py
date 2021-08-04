import numpy as np
import tensorflow as tf


def find_nonsilence_data(original_energy_all_chapiter):
    """
    This function is used to find out where correpsonding the silences from energy file
    :return:
    """
    # the data for all the wave files of which the energy is greater than 0.005
    index_speech_part = np.flatnonzero(original_energy_all_chapiter > 0.005)
    return index_speech_part


def find_silence_data(original_energy_all_chapiter):
    # the data for all the wave files of which the energy is greater than 0.005
    index_silence_part = np.flatnonzero(original_energy_all_chapiter <= 0.005)
    return index_silence_part


def pick_data(alldata_npy, index):
    data_one_part = alldata_npy[index]
    return data_one_part


def verify_MSE_lsf(lsf_original, lsf_predicted_ch7, index_speech_chapiter_7, index_silence_chapiter7,
                   nombre_frames_test=15951):
    # load the original lsf for all chapiter and pick up the parts for chapiter 7
    original_lsf_chapiter7 = lsf_original[-nombre_frames_test:]
    # load the lsf predicted for chapiter 7
    mse = tf.keras.losses.MeanSquaredError()
    # calculate the mse of the totalite ch7
    mse_totalite_ch7 = mse(original_lsf_chapiter7, lsf_predicted_ch7).numpy()
    # split the vocal parts and the silences for the original ch7 chapiter and the predicted ch7 chapiter
    original_vocal_part_lsf_ch7 = pick_data(lsf_original, index_speech_chapiter_7)
    predicted_vocal_part_lsf_ch7 = pick_data(lsf_predicted_ch7, index_speech_chapiter_7)
    # mse for vocal part
    mse_vocal_part_ch7 = mse(original_vocal_part_lsf_ch7, predicted_vocal_part_lsf_ch7).numpy()
    original_silence_part_lsf_ch7 = pick_data(lsf_original, index_silence_chapiter7)
    predicted_silence_part_lsf_ch7 = pick_data(lsf_predicted_ch7, index_silence_chapiter7)
    # mse for silence parts
    mse_silence_part_ch7 = mse(original_silence_part_lsf_ch7, predicted_silence_part_lsf_ch7).numpy()

    return mse_totalite_ch7, mse_vocal_part_ch7, mse_silence_part_ch7


if __name__ == "__main__":
    """
    original_lsf = np.load("../../LSF_data/lsp_all_chapiter.npy")
    predicted_ch7_lsf = np.load("lsf_predit_image2lsf_model7_dr03.npy")
    original_energy_all = np.load("../../LSF_data/energy_all_chapiters.npy")
    index_nonsilence_allchapiter = find_nonsilence_data(original_energy_all)
    index_nonsilence_chapiter_7 = index_nonsilence_allchapiter[index_nonsilence_allchapiter >= 84679-15951]-68728
    index_silence_allchapiter = find_silence_data(original_energy_all)
    index_silence_chapiter7 = index_silence_allchapiter[index_silence_allchapiter >= 84679-15951]-68728
    verify_MSE_lsf(original_lsf, predicted_ch7_lsf, index_nonsilence_chapiter_7, index_silence_chapiter7)
    np.save("index_non_silence_chapiter_7.npy", index_nonsilence_chapiter_7)
    np.save("index_silence_chapiter_7.npy", index_silence_chapiter7)
    print("aaa")
    """

    original_energy_all = np.load("../../LSF_data/energy_all_chapiters.npy")
    index_nonsilence_allchapiter = find_nonsilence_data(original_energy_all)
    spectrum_all_chapiter = np.load("../../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    spectrum_all_chapiter = np.matrix.transpose(spectrum_all_chapiter)  # (84679, 736)
    spectrum_non_silence = pick_data(spectrum_all_chapiter, index_nonsilence_allchapiter)
    print(spectrum_non_silence.shape)
    np.save("spectrum_nonsilence_all_chapiter.npy", spectrum_non_silence)

