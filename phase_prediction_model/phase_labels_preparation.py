import numpy as np
import librosa


def convert_wav_to_phase_spectrum(file_path, sample_rate=44100, nfft=735*2, window_length=735*2, hop_length=735):
    """
    :param file_path:
    :param sample_rate:
    :param nfft:
    :param window_length:
    :param hop_length:
    :return:
    """
    y, sr = librosa.load(file_path, sr=sample_rate)  # load the wav file with the sample rate chosen
    # calculate the phase spectrum in radians
    phase_y = np.angle(librosa.stft(y, n_fft=nfft, hop_length=hop_length, win_length=window_length))
    # shape : (736, N)
    return phase_y


if __name__ == "__main__":
    filepath_ch1 = "../../wav_files/chapiter1.wav"
    spect_ch1 = convert_wav_to_phase_spectrum(filepath_ch1)
    spect_ch1 = spect_ch1[:, :10054]
    filepath_ch2 = "../../wav_files/chapiter2.wav"
    spect_ch2 = convert_wav_to_phase_spectrum(filepath_ch2)
    spect_ch2 = spect_ch2[:, :14441]
    filepath_ch3 = "../../wav_files/chapiter3.wav"
    spect_ch3 = convert_wav_to_phase_spectrum(filepath_ch3)
    spect_ch3 = spect_ch3[:, :8885]

    filepath_ch4 = "../../wav_files/chapiter4.wav"
    spect_ch4 = convert_wav_to_phase_spectrum(filepath_ch4)
    spect_ch4 = spect_ch4[:, :15621]

    filepath_ch5 = "../../wav_files/chapiter5.wav"
    spect_ch5 = convert_wav_to_phase_spectrum(filepath_ch5)
    spect_ch5 = spect_ch5[:, :14553]

    filepath_ch6 = "../../wav_files/chapiter6.wav"
    spect_ch6 = convert_wav_to_phase_spectrum(filepath_ch6)
    spect_ch6 = spect_ch6[:, :5174]

    filepath_ch7 = "../../wav_files/chapiter7.wav"
    spect_ch7 = convert_wav_to_phase_spectrum(filepath_ch7)
    spect_ch7 = spect_ch7[:, :15951]

    result = np.concatenate((spect_ch1, spect_ch2, spect_ch3, spect_ch4, spect_ch5, spect_ch6, spect_ch7), axis=1)
    print(result.shape)
    # axis 0 is the frequency axis corresponding to 736 bins of frequency. axis 1 is the time
    # axis corresponding to seconds
    np.save("phase_spectrogrammes_all_chapitre_corresponding.npy", result)
