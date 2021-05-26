"""
This file is used to convert wav files to spectrograms then concatenate and save them
into a npy file
"""
import numpy as np
import librosa


def convert_wav_to_spectrum(file_path, sample_rate=44100, nfft=735*2, window_length=735*2, hop_length=735):
    """
    :param file_path:
    :param sample_rate:
    :param nfft:
    :param window_length:
    :param hop_length:
    :return:
    """
    y, sr = librosa.load(file_path, sr=sample_rate)  # load the wav file with the sample rate chosen
    spectrogram_y = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop_length, win_length=window_length))
    # shape : (736, N)
    return spectrogram_y


if __name__ == "__main__":
    filepath_ch1 = "../wav_files/chapiter1.wav"
    spect_ch1 = convert_wav_to_spectrum(filepath_ch1)
    spect_ch1 = spect_ch1[:, :10054]
    filepath_ch2 = "../data/20200617_150421_RecFile_1_bruce_ch2" \
                   "/RecFile_1_20200617_150421_Sound_Capture_DShow_5_monoOutput1.wav"
    spect_ch2 = convert_wav_to_spectrum(filepath_ch2)
    spect_ch2 = spect_ch2[:, :14441]
    filepath_ch3 = "../data/20200617_151112_RecFile_1_bruce_ch3" \
                   "/RecFile_1_20200617_151112_Sound_Capture_DShow_5_monoOutput1.wav"
    spect_ch3 = convert_wav_to_spectrum(filepath_ch3)
    spect_ch3 = spect_ch3[:, :8885]

    filepath_ch4 = "../data/20200617_151519_RecFile_1_bruce_ch4" \
                   "/RecFile_1_20200617_151519_Sound_Capture_DShow_5_monoOutput1.wav"
    spect_ch4 = convert_wav_to_spectrum(filepath_ch4)
    spect_ch4 = spect_ch4[:, :15621]

    filepath_ch5 = "../data/20200617_152851_RecFile_1_bruce_ch5" \
                   "/RecFile_1_20200617_152851_Sound_Capture_DShow_5_monoOutput1.wav"
    spect_ch5 = convert_wav_to_spectrum(filepath_ch5)
    spect_ch5 = spect_ch5[:, :14553]

    filepath_ch6 = "../data/20200617_153450_RecFile_1_bruce_ch6" \
                   "/RecFile_1_20200617_153450_Sound_Capture_DShow_5_monoOutput1.wav"
    spect_ch6 = convert_wav_to_spectrum(filepath_ch6)
    spect_ch6 = spect_ch6[:, :5174]

    filepath_ch7 = "../data/20200617_153719_RecFile_1_bruce_ch7" \
                   "/RecFile_1_20200617_153719_Sound_Capture_DShow_5_monoOutput1.wav"
    spect_ch7 = convert_wav_to_spectrum(filepath_ch7)
    spect_ch7 = spect_ch7[:, :15951]

    result = np.concatenate((spect_ch1, spect_ch2, spect_ch3, spect_ch4, spect_ch5, spect_ch6, spect_ch7), axis=1)
    print(result.shape)
    # axis 0 is the frequency axis corresponding to 736 bins of frequency. axis 1 is the time
    # axis corresponding to seconds
    np.save("spectrogrammes_all_chapitre_corresponding.npy", result)
