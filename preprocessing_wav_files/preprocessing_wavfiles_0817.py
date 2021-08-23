import numpy as np
import librosa
import soundfile as sf


def couper_echantillons_son_au_nombre_images(fichier_origine, nb_images):
    nb_echantillons_origine = len(fichier_origine)
    len_fenetre = nb_echantillons_origine / nb_images
    index_k = []
    for k in range(nb_images):
        index_k.append(np.floor(len_fenetre * k))
    index_k_npy = np.array(index_k, dtype=int)
    chapitre_coupe = np.zeros(nb_images*735)
    n = 0
    for index in index_k_npy:
        chapitre_coupe[735 * n:735 * n + 735] = fichier_origine[index:index + 735]
        n += 1
    return chapitre_coupe


def convert_wav_to_spectrum(wavfile, nfft=735*2, window_length=735*2, hop_length=735):
    """
    :param nfft:
    :param window_length:
    :param hop_length:
    :return:
    """

    spectrogram_y = np.abs(librosa.stft(wavfile, n_fft=nfft, hop_length=hop_length, win_length=window_length))
    # shape : (736, N)
    return spectrogram_y


if __name__ == "__main__":
    # couper les chapitres pour mettre en correspondance avec le nombre d'images
    chapitre1, _ = librosa.load("../../wav_files/chapiter1.wav", sr=44100)
    ch1_coupe = couper_echantillons_son_au_nombre_images(chapitre1, 10054)

    chapitre2, _ = librosa.load("../../wav_files/chapiter2.wav", sr=44100)
    ch2_coupe = couper_echantillons_son_au_nombre_images(chapitre2, 14441)

    chapitre3, _ = librosa.load("../../wav_files/chapiter3.wav", sr=44100)
    ch3_coupe = couper_echantillons_son_au_nombre_images(chapitre3, 8885)

    chapitre4, _ = librosa.load("../../wav_files/chapiter4.wav", sr=44100)
    ch4_coupe = couper_echantillons_son_au_nombre_images(chapitre4, 15621)

    chapitre5, _ = librosa.load("../../wav_files/chapiter5.wav", sr=44100)
    ch5_coupe = couper_echantillons_son_au_nombre_images(chapitre5, 14553)

    chapitre6, _ = librosa.load("../../wav_files/chapiter6.wav", sr=44100)
    ch6_coupe = couper_echantillons_son_au_nombre_images(chapitre6, 5174)

    chapitre7, _ = librosa.load("../../wav_files/chapiter7.wav", sr=44100)
    ch7_coupe = couper_echantillons_son_au_nombre_images(chapitre7, 15951)

    # calculation des spectrogrammes
    spect_ch1 = convert_wav_to_spectrum(ch1_coupe)[:, :-1]
    spect_ch2 = convert_wav_to_spectrum(ch2_coupe)[:, :-1]
    spect_ch3 = convert_wav_to_spectrum(ch3_coupe)[:, :-1]
    spect_ch4 = convert_wav_to_spectrum(ch4_coupe)[:, :-1]
    spect_ch5 = convert_wav_to_spectrum(ch5_coupe)[:, :-1]
    spect_ch6 = convert_wav_to_spectrum(ch6_coupe)[:, :-1]
    spect_ch7 = convert_wav_to_spectrum(ch7_coupe)[:, :-1]

    result = np.concatenate((spect_ch1, spect_ch2, spect_ch3, spect_ch4, spect_ch5, spect_ch6, spect_ch7), axis=1)
    result = np.transpose(result)
    print(result.shape)
    validation_spectrum = np.transpose(spect_ch7)
    np.save("validation_spectrum_coupe.npy", validation_spectrum)
    