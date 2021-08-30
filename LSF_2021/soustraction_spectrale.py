import librosa
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import librosa.display
if __name__ == "__main__":
    y, _ = librosa.load("../../results/week_0816/ch7_0816_data_coupe.wav", sr=44100)
    module = np.abs(librosa.stft(y, n_fft=735*2, hop_length=735, win_length=735*2))
    phase = np.angle(librosa.stft(y, n_fft=735*2, hop_length=735, win_length=735*2))
    # calculate the mean value of the first tenth frames
    spectral_bruit = np.nanmean(module[:, :10], axis=1)
    new_module = np.zeros(module.shape)
    for i in range(module.shape[1]):
        new_module[:, i] = module[:, i] - spectral_bruit
    tf_reconstruit = new_module * np.exp(phase*1j)
    yvect = librosa.istft(tf_reconstruit, hop_length=735, win_length=735*2)
    sf.write("ch7_0816_data_coupe_soustrait_spectrale.wav", yvect, samplerate=44100)
    fig, ax = plt.subplots(nrows=2)
    img = librosa.display.specshow(librosa.amplitude_to_db(module,
                                                           ref=np.max), sr=44100, hop_length=735,
                                   y_axis='linear', x_axis='time', ax=ax[0])
    ax[0].set_title('original spectrum')
    librosa.display.specshow(librosa.amplitude_to_db(new_module, ref=np.max), sr=44100, hop_length=735,
                             y_axis='linear', x_axis='time', ax=ax[1])
    ax[1].set_title('spectrum filtre')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
