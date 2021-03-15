import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


if __name__ == "__main__":
    y, sr = librosa.load("../data/20200616_154520_RecFile_1_bruce_ch1/RecFile_1_20200616_154520_Sound_Capture_DShow_5_monoOutput1.wav", sr=44100)
    S = np.abs(librosa.stft(y, n_fft=735*2 , hop_length=735))
    print(S.shape)

    fig, ax = plt.subplots()

    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax)

    ax.set_title('Power spectrogram')

    fig.colorbar(img, ax=ax, format="%+2.0f dB")