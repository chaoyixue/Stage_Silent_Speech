import librosa
import librosa.display
import numpy as np
import soundfile as sf
from pydub import AudioSegment


# parole : audio preprocessing


def resample(dir_path):
    """
    resampling all the wav files in the directory
    :param dir_path: the path of wav files that contains all the wav files named after 1.wav 2.wav etc.
    :return:
    """
    ori_sr = 44100
    target_sr = 22050
    for num in range(1, 7 + 1):
        try:
            y, sr = librosa.load(dir_path + "%d.wav" % num, sr=ori_sr)  # y wav, sr sample rate
            print(y.shape)
            y_rs = librosa.core.resample(y, orig_sr=ori_sr, target_sr=target_sr)  # resampling
            print(y_rs.shape)
            sf.write(dir_path + "%d_22050.wav" % num, y_rs, target_sr)
        except:
            print(num, "error")


def concat_wav4(dir_path):
    for i in range(1, 7 + 1):
        wav = AudioSegment.from_wav(dir_path + 'out%d.wav' % i)
        if i == 1:
            outwav = wav
        else:
            outwav = outwav + wav
    outwav.export(dir_path + '0819out_p4.wav', format='wav')


def samp_audio4(dir_path):
    frames = [10054, 14441, 8885, 15621, 14553, 5174, 15951]  # parole4
    # data_drop = [-1004, -1087, -1692, -392, -1952, -607, -720]    # -1952 calcule LSF
    # frames = [9233, 13644, 8456, 14003, 13905, 4362, 13977, 13883, 8266, 23314, 6132, 3259, 15489] # parole5
    for i in range(1, 7 + 1):
        file = dir_path + '%d.wav' % i
        _y, sr = librosa.load(file, sr=None)
        print(_y.shape)
        # drop=data_drop[i-1]
        # _y=_y[:drop]
        # print(_y.shape)
        frame = frames[i - 1]
        base_fs = len(_y) / frame
        print(base_fs)
        hop = 735
        for j in range(0, frame):
            if j == 0:
                y = _y[int(base_fs * j):int(base_fs * j) + hop]
            else:
                y = np.concatenate((y, _y[int(base_fs * j):int(base_fs * j) + hop]), axis=0)
        print(y.shape)
        librosa.output.write_wav(dir_path + 'out%s.wav' % i, y, sr)


if __name__ == '__main__':
    dir_path = './ssi/data/new_data/new_audio/'
    # resample(dir_path)
    samp_audio4(dir_path)
    concat_wav4(dir_path)
