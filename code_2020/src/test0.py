import torch
print(torch.__version__)
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
from scipy import signal

# print('[INFO] generate wave')
# ref_db=0
# max_db=80
# test_label = np.load('../data/database/0521_13_256pp_test_label.npy')
# test_label=test_label[:500,:]
# print(test_label.shape)
# print('[INFO] save spectrogram picture')
# plt.figure(figsize=(15, 6))
# librosa.display.specshow(test_label.T,sr =44100,hop_length=735, x_axis='s', y_axis='mel')
# plt.title('mel spectrogram (test_label)')
# plt.savefig('../out/ppn13_melspectrogrammme.png')
# print(test_label.shape)

# test_label= (np.clip(test_label, 1e-8, 1) * max_db) - max_db + ref_db # de-norm

# print('[INFO] save spectrogram picture')
# plt.figure(figsize=(15, 6))
# librosa.display.specshow(test_label.T,sr =44100,hop_length=735, x_axis='s', y_axis='mel')
# plt.title('mel spectrogram (test_label)')
# plt.savefig('../out/ppn13_melspectrogrammme0.png')

# # ref_melspec=np.load('./ssi/data/database/0525_13pp7_256ref.npy')
# # print(ref_melspec)
# s1 = librosa.db_to_power(test_label.T, ref=805.06226)
# _wavlabel = librosa.feature.inverse.mel_to_audio(s1,sr=44100,n_fft=735*13, hop_length=735, power=2,n_iter=50)
# _wavlabel = signal.lfilter([1], [1, -0.95], _wavlabel)

# plt.figure(figsize=(15, 7))
# librosa.display.waveplot(_wavlabel, sr=44100,x_axis='s')
# plt.title('test label')
# plt.savefig('../out/ppn13_wav_powmel.png')
# librosa.output.write_wav("../out/ppn13_original_test.wav", _wavlabel, sr=44100)



