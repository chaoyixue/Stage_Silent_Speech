from ssi4_input_data import *
from ssi6_train0507_2 import test_model, CNN
# from ssi6_train_gpu import CNN
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import cv2
import numpy as np 
import librosa
import librosa.display
import sys
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from scipy import signal
BATCH_SIZE = 100
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
root='../out/0504'  

def match_image_label(image_data):
    l=image_data.shape[0]
    image_match=[]
    for i in range(l-1):
        image_con = []
        image_con.append(image_data[i])
        image_con.append(image_data[i+1])
        image_match.append(image_con)
    image_match = np.array(image_match)

    return image_match

def match_image_label2(image_data):
    image_data=np.expand_dims(image_data, axis=1)
    l=image_data.shape[0]
    image_match=[]
    for i in range(l-1):
        image_con = np.concatenate((image_data[i],image_data[i+1]),axis=-1)
        image_match.append(image_con)
    image_match = np.array(image_match)

    return image_match

def TestDatasets():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    _, test_lips, _, test_tongue, _, test_label = load_dataset()
    test_lips = test_lips[:1001,:,:,:]
    test_tongue = test_tongue[:1001,:,:,:]
    test_label = test_label[:1000,:]
    # train_lips, _, train_tongue, _, train_label, _ = load_dataset()
    # test_lips = train_lips[:6816,:,:,:]
    # test_tongue = train_tongue[:6816,:,:,:]
    # test_label = train_label[:6815,:]

    # #preprocessing
    test_lips = match_image_label(test_lips)
    test_tongue = match_image_label(test_tongue)
    
    #to torch.tensor
    test_lips = torch.from_numpy(test_lips).float()
    test_tongue = torch.from_numpy(test_tongue).float()
    test_label = torch.from_numpy(test_label).float()

    #change dimension (x,64,64,1) --> (x,1,64,64)
    test_lips = test_lips.permute(0,4,1,2,3)
    test_tongue = test_tongue.permute(0,4,1,2,3)

    #两通道
    test_datasets = TensorDataset(test_lips, test_tongue, test_label)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets, test_loader

loss_func = nn.MSELoss() 

if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    model=CNN()
    # model.load_state_dict(torch.load('./ssi/picture/0428checkpoint.pt'))
    # model.eval()
    test_datasets, test_loader = TestDatasets()
    print('[INFO] begin test')
    prediction = test_model(test_datasets, test_loader)
    print('[INFO] save test output')
    spec = prediction.detach().numpy()  #(20451, 1025)
    # np.save(root+"test_predict.npy", spec)

    # spec = np.load(root+'test_predict_train.npy') #(6815, 64)
    test_label = np.load('./ssi/data/database/test_label.npy')
    test_label=test_label[:1000,:]
    print(spec.shape)
    print(test_label.shape)

    print('[INFO] save spectrogram picture')
    plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1)
    librosa.display.specshow(spec.T,sr =44100, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('mel spectrogram (predict)')
    # plt.title('Linear-frequency power spectrogram (predict)')
    plt.subplot(2,1,2)
    librosa.display.specshow(test_label.T,sr =44100, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    plt.title('mel spectrogram (test_label)')
    plt.savefig(root+'_129spectrogram_predict_test.png')

    print('[INFO] generate wave')
    ref_db=1e-1
    max_db=80
    
    spec = (np.clip(spec, 0, 1) * max_db) - max_db + ref_db #把小于0的都改为0，大于1的都改为1
    s = librosa.db_to_amplitude(spec.T) #(64, 6815)
    _wav = librosa.feature.inverse.mel_to_audio(s,sr=44100,n_fft=2048, hop_length=735, win_length =1470, power=1.2,n_iter=50)

    test_label = (np.clip(test_label, 0, 1) * max_db) - max_db + ref_db #把小于0的都改为0，大于1的都改为1
    s1 = librosa.db_to_amplitude(test_label.T)
    _wavlabel = librosa.feature.inverse.mel_to_audio(s1,sr=44100,n_fft=2048, hop_length=735, win_length =1470, power=1.2,n_iter=50)
    plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1)
    librosa.display.waveplot(_wav, sr=44100,x_axis= 'time')
    # ipd.Audio(_wav, rate=sr)
    plt.title('from predict images')
    plt.subplot(2,1,2)
    librosa.display.waveplot(_wavlabel, sr=44100,x_axis='time')
    plt.title('from train_label')
    plt.savefig(root+'_129wav_reconstruit_train.png')
    # librosa.output.write_wav(root+"reconstruct_train.wav", _wav, sr=44100)
    # librosa.output.write_wav(root+"original_train.wav", _wavlabel, sr=44100)

    print('[INFO] valeur')
    plt.figure(figsize=(15, 6))
    plt.subplot(1,2,1)
    plt.plot(spec.T)
    plt.xlabel('64 bins')
    plt.title('valeur (predict)')
    plt.subplot(1,2,2)
    plt.plot(test_label.T)
    plt.xlabel('64 bins')
    plt.title('valeur (test_label)')
    plt.savefig(root+'_129valeur.png')
    print('[INFO] finished')

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))