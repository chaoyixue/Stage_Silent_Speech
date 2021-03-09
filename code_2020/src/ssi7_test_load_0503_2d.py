from ssi4_input_data import *
# from ssi8_melspectrogram2wav import *
from ssi6_train0517_2 import CNN
# from ssi6_train0507 import test_model,CNN
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import cv2
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
# import librosa
# import librosa.display
import sys
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from torchvision import transforms
# from scipy import signal
BATCH_SIZE = 100
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
root='../out/0504'  

def match_image_label0(image_data): #2D 2images
    l=image_data.shape[0]
    image_match=[]
    for i in range(l-1):
        image_con = np.concatenate((image_data[i],image_data[i+1]),axis=-1)
        image_match.append(image_con)
    image_match = np.array(image_match)

    return image_match

def match_image_label(image_data): #2D 6images
    l=image_data.shape[0]
    image_match=[]
    for i in range(l-12):
        image_con=np.concatenate((image_data[i:i+13]),axis=-1) #(batch_size, 64, 64, 6)
        image_match.append(image_con)
    image_match = np.array(image_match)
    
    return image_match

def TestDatasets():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    _, test_lips, _, test_tongue, _, test_label = load_dataset()
    test_lips = test_lips[6403:,:,:,:]
    test_tongue = test_tongue[6403:,:,:,:]
    test_label = test_label[6403:,:]
    # test_lips = test_lips[800:1012,:,:,:]
    # test_tongue = test_tongue[800:1012,:,:,:]
    # test_label = test_label[800:1000,:]
    # train_lips, _, train_tongue, _, train_label, _ = load_dataset()
    # test_lips = train_lips[1000:1412,:,:,:]
    # test_tongue = train_tongue[1000:1412,:,:,:]
    # test_label = train_label[1000:1400,:]

    # #preprocessing
    test_lips = match_image_label(test_lips)
    test_tongue = match_image_label(test_tongue)
    
    #to torch.tensor
    test_lips = torch.from_numpy(test_lips).float()
    test_tongue = torch.from_numpy(test_tongue).float()
    test_label = torch.from_numpy(test_label).float()

    #change dimension (x,64,64,1) --> (x,1,64,64)
    test_lips = test_lips.permute(0,3,1,2)
    test_tongue = test_tongue.permute(0,3,1,2)

    #两通道
    test_datasets = TensorDataset(test_lips, test_tongue, test_label)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets, test_loader

loss_func = nn.MSELoss() 

def test_model(test_datasets, test_loader):
    model.load_state_dict(torch.load(root+'checkpoint.pt'))
    print('[INFO] start testing, output predict')
    # model.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    mae, test_mae=0.0, 0.0
    for step,(test_lips, test_tongue, test_label) in enumerate(test_loader):
        # test_lips, test_tongue, test_label = Variable(test_lips), Variable(test_tongue), Variable(test_label)
        test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
        output = model(test_lips, test_tongue)
        loss = loss_func(output,test_label)
        test_loss += float(loss.item()*test_lips.size(0))
        mae = mean_absolute_error(test_label.cpu().detach().numpy(),output.cpu().detach().numpy())
        test_mae += float(mae*test_lips.size(0))     
        if step==0:
            # prediction=output.view(-1,128)
            prediction=output
        else:
            prediction=torch.cat((prediction,output),0) #按行竖着接
            # prediction=torch.cat((prediction,output.view(-1,128)),0) #按行竖着接
    # print('=====> Average loss: %.4f ' % (test_loss/len(test_datasets)))
    print('=====> Average loss: %.4f ' % (test_loss/len(test_datasets)), ' | Test mean absolute error: %.4f ' % (test_mae/len(test_datasets)))
    print('[INFO] test complete')

    return prediction

if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    model=CNN()
    model.cuda()
    # model.load_state_dict(torch.load('./ssi/picture/0428checkpoint.pt'))
    # model.eval()
    test_datasets, test_loader = TestDatasets()
    print('[INFO] begin test')
    prediction = test_model(test_datasets, test_loader)
    print('[INFO] save test output')
    spec = prediction.cpu().detach().numpy()
    # spec = prediction.detach().numpy()  #(20451, 1025)

    # # spec = min_max_scaler2.inverse_transform(spec)
    np.save(root+"test_predict.npy", spec)

    # # spec = np.load(root+'test_predict.npy') #(6817, 64)
    # # test_label = np.load('./ssi/data/database/train_label.npy')
    # # test_label=test_label[1000:1400,:]
    # test_label = np.load('./ssi/data/database/test_label.npy')
    # test_label=test_label[6403:,:]
    # print(spec.shape)
    # print(test_label.shape)

    # print('[INFO] save spectrogram picture')
    # plt.figure(figsize=(15, 10))
    # plt.subplot(2,1,1)
    # librosa.display.specshow(spec.T,sr =44100, hop_length=735, x_axis='s', y_axis='mel')
    # # plt.colorbar(format='%+2.0f dB')
    # plt.title('mel spectrogram (predict)')
    # # plt.title('Linear-frequency power spectrogram (predict)')
    # plt.subplot(2,1,2)
    # librosa.display.specshow(test_label.T,sr =44100,hop_length=735, x_axis='s', y_axis='mel')
    # # librosa.display.specshow(test_label.T, x_axis='time', y_axis='linear')
    # # plt.colorbar(format='%+2.0f dB')
    # plt.title('mel spectrogram (test_label)')
    # # plt.title('Linear-frequency power spectrogram (test_label)')
    # plt.savefig(root+'_129spectrogram_predict_test.png')


    # # print('[INFO] generate wave')
    # # # # _wav=spectrogram2wav(spec)   #(1025,20451)-->(20451, 1025)
    # # # # _wavlabel=spectrogram2wav(test_label)
    # # # _wav=melspectrogram2wav(spec)   #(1025,20451)-->(20451, 1025)
    # # # _wavlabel=melspectrogram2wav(test_label)

    # # ref_db=1e-1
    # # max_db=78 #80
    # # spec = (np.clip(spec, 0, 1) * max_db) - max_db + ref_db # de-norm 把小于0的都改为0，大于1的都改为1
    # # # s = librosa.db_to_amplitude(spec.T)
    # # s = librosa.db_to_power(spec.T)
    # # _wav = librosa.feature.inverse.mel_to_audio(s,sr=44100,n_fft=9555, hop_length=735, power=1.2,n_iter=50)

    # # test_label= (np.clip(test_label, 0, 1) * max_db) - max_db + ref_db #把小于0的都改为0，大于1的都改为1
    # # s1 = librosa.db_to_power(test_label.T)
    # # _wavlabel = librosa.feature.inverse.mel_to_audio(s1,sr=44100,n_fft=9555, hop_length=735, power=1.2,n_iter=50)

    # # plt.figure(figsize=(15, 10))
    # # plt.subplot(2,1,1)
    # # librosa.display.waveplot(_wav, sr=44100,x_axis= 's')
    # # # ipd.Audio(_wav, rate=sr)
    # # plt.title('from predict images')
    # # plt.subplot(2,1,2)
    # # librosa.display.waveplot(_wavlabel, sr=44100,x_axis='s')
    # # plt.title('from test_label')
    # # plt.savefig(root+'_129wav_reconstruit_test.png')
    # # # librosa.output.write_wav(root+"reconstruct_train.wav", _wav, sr=44100)
    # # # librosa.output.write_wav(root+"original_train.wav", _wavlabel, sr=44100)

    # print('[INFO] valeur')
    # plt.figure(figsize=(15, 6))
    # plt.subplot(1,2,1)
    # plt.plot(spec.T)
    # plt.xlabel('64 bins')
    # plt.title('valeur (predict)')
    # # plt.title('Linear-frequency power spectrogram (predict)')
    # plt.subplot(1,2,2)
    # plt.plot(test_label.T)
    # plt.xlabel('64 bins')
    # plt.title('valeur (test_label)')
    # # plt.title('Linear-frequency power spectrogram (test_label)')
    # plt.savefig(root+'_129valeur_test.png')
    # print('[INFO] finished')


    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))