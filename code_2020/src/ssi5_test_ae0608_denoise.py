from ssi11_input_data_new import *
# from ssi4_input_data import *
# from ssi8_melspectrogram2wav import *
from ssi5_train_ae0608 import AutoEncoder, match_image_label
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
BATCH_SIZE = 128
MOMENTUM=0.9
root='../out/'  
i=2

def TestDatasets1():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    _, test_lips, _, test_tongue, _, test_label = load_dataset()
    test_lips = test_lips[:400+i-1,:,:,:]
    test_tongue = test_tongue[:400+i-1,:,:,:]
    test_label = test_label[:400,:]

    # #preprocessing
    test_lips = img_train_normalize(test_lips)
    test_tongue = img_train_normalize(test_tongue)
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
    test_datasets1 = TensorDataset(test_lips, test_tongue, test_label)
    test_loader1 = DataLoader(dataset=test_datasets1, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets1, test_loader1

def TestDatasets2():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    _, test_lips, _, test_tongue, _, test_label = load_dataset()
    test_lips = test_lips[400:800+i-1,:,:,:]
    test_tongue = test_tongue[400:800+i-1,:,:,:]
    test_label = test_label[400:800,:]

    # #preprocessing
    test_lips = img_train_normalize(test_lips)
    test_tongue = img_train_normalize(test_tongue)
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
    test_datasets2 = TensorDataset(test_lips, test_tongue, test_label)
    test_loader2 = DataLoader(dataset=test_datasets2, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets2, test_loader2

def TestDatasets3():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    _, test_lips, _, test_tongue, _, test_label = load_dataset()
    test_lips = test_lips[1000:1400+i-1,:,:,:]
    test_tongue = test_tongue[1000:1400+i-1,:,:,:]
    test_label = test_label[1000:1400,:]

    # #preprocessing
    test_lips = img_train_normalize(test_lips)
    test_tongue = img_train_normalize(test_tongue)
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
    test_datasets3 = TensorDataset(test_lips, test_tongue, test_label)
    test_loader3 = DataLoader(dataset=test_datasets3, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets3, test_loader3

def TestDatasets4():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    _, test_lips, _, test_tongue, _, test_label = load_dataset()
    test_lips = test_lips[:10+i-1,:,:,:]
    test_tongue = test_tongue[:10+i-1,:,:,:]
    test_label = test_label[:10,:]

    # #preprocessing
    test_lips = img_train_normalize(test_lips)
    test_tongue = img_train_normalize(test_tongue)
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
    test_datasets4 = TensorDataset(test_lips, test_tongue, test_label)
    test_loader4 = DataLoader(dataset=test_datasets4, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets4, test_loader4

loss_func1 = nn.MSELoss()  
loss_func2 = nn.MSELoss()  
loss_func3 = nn.MSELoss() 

def test_autoencoder(test_datasets, test_loader):
    print('[INFO] start testing, output predict')
    autoencoder.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    mae, test_mae=0.0, 0.0
    for step,(test_lips, test_tongue, test_label) in enumerate(test_loader):
        # test_lips, test_tongue, test_label = Variable(test_lips), Variable(test_tongue), Variable(test_label)
        test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
        output, output_lips, output_tongue = autoencoder(test_lips,test_tongue)
        loss_lips=loss_func1(output_lips,test_lips)
        loss_tongue=loss_func2(output_tongue,test_tongue)
        loss_lipstongue=loss_func3(output,test_label)
        loss=loss_lipstongue+loss_lips+loss_tongue
        test_loss += float(loss.item()*test_lips.size(0))
        mae = mean_absolute_error(test_label.cpu().detach().numpy(),output.cpu().detach().numpy())
        test_mae += float(mae*test_lips.size(0))     
        if step==0:
            prediction=output
        else:
            prediction=torch.cat((prediction,output),0) #按行竖着接
    # print('=====> Average loss: %.4f ' % (test_loss/len(test_datasets)))
    print('=====> Average loss: %.4f ' % (test_loss/len(test_datasets)), ' | Test mean absolute error: %.4f ' % (test_mae/len(test_datasets)))
    print('[INFO] test complete')

    return prediction, output_lips

if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    autoencoder=AutoEncoder()
    autoencoder.cuda()
    # model.load_state_dict(torch.load('./ssi/picture/0428checkpoint.pt'))
    # model.eval()
    autoencoder.load_state_dict(torch.load(root+'checkpoint.pt'))

    test_datasets1, test_loader1 = TestDatasets1()
    print('[INFO] begin test')
    prediction1,_ = test_autoencoder(test_datasets1, test_loader1)
    print('[INFO] save test output')
    spec1 = prediction1.cpu().detach().numpy()
    np.save(root+"test_predict_me1.npy", spec1)

    test_datasets2, test_loader2 = TestDatasets2()
    print('[INFO] begin test')
    prediction2,_ = test_autoencoder(test_datasets2, test_loader2)
    print('[INFO] save test output')
    spec2 = prediction2.cpu().detach().numpy()
    np.save(root+"test_predict_me2.npy", spec2)

    test_datasets3, test_loader3 = TestDatasets3()
    print('[INFO] begin test')
    prediction3,_ = test_autoencoder(test_datasets3, test_loader3)
    print('[INFO] save test output')
    spec3 = prediction3.cpu().detach().numpy()
    np.save(root+"test_predict_me3.npy", spec3)

    test_datasets4, test_loader4 = TestDatasets4()
    print('[INFO] begin test')
    _,recon_lips4 = test_autoencoder(test_datasets4, test_loader4)
    # print(recon_lips4.shape)
    print('[INFO] save test output')
    # spec4 = prediction4.cpu().detach().numpy()
    recon_lips4 = recon_lips4.cpu().detach().numpy()
    for i in range(10):
        lips=recon_lips4[i:i+1,:1,:,:]
        # print(lips.shape)
        lips=np.squeeze(lips,axis=None)
        # print(lips.shape)
        cv2.imwrite(root+'lips_recon%s.png' %i, lips)
    # np.save(root+"test_predict_me4.npy", spec4)

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))