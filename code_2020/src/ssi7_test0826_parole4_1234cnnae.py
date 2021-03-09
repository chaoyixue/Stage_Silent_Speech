from ssi11_input_data_new import *
# from ssi8_melspectrogram2wav import *
from ssi6_train0825_parole4_cnn2d_ae2 import CNN, match_image_label
# from ssi6_train0507 import test_model,CNN
import matplotlib.pyplot as plt
import torch.optim as optim
import os
# import cv2
import numpy as np
# from sklearn.metrics import r2_score, mean_absolute_error
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
BATCH_SIZE = 32
MOMENTUM=0.9
root='../out/'  
i=8
a=0
max_f01, min_f0 = 240.0, 0

def TestDatasets1():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    train_lips, _, train_tongue, _, _, _ = load_dataset()
    train_label=np.load(root+'train_predict_me_ae.npy')
    f0 = np.load('../data/new_database/train_f0.npy')
    if i==1 or i==2:
        pass
    else:
        train_label = train_label[:-i+2,:]
        f0=f0[:-i+1,:]
    
    max_f01, min_f0 = f0.max(), f0.min()
    f0 = np.clip((f0 - min_f0) / (max_f01 - min_f0), 1e-8, 1)

    test_lips=train_lips[70000:,:,:,:]
    test_tongue=train_tongue[70000:,:,:,:]
    test_label=train_label[70000-i+1:,:]
    test_f0=f0[70000:,:]

    test_lips = test_lips[a:a+400+i-1,:,:,:]
    test_tongue = test_tongue[a:a+400+i-1,:,:,:]
    test_label = test_label[a:a+400,:]
    test_f0=test_f0[a:a+400,:]
    test_label=np.concatenate((test_label,test_f0),axis=1)


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
    test_datasets1 = TensorDataset(test_lips, test_tongue, test_label)
    test_loader1 = DataLoader(dataset=test_datasets1, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets1, test_loader1

loss_func1 = nn.MSELoss()      #默认reduce=true返回标量，size_average=true返回loss.mean
loss_func2 = nn.MSELoss()   

def test_model(test_datasets, test_loader):
    print('[INFO] start testing, output predict')
    # model.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    mae, test_mae=0.0, 0.0
    # h1,h2=None,None
    for step,(test_lips, test_tongue, test_label) in enumerate(test_loader):
        # test_lips, test_tongue, test_label = Variable(test_lips), Variable(test_tongue), Variable(test_label)
        test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
        output = model(test_lips, test_tongue)
        # output,h1,h2 = model(test_lips, test_tongue,h1,h2)
        # h1,h2 = h1.data,h2.data #h1.data
        # loss = loss_func(output,test_label)
        loss1 = loss_func1(output[:,:736],test_label[:,:736])
        loss2 = loss_func1(output[:,736:],test_label[:,736:])
        loss=loss1+loss2
        test_loss += float(loss.item()*test_lips.size(0))
        # mae = mean_absolute_error(test_label.cpu().detach().numpy(),output.cpu().detach().numpy())
        # test_mae += float(mae*test_lips.size(0))     
        if step==0:
            prediction=output[:,:736]
            prediction_f0=output[:,736:]
            # prediction=output.view(-1,368)
        else:
            prediction=torch.cat((prediction,output[:,:736]),0) #按行竖着接
            prediction_f0=torch.cat((prediction_f0,output[:,736:]),0) #按行竖着接
            # prediction=torch.cat((prediction,output.view(-1,368)),0) #按行竖着接
    print('=====> Average loss: %.8f ' % (test_loss/len(test_datasets)))
    # print('=====> Average loss: %.4f ' % (test_loss/len(test_datasets)), ' | Test mean absolute error: %.4f ' % (test_mae/len(test_datasets)))
    print('[INFO] test complete')

    return prediction,prediction_f0

if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    model=CNN()
    model.cuda()
    # model.load_state_dict(torch.load('./ssi/picture/0428checkpoint.pt'))
    # model.eval()
    model.load_state_dict(torch.load(root+'checkpoint.pt'))

    test_datasets1, test_loader1 = TestDatasets1()
    print('[INFO] begin test')
    prediction1,prediction1_f0 = test_model(test_datasets1, test_loader1)
    print('[INFO] save test output')
    spec1 = prediction1.cpu().detach().numpy()
    f0 = prediction1_f0.cpu().detach().numpy()
    f0 = (np.clip(f0,1e-8,1)*(max_f01-min_f0))+min_f0
    print(spec1.shape,f0.shape)
    np.save(root+"test_predict_me%s.npy"%a, spec1)
    np.save(root+"test_predict_me_f0%s.npy"%a, f0)

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))