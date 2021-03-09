# from ssi11_input_data_new import *
from ssi6_train0820_parole4_cnn2d import match_image_label
from ssi6_train0820_parole4_cnn2d_image import CNN_IMAGE,match_image_label2,load_dataset
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import numpy as np
import sys
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from torchvision import transforms
BATCH_SIZE = 32
MOMENTUM=0.9
root='../out/'  
i=2
a=800
max_f02, min_f0 = 213.04347826086956, 0
#images to lsff0uv
def TestDatasets1():
    print('[INFO] set datasets')
    _, test_lsf, _, test_f0, _, test_lips, _, test_uv, _, test_tongue = load_dataset()

    #preprocessing
    test_lips = match_image_label2(test_lips)
    test_tongue = match_image_label2(test_tongue)
    test_lsf = match_image_label(test_lsf)
    test_f0=test_f0[:-1]
    test_uv=test_uv[:-1]

    test_lips = test_lips[a:a+400,:,:,:]
    test_tongue = test_tongue[a:a+400,:,:,:]
    test_lsf= test_lsf[a:a+400,:]
    test_f0= test_f0[a:a+400,:]
    test_uv= test_uv[a:a+400,:]

    #normaliser f0
    # print(train_f0.max(),train_f0.min(),test_f0.max(),test_f0.min()) #240.0 0.0 213.04347826086956 0.0
    # max_f02, min_f0 = test_f0.max(), test_f0.min()
    test_f0 = np.clip((test_f0 - min_f0) / (max_f02 - min_f0), 1e-8, 1)

    #to torch.tensor
    test_lsf = torch.from_numpy(test_lsf).float()
    test_f0 = torch.from_numpy(test_f0).float()
    test_uv = torch.from_numpy(test_uv).float()
    test_lips = torch.from_numpy(test_lips).float()
    test_tongue = torch.from_numpy(test_tongue).float()

    #change dimension match: (x,64,64,6) --> (x,6,64,64)
    test_lips = test_lips.permute(0,3,1,2) 
    test_tongue = test_tongue.permute(0,3,1,2)

    #set datasets and dataloader
    test_datasets = TensorDataset(test_lsf, test_f0, test_uv, test_lips, test_tongue)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets, test_loader

loss_func1 = nn.MSELoss()      #默认reduce=true返回标量，size_average=true返回loss.mean
loss_func2 = nn.MSELoss()      #默认reduce=true返回标量，size_average=true返回loss.mean
loss_func3 = nn.BCELoss()

def test_model(test_datasets, test_loader):
    # model.load_state_dict(torch.load(root+'checkpoint.pt'))
    print('[INFO] start testing, output predict')
    model.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    for step,(test_lsf, test_f0, test_uv, test_lips, test_tongue) in enumerate(test_loader):
        test_lsf, test_f0, test_uv, test_lips, test_tongue = Variable(test_lsf).cuda(), Variable(test_f0).cuda(), Variable(test_uv).cuda(), \
            Variable(test_lips).cuda(), Variable(test_tongue).cuda()
        out_lsf, out_f0, out_uv = model(test_lips, test_tongue)
        loss1 = loss_func1(out_lsf.view(-1,2,12),test_lsf)
        loss2 = loss_func2(out_f0,test_f0)
        loss3 = loss_func3(out_uv,test_uv)
        loss = loss1+loss2+loss3
        test_loss += float(loss.item()*test_lsf.size(0))
        mask=(out_uv>0.5).float()
        if step==0:
            prediction_lsf = out_lsf.view(-1,2,12)
            prediction_f0 = out_f0
            prediction_uv = mask
        else:
            prediction_lsf=torch.cat((prediction_lsf,out_lsf.view(-1,2,12)),0) #按行竖着接
            prediction_f0=torch.cat((prediction_f0,out_f0),0) 
            prediction_uv=torch.cat((prediction_uv,mask),0) 
    print('=====> Average loss: %.8f ' % (test_loss/len(test_datasets)))
    print('[INFO] test complete')

    return prediction_lsf,prediction_f0,prediction_uv

if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    model=CNN_IMAGE()
    model.cuda()
    model.load_state_dict(torch.load(root+'checkpoint_image.pt'))

    test_datasets1, test_loader1 = TestDatasets1()
    print('[INFO] begin test')
    prediction_lsf,prediction_f0,prediction_uv = test_model(test_datasets1, test_loader1)
    print('[INFO] save test output')
    spec1 = prediction_lsf.cpu().detach().numpy()
    spec2 = prediction_f0.cpu().detach().numpy()
    spec2 = (np.clip(spec2,1e-8,1)*(max_f02-min_f0))+min_f0
    spec3 = prediction_uv.cpu().detach().numpy()
    np.save(root+"test_predict_melsf%s.npy" %a, spec1)
    np.save(root+"test_predict_mef0%s.npy" %a, spec2)
    np.save(root+"test_predict_meuv%s.npy" %a, spec3)

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))