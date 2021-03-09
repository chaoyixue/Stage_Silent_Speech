from ssi11_input_data_new import *
from ssi6_train0825_parole4_cnn2d import CNN,match_image_label,load_dataset
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
i=1
a=800

#lsff0uv to spectrogramme


def TestDatasets1():
    print('[INFO] set datasets')
    _, test_lsf, _, test_f0, _, test_label, _, test_uv = load_dataset()
    #preprocessing
    # test_lsf = match_image_label(test_lsf)
    test_lsf=test_lsf[:,:-2,:]
    test_label=test_label[:-1]

    #to torch.tensor
    test_lsf = torch.from_numpy(test_lsf).float()
    test_f0 = torch.from_numpy(test_f0).float()
    test_uv = torch.from_numpy(test_uv).float()
    test_label = torch.from_numpy(test_label).float()
    test_lsf = test_lsf.permute(1,2,0) 

    test_lsf=test_lsf[a:a+400]
    test_f0=test_f0[a:a+400]
    test_uv=test_uv[a:a+400]
    test_label=test_label[a:a+400]

    #set datasets and dataloader
    test_datasets = TensorDataset(test_lsf, test_f0, test_uv, test_label)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets, test_loader

loss_func = nn.MSELoss() 

def test_model(test_datasets, test_loader):
    # model.load_state_dict(torch.load(root+'checkpoint.pt'))
    print('[INFO] start testing, output predict')
    model.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    for step,(test_lsf, test_f0, test_uv, test_label) in enumerate(test_loader):
        test_lsf, test_f0, test_uv, test_label = Variable(test_lsf).cuda(), Variable(test_f0).cuda(), Variable(test_uv).cuda(), Variable(test_label).cuda()
        output = model(test_lsf, test_f0, test_uv)
        loss = loss_func(output,test_label)
        test_loss += float(loss.item()*test_lsf.size(0))
        if step==0:
            prediction=output
        else:
            prediction=torch.cat((prediction,output),0) #按行竖着接
    print('=====> Average loss: %.4f ' % (test_loss/len(test_datasets)))
    print('[INFO] test complete')

    return prediction

if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    model=CNN()
    model.cuda()
    model.load_state_dict(torch.load(root+'checkpoint.pt'))

    test_datasets1, test_loader1 = TestDatasets1()
    print('[INFO] begin test')
    prediction1 = test_model(test_datasets1, test_loader1)
    print('[INFO] save test output')
    spec1 = prediction1.cpu().detach().numpy()
    np.save(root+"test_predict_me%s.npy" %a, spec1)

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))