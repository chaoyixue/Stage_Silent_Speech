from ssi11_input_data_new import *
from ssi6_train0826_parole4_cnn2d_image_lsf import CNN_IMAGE as CNN1
from ssi6_train0826_parole4_cnn2d_image_lsf import match_image_label2
from ssi6_train0826_parole4_cnn2d_image_f0 import CNN_IMAGE as CNN2
from ssi6_train0826_parole4_cnn2d_image_uv import CNN_IMAGE as CNN3
from ssi6_train0825_parole4_cnn2d import CNN
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
BATCH_SIZE = 128
MOMENTUM=0.9
root='../out/'  
i=2
a=4000
max_f02, min_f0 = 213.04347826086956, 0
# print(train_f0.max(),train_f0.min(),test_f0.max(),test_f0.min()) #240.0 0.0 213.04347826086956 0.0


#images to lsff0uv to spectrogram
def TestDatasets1():
    print('[INFO] set datasets')
    _, test_lips, _, test_tongue, _, test_label = load_dataset()
    #preprocessing
    test_lips = test_lips[a:a+400+i-1,:,:,:]
    test_tongue = test_tongue[a:a+400+i-1,:,:,:]
    test_label = test_label[a:a+400,:]

    test_lips = match_image_label2(test_lips)
    test_tongue = match_image_label2(test_tongue)

    #to torch.tensor
    test_lips = torch.from_numpy(test_lips).float()
    test_tongue = torch.from_numpy(test_tongue).float()
    test_label = torch.from_numpy(test_label).float()

    #change dimension (x,64,64,1) --> (x,1,64,64)
    test_lips = test_lips.permute(0,3,1,2)
    test_tongue = test_tongue.permute(0,3,1,2)

    #set datasets and dataloader
    test_datasets = TensorDataset(test_lips, test_tongue, test_label)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets, test_loader

loss_func = nn.MSELoss()      #默认reduce=true返回标量，size_average=true返回loss.mean

def test_model(test_datasets, test_loader):
    # model.load_state_dict(torch.load(root+'checkpoint.pt'))
    print('[INFO] start testing, output predict')
    model_1.eval() #不启用batchnormalization和dropout
    model_2.eval() #不启用batchnormalization和dropout
    model_3.eval() #不启用batchnormalization和dropout
    model2.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    for step,(test_lips, test_tongue, test_label) in enumerate(test_loader):
        test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
        out_lsf = model_1(test_lips, test_tongue)
        out_f0 = model_2(test_lips, test_tongue)
        out_uv = model_3(test_lips, test_tongue)
        mask=(out_uv>0.5).float() #torch.Size([32, 1])
        # out_f0 = (torch.clamp(out_f0,1e-8,1)*(max_f02-min_f0))+min_f0
        output=model2(out_lsf.unsqueeze(1),out_f0,mask)
        loss = loss_func(output,test_label)
        test_loss += float(loss.item()*test_lips.size(0))
        if step==0:
            prediction=output
        else:
            prediction=torch.cat((prediction,output),0)
    print('=====> Average loss: %.8f ' % (test_loss/len(test_datasets)))
    print('[INFO] test complete')

    return prediction

if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    model_1=CNN1()
    model_1.cuda()
    model_1.load_state_dict(torch.load(root+'checkpoint_lsf.pt'))
    model_2=CNN2()
    model_2.cuda()
    model_2.load_state_dict(torch.load(root+'checkpoint_f0_st.pt'))
    model_3=CNN3()
    model_3.cuda()
    model_3.load_state_dict(torch.load(root+'checkpoint_uv.pt'))
    model2=CNN()
    model2.cuda()
    model2.load_state_dict(torch.load(root+'checkpoint.pt'))

    test_datasets1, test_loader1 = TestDatasets1()
    print('[INFO] begin test')
    prediction1 = test_model(test_datasets1, test_loader1)
    print('[INFO] save test output')
    spec1 = prediction1.cpu().detach().numpy()
    np.save(root+"test_predict_me%s.npy" %a, spec1)

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))