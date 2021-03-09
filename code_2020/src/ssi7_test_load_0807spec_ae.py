from ssi6_train0811_parole4_ae import AutoEncoder,match_image_label
from ssi11_input_data_new import *
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

i=2
BATCH_SIZE=32

def TestDatasets_ae():
    train_lips, _, _, _, train_label, _ = load_dataset()
    if i==1 or i==2:
        pass
    else:
        train_label = train_label[:-i+2,:]
    train_lips = match_image_label(train_lips)
    train_lips = torch.from_numpy(train_lips).float()
    train_label = torch.from_numpy(train_label).float().unsqueeze(1)
    train_datasets = TensorDataset(train_lips, train_label)
    train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=False)
    return train_datasets, train_loader

def test_autoencoder(train_datasets, train_loader):
    model2.eval()
    for step,(_,train_label) in enumerate(train_loader):
        spec = Variable(train_label).cuda()
        output,_ = model2(spec)
        if step==0:
            prediction=output
        else:
            prediction=torch.cat((prediction,output),0) #按行竖着接
    print(prediction.shape)
    print('[INFO] test complete')

    return prediction

if __name__ == "__main__":
    start=time.perf_counter()
    model2=AutoEncoder()
    model2.cuda()
    model2.load_state_dict(torch.load('../out/ae_checkpoint.pt'))
    train_datasets2, train_loader2 = TestDatasets_ae()
    prediction2 = test_autoencoder(train_datasets2, train_loader2)
    spec2 = prediction2.cpu().detach().numpy()
    spec2 = np.squeeze(spec2,axis=1)
    np.save("../out/train_predict_me_ae.npy", spec2)
    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))