#import struct
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# from ssi11_input_data_new import *
from ssi6_train0821_parole4_cnn2d import match_image_label
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
# from sklearn.metrics import r2_score, mean_absolute_error
# from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
# import cv2
import numpy as np 
# import librosa
# import librosa.display
import time
#from torch.utils.checkpoint import checkpoint
from pytorchtools_image import EarlyStopping
# from torchvision.models import AlexNet
# from torchviz import make_dot
# from torchvision import models

#images to lsff0uv

root="../out/"

BATCH_SIZE = 32
BASE_LR= 1e-3
NUM_EPOCH = 50
WEIGHT_DECAY=1e-7
MOMENTUM=0.9
PATIENCE=5
DROPOUT=0.2
plt.switch_backend('agg')
i=2

def load_dataset(dir_path="../data/new_database/"):
	if os.path.exists(dir_path):
		arr = []
		for num in enumerate(['train_lsf', 'train_f0', 'test_lsf', 'test_f0', \
            'train_lips', 'test_lips', 'train_uv', 'test_uv','train_tongue','test_tongue']):
			path = "../data/new_database/%s.npy"%num[1]
			arr.append(np.load(path))
		train_lsf, test_lsf, train_f0, test_f0, train_lips, test_lips, train_uv, test_uv, train_tongue, test_tongue = \
            arr[0], arr[2], arr[1], arr[3], arr[4], arr[5], arr[6], arr[7],  arr[8], arr[9]
	else:
		print("Error")
	return train_lsf, test_lsf, train_f0, test_f0, train_lips, test_lips, train_uv, test_uv, train_tongue, test_tongue

#数据输入input data----------
def match_image_label2(image_data): #2D
    l=image_data.shape[0]
    image_match=[]
    for m in range(l-i+1):
        image_con=np.concatenate((image_data[m:m+i]),axis=-1) #(batch_size, 64, 64, 6)
        image_match.append(image_con)
    image_match = np.array(image_match)
    
    return image_match

def SSIDatasets():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    train_lsf, test_lsf, train_f0, test_f0, train_lips, test_lips, train_uv, test_uv, train_tongue, test_tongue = load_dataset()
    # print('1: ',train_lsf.shape, test_lsf.shape, train_f0.shape, test_f0.shape, train_lips.shape, test_lips.shape, train_uv.shape, test_uv.shape,train_tongue.shape, test_tongue.shape )

    #preprocessing
    train_lips = match_image_label2(train_lips)
    train_tongue = match_image_label2(train_tongue)
    test_lips = match_image_label2(test_lips)
    test_tongue = match_image_label2(test_tongue)
    # train_lsf = match_image_label(train_lsf)
    # test_lsf = match_image_label(test_lsf)
    train_f0=train_f0[:-1]
    test_f0=test_f0[:-1]
    train_uv=train_uv[:-1]
    test_uv=test_uv[:-1]
    train_lsf=train_lsf[:,:-1,:]
    test_lsf=test_lsf[:,:-1,:]

    #normaliser f0
    # print(train_f0.max(),train_f0.min(),test_f0.max(),test_f0.min()) #240.0 0.0 213.04347826086956 0.0
    max_f01, max_f02, min_f0 = train_f0.max(), test_f0.max(), train_f0.min()
    train_f0 = np.clip((train_f0 - min_f0) / (max_f01 - min_f0), 1e-8, 1)
    test_f0 = np.clip((test_f0 - min_f0) / (max_f02 - min_f0), 1e-8, 1)

    #to torch.tensor
    train_lsf = torch.from_numpy(train_lsf).float()
    test_lsf = torch.from_numpy(test_lsf).float()
    train_f0 = torch.from_numpy(train_f0).float()
    test_f0 = torch.from_numpy(test_f0).float()
    train_uv = torch.from_numpy(train_uv).float()
    test_uv = torch.from_numpy(test_uv).float()
    train_lips = torch.from_numpy(train_lips).float()
    test_lips = torch.from_numpy(test_lips).float()
    train_tongue = torch.from_numpy(train_tongue).float()
    test_tongue = torch.from_numpy(test_tongue).float()

    #change dimension match: (x,64,64,6) --> (x,6,64,64)
    train_lips = train_lips.permute(0,3,1,2)  
    test_lips = test_lips.permute(0,3,1,2) 
    train_tongue = train_tongue.permute(0,3,1,2)  
    test_tongue = test_tongue.permute(0,3,1,2)
    train_lsf = train_lsf.permute(1,2,0)  
    test_lsf = test_lsf.permute(1,2,0) 

    #set datasets and dataloader
    train_datasets = TensorDataset(train_lsf, train_f0, train_uv, train_lips, train_tongue)
    train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
    eval_datasets = TensorDataset(test_lsf, test_f0, test_uv, test_lips, test_tongue)
    eval_loader = DataLoader(dataset=eval_datasets, batch_size=BATCH_SIZE, shuffle=True)
    test_datasets = TensorDataset(test_lsf, test_f0, test_uv, test_lips, test_tongue)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return train_datasets, train_loader, eval_datasets, eval_loader, test_datasets, test_loader

#调整lr，adjust lr-----------
def adjust_lr(optimizer,epoch):
    if (epoch+1)%10==0:
        for param_group in optimizer.param_groups:
            param_group['lr']=param_group['lr']*0.1 #每10个epoch lr*0.1

class CNN_IMAGE(nn.Module):
    def cnn2d(self,in_c, out_c,k_s=3,padd=1):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_s, padding=padd, bias=True),#（2*64*64) 若在卷积后加bn，最好bias=False
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=k_s, padding=padd, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),    #（16*32*32）
            nn.BatchNorm2d(out_c))
            # nn.Dropout(DROPOUT))
        return conv

    def __init__(self):
        super(CNN_IMAGE, self).__init__()
        self.conv1 = self.cnn2d(i,8)
        self.conv2 = self.cnn2d(8,16)
        self.conv3 = self.cnn2d(16,32)
        self.conv7 = self.cnn2d(32,64)
        self.conv8 = self.cnn2d(64,128)

        self.conv4 = self.cnn2d(i,8)
        self.conv5 = self.cnn2d(8,16)
        self.conv6 = self.cnn2d(16,32)
        self.conv9 = self.cnn2d(32,64)
        self.conv10 = self.cnn2d(64,128)

        self.dense1 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT))

        self.dense2 = nn.Sequential(
            nn.Linear(128, 12),
            nn.ReLU())
        self.dense3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU())# 旧，jyr

        self.dense4 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())# 旧，jyr
            # nn.LeakyReLU())
            # nn.Softmax()#分类
            # nn.Dropout(DROPOUT))

    def forward(self, lips, tongue):
        out1 = self.conv1(lips)                
        out1 = self.conv2(out1)                
        out1 = self.conv3(out1)  
        out1 = self.conv7(out1)     
        out1 = self.conv8(out1)   
        out1 = out1.view(out1.size(0),-1)
        
        out2 = self.conv4(tongue)              
        out2 = self.conv5(out2)                
        out2 = self.conv6(out2)    
        out2 = self.conv9(out2)
        out2 = self.conv10(out2)
        out2 = out2.view(out2.size(0),-1) 
        
        out = torch.cat((out1, out2),dim=1)
        out = self.dense1(out)

        out_lsf = self.dense2(out)
        out_f0 = self.dense3(out)
        out_uv = self.dense4(out)

        return out_lsf, out_f0, out_uv

model = CNN_IMAGE()
model.cuda()
print('[INFO] cnn model ---------------------------------------')
print(model)

#优化和损失函数optimizer and loss function----------
# optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)  #随机梯度下降
optimizer = optim.Adam(model.parameters(), lr=BASE_LR, betas=(0.9, 0.999),eps=1e-08, weight_decay=WEIGHT_DECAY)  # wd正则化
loss_func1 = nn.MSELoss()      #默认reduce=true返回标量，size_average=true返回loss.mean
loss_func2 = nn.MSELoss()      #默认reduce=true返回标量，size_average=true返回loss.mean
loss_func3 = nn.BCELoss()

def main():
    train_losses, eval_losses=[], []
    early_stopping=EarlyStopping(patience=PATIENCE,verbose=True)
    for epoch in range(NUM_EPOCH):
        print('[INFO] start training ')
        model.train() #启用batchnormalization和dropout
        train_loss=0.0
        train_loss_lsf, train_loss_f0, train_loss_uv=0.0,0.0,0.0
        for step, (train_lsf, train_f0, train_uv, train_lips, train_tongue) in enumerate(train_loader):
            train_lsf, train_f0, train_uv, train_lips, train_tongue = Variable(train_lsf).cuda(), Variable(train_f0).cuda(), \
                Variable(train_uv).cuda(),Variable(train_lips).cuda(), Variable(train_tongue).cuda()
            optimizer.zero_grad()     #梯度值初始化为0
            out_lsf, out_f0, out_uv = model(train_lips, train_tongue)
            loss1 = loss_func1(out_lsf.unsqueeze(1),train_lsf)
            loss2 = loss_func2(out_f0,train_f0)
            loss3 = loss_func3(out_uv,train_uv)
            loss = loss1+10*loss2+0.5*loss3
            loss.backward()           #反向传播                  
            optimizer.step()          #更新参数
            train_loss += float(loss.item()*train_lips.size(0))
            train_loss_lsf += float(loss1.item()*train_lips.size(0))
            train_loss_f0 += float(loss2.item()*train_lips.size(0))
            train_loss_uv += float(loss3.item()*train_lips.size(0))
            if step%100==99:
                print('Epoch:[%d/%d], Step:[%d/%d], Step loss: %.8f' % (epoch + 1, NUM_EPOCH, step + 1, len(train_datasets) // BATCH_SIZE, loss.item()))
                # print('lsf Step loss: %.8f, f0 Step loss: %.8f , vuv Step loss: %.8f' % (loss1.item(), loss2.item(), loss3.item()))
        train_losses.append(train_loss/len(train_datasets))
        print('=====> Epoch:',epoch+1, ' | Average epoch train loss: %.8f' % (train_loss/len(train_datasets)))
        print('=====> lsf epoch loss: %.8f, f0 epoch loss: %.8f , vuv epoch loss: %.8f' % (train_loss_lsf/len(train_datasets), train_loss_f0/len(train_datasets), train_loss_uv/len(train_datasets)))
        
        adjust_lr(optimizer,epoch) 

        #eval-----------
        print('[INFO] start evaluation')
        model.eval() #不启用batchnormalization和dropout
        with torch.no_grad():
            eval_loss=0.0
            total, correct = 0.0, 0.0
            for step,(test_lsf, test_f0, test_uv, test_lips, test_tongue) in enumerate(eval_loader):
                test_lsf, test_f0, test_uv, test_lips, test_tongue = Variable(test_lsf).cuda(), Variable(test_f0).cuda(), Variable(test_uv).cuda(), \
                    Variable(test_lips).cuda(), Variable(test_tongue).cuda()
                out_lsf, out_f0, out_uv = model(test_lips, test_tongue)
                loss1 = loss_func1(out_lsf.unsqueeze(1),test_lsf)
                loss2 = loss_func2(out_f0,test_f0)
                loss3 = loss_func3(out_uv,test_uv)
                loss = loss1+10*loss2+0.5*loss3
                eval_loss += float(loss.item()*test_lips.size(0))
                total += test_uv.size(0)
                output = (out_uv>0.5).float()
                correct+=(output==test_uv).sum().item()
            eval_losses.append(eval_loss/len(eval_datasets))
            # print('=====> Epoch:',epoch+1, ' | Average epoch eval loss: %.8f ' % (eval_loss/len(eval_datasets)))
            print('=====> Epoch:',epoch+1, ' | Average epoch eval loss: %.8f ' % (eval_loss/len(eval_datasets)), ' | Average epoch eval accuracy: %.8f ' %(100*correct/total))
            print('[INFO] evaluation complete')

        early_stopping(eval_loss/len(test_datasets),model)
        if early_stopping.early_stop:
            print('[INFO] early stop')
            break
        
    return train_losses, eval_losses


def test_model():
    model.load_state_dict(torch.load(root+'checkpoint.pt'))
    print('[INFO] start testing, output predict')
    model.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    for step,(test_lsf, test_f0, test_uv, test_lips, test_tongue) in enumerate(test_loader):
        test_lsf, test_f0, test_uv, test_lips, test_tongue = Variable(test_lsf).cuda(), Variable(test_f0).cuda(), Variable(test_uv).cuda(), \
            Variable(test_lips).cuda(), Variable(test_tongue).cuda()
        out_lsf, out_f0, out_uv = model(test_lips, test_tongue)
        loss1 = loss_func1(out_lsf.unsqueeze(1),test_lsf)
        loss2 = loss_func2(out_f0,test_f0)
        loss3 = loss_func3(out_uv,test_uv)
        loss = loss1+loss2+loss3
        test_loss += float(loss.item()*test_lsf.size(0))
        mask=(out_uv>0.5).float()
        if step==0:
            prediction_lsf = out_lsf.unsqueeze(1)
            prediction_f0 = out_f0
            prediction_uv = mask
        else:
            prediction_lsf=torch.cat((prediction_lsf,out_lsf.unsqueeze(1)),0) #按行竖着接
            prediction_f0=torch.cat((prediction_f0,out_f0),0) 
            prediction_uv=torch.cat((prediction_uv,mask),0) 
    print('=====> Average loss: %.8f ' % (test_loss/len(test_datasets)))
    print('[INFO] test complete')

    return prediction_lsf,prediction_f0,prediction_uv

if __name__ == "__main__":
    start=time.perf_counter()
    train_datasets, train_loader, eval_datasets, eval_loader, test_datasets, test_loader = SSIDatasets()
    train_losses, eval_losses = main()

    print('[INFO] save train result picture')
    fig=plt.figure(figsize=(10,8))
    plt.plot(train_losses,color='red')
    plt.plot(eval_losses,color='blue')
    minloss=eval_losses.index(min(eval_losses))
    plt.axvline(minloss,linestyle='--',color='green')
    plt.legend(['Train Loss','Eval Loss'],loc='upper right')
    plt.title('epoch loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(root+"epoch_loss.png")

    fig2=plt.figure(figsize=(10,8))
    plt.plot(eval_losses,color='green')
    plt.legend(['Eval loss'],loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('Eval loss')
    plt.grid(True)
    plt.savefig(root+"eval_loss.png")

    np.save(root+"train_losses.npy", np.array(train_losses))
    np.save(root+"eval_losses.npy", np.array(eval_losses))

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))