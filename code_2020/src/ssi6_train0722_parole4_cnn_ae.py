#autoencoder可用来初始化神经网络的权重(pre-training)和降维。若激活函数为linear，相当于PCA
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# from ssi4_input_data import *
from ssi11_input_data_new import *
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
import time
#from torch.utils.checkpoint import checkpoint
from pytorchtools import EarlyStopping
# from torchvision.models import AlexNet
# from torchviz import make_dot
# from torchvision import models

#配置参数parameters
root="../out/"

BATCH_SIZE = 32
BASE_LR= 1e-3
NUM_EPOCH = 50
WEIGHT_DECAY=1e-7
MOMENTUM=0.9
PATIENCE=5
DROPOUT=0.2
plt.switch_backend('agg')  # cuda
i=8

#数据输入input data----------
def match_image_label(image_data): #2D
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
    train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = load_dataset()
    # train_lips = train_lips[:25005,:,:,:]
    # train_tongue = train_tongue[:25005,:,:,:]
    # train_label = train_label[:25000,:]
    # test_lips = test_lips[:11,:,:,:]
    # test_tongue = test_tongue[:11,:,:,:]
    # test_label = test_label[:10,:]
    if i==1 or i==2:
        pass
    else:
        train_label = train_label[:-i+2,:]
        test_label = test_label[:-i+2,:]
        
    #preprocessing
    train_lips = img_train_normalize(train_lips)
    test_lips = img_train_normalize(test_lips)
    train_tongue = img_train_normalize(train_tongue)
    test_tongue = img_train_normalize(test_tongue)
    train_lips = match_image_label(train_lips)
    train_tongue = match_image_label(train_tongue)
    test_lips = match_image_label(test_lips)
    test_tongue = match_image_label(test_tongue)

    #to torch.tensor
    train_lips = torch.from_numpy(train_lips).float()
    test_lips = torch.from_numpy(test_lips).float()
    train_tongue = torch.from_numpy(train_tongue).float()
    test_tongue = torch.from_numpy(test_tongue).float()
    train_label = torch.from_numpy(train_label).float() 
    test_label = torch.from_numpy(test_label).float()

    #change dimension match: (x,64,64,6) --> (x,6,64,64)
    train_lips = train_lips.permute(0,3,1,2)  
    test_lips = test_lips.permute(0,3,1,2) 
    train_tongue = train_tongue.permute(0,3,1,2)  
    test_tongue = test_tongue.permute(0,3,1,2)

    #set datasets and dataloader
    train_datasets = TensorDataset(train_lips, train_tongue, train_label)
    train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
    eval_datasets = TensorDataset(test_lips, test_tongue, test_label)
    eval_loader = DataLoader(dataset=eval_datasets, batch_size=BATCH_SIZE, shuffle=True)
    test_datasets = TensorDataset(test_lips, test_tongue, test_label)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return train_datasets, train_loader, eval_datasets, eval_loader, test_datasets, test_loader

#Encoder和Decoder
class AutoEncoder(nn.Module):
    def cnn2d(self,in_c, out_c,k_s=3,padd=1):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_s, padding=padd, bias=True),#（2*64*64) 若在卷积后加bn，最好bias=False
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=k_s, padding=padd, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2))
            # nn.Dropout(DROPOUT))
        return conv

    def deco2d(self,in_c, out_c,k_s=3,padd=1):
        conv = nn.Sequential(
            nn.ConvTranspose2d(in_c, in_c, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,mode='nearest'))    #或者bilinear
            # nn.Dropout(DROPOUT))
        return conv

    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder1 = self.cnn2d(i,64)
        # self.encoder2 = self.cnn2d(16,32)
        # self.encoder3 = self.cnn2d(32,64)
        self.encoder4 = self.cnn2d(64,128)
        self.encoder5 = self.cnn2d(128,256)
        self.decoder1 = self.deco2d(256,128)
        self.decoder2 = self.deco2d(128,64)
        # self.decoder3 = self.deco2d(64,32)
        # self.decoder4 = self.deco2d(32,16)
        self.decoder5 = self.deco2d(64,i)
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(i, i, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(i),
            nn.ReLU())
            # nn.Dropout(DROPOUT))

        self.encoder11 = self.cnn2d(i,64)
        # self.encoder12 = self.cnn2d(16,32)
        # self.encoder13 = self.cnn2d(32,64)
        self.encoder14 = self.cnn2d(64,128)
        self.encoder15 = self.cnn2d(128,256)
        self.decoder11 = self.deco2d(256,128)
        self.decoder12 = self.deco2d(128,64)
        # self.decoder13 = self.deco2d(64,32)
        # self.decoder14 = self.deco2d(32,16)
        self.decoder15 = self.deco2d(64,i)
        self.decoder16 = nn.Sequential(
            nn.ConvTranspose2d(i, i, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(i),
            nn.ReLU())
            # nn.Dropout(DROPOUT))

        self.dense1 = nn.Sequential(
            nn.Linear(8*8*256*2, 1024), # 卷积核3*3*16， 64-3+1=62， 输出62*62*16
            nn.ReLU(),
            nn.Dropout(DROPOUT))
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 736), # 卷积核3*3*16， 64-3+1=62， 输出62*62*16
            nn.ReLU())

    def forward(self, lips, tongue):
        out1 = self.encoder1(lips)                
        # out1 = self.encoder2(out1)
        # out1 = self.encoder3(out1)  
        out1 = self.encoder4(out1)                
        out1 = self.encoder5(out1)                          
        out_lips = self.decoder1(out1)
        out_lips = self.decoder2(out_lips)
        # out_lips = self.decoder3(out_lips)
        # out_lips = self.decoder4(out_lips)
        out_lips = self.decoder5(out_lips)
        out_lips = self.decoder6(out_lips)
        out1 = out1.view(out1.size(0),-1)
        
        out2 = self.encoder11(tongue)                
        # out2 = self.encoder12(out2)
        # out2 = self.encoder13(out2)  
        out2 = self.encoder14(out2)                
        out2 = self.encoder15(out2)                          
        out_tongue = self.decoder11(out2)
        out_tongue = self.decoder12(out_tongue)
        # out_tongue = self.decoder13(out_tongue)
        # out_tongue = self.decoder14(out_tongue)
        out_tongue = self.decoder15(out_tongue)
        out_tongue = self.decoder16(out_tongue)
        out2 = out2.view(out2.size(0),-1)
        
        out = torch.cat((out1, out2),dim=1)
        out = self.dense1(out)
        out = self.dense2(out)

        return out, out_lips, out_tongue

autoencoder=AutoEncoder()
autoencoder.cuda()
print(autoencoder)

#优化和损失函数optimizer and loss function
# parameters=list(encoder.parameters())+list(decoder.parameters())
# optimizer = optim.Adam(autoencoder.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)  
# loss_func = nn.CrossEntropyLoss()      
optimizer = optim.Adam(autoencoder.parameters(), lr=BASE_LR, betas=(0.9, 0.999),eps=1e-08, weight_decay=WEIGHT_DECAY)  # wd正则化
loss_func1 = nn.MSELoss()  
loss_func2 = nn.MSELoss()  
loss_func3 = nn.MSELoss()  

###自编码器训练
def main():
    train_losses, eval_losses=[], []
    train_lips_losses,train_tongue_losses,train_lipstongue_losses=[], [], []
    early_stopping=EarlyStopping(patience=PATIENCE,verbose=True)
    for epoch in range(NUM_EPOCH):
        print('[INFO] start training ')
        autoencoder.train()
        train_loss, train_lips_loss, train_tongue_loss, train_lipstongue_loss=0.0, 0.0, 0.0, 0.0
        for step, (train_lips, train_tongue, train_label) in enumerate(train_loader):
            train_lips, train_tongue, train_label = Variable(train_lips).cuda(), Variable(train_tongue).cuda(), Variable(train_label).cuda()
            optimizer.zero_grad()
            output, output_lips, output_tongue = autoencoder(train_lips,train_tongue)
            loss_lips=loss_func1(output_lips,train_lips)
            loss_tongue=loss_func2(output_tongue,train_tongue)
            loss_lipstongue=loss_func3(output,train_label)
            loss=loss_lipstongue+loss_lips+loss_tongue
            loss.backward()        
            optimizer.step()
            train_loss += float(loss.item()*train_lips.size(0))
            train_lips_loss += float(loss_lips.item()*train_lips.size(0))
            train_tongue_loss += float(loss_tongue.item()*train_lips.size(0))
            train_lipstongue_loss += float(loss_lipstongue.item()*train_lips.size(0))
            if step%100==99:
                print('Epoch:[%d/%d], Step:[%d/%d], Step loss: %.4f' % (epoch + 1, NUM_EPOCH, step + 1, len(train_datasets) // BATCH_SIZE, loss.item()))
        train_losses.append(train_loss/len(train_datasets))
        train_lips_losses.append(train_lips_loss/len(train_datasets))
        train_tongue_losses.append(train_tongue_loss/len(train_datasets))
        train_lipstongue_losses.append(train_lipstongue_loss/len(train_datasets))
        print('=====> Epoch:',epoch+1, ' | Average epoch train loss total: %.4f' % (train_loss/len(train_datasets)))
        
        print('[INFO] start evaluation')
        autoencoder.eval()
        with torch.no_grad():
            eval_loss=0.0
            for step,(test_lips, test_tongue, test_label) in enumerate(eval_loader):
                test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
                output, output_lips, output_tongue = autoencoder(test_lips,test_tongue)
                loss_lips=loss_func1(output_lips,test_lips)
                loss_tongue=loss_func2(output_tongue,test_tongue)
                loss_lipstongue=loss_func3(output,test_label)
                loss=loss_lipstongue+loss_lips+loss_tongue
                eval_loss += float(loss.item()*test_lips.size(0))
            eval_losses.append(eval_loss/len(eval_datasets))
            print('=====> Epoch:',epoch+1, ' | Average epoch eval loss: %.4f ' % (eval_loss/len(eval_datasets)))
            print('[INFO] evaluation complete')

        # early_stopping(train_loss/len(train_datasets),autoencoder)
        early_stopping(eval_loss/len(test_datasets),autoencoder)
        if early_stopping.early_stop:
            print('[INFO] early stop')
            break

        # torch.save(encoder.state_dict(),'./autoencoder.pth')
    return train_losses, eval_losses, train_lips_losses, train_tongue_losses, train_lipstongue_losses

def test_autoencoder():
    autoencoder.load_state_dict(torch.load(root+'checkpoint.pt'))
    print('[INFO] start testing, output predict')
    autoencoder.eval()
    test_loss=0.0
    # mae, test_mae=0.0, 0.0
    for step,(test_lips, test_tongue, test_label) in enumerate(test_loader):
        test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
        output, output_lips, output_tongue = autoencoder(test_lips,test_tongue)
        loss_lips=loss_func1(output_lips,test_lips)
        loss_tongue=loss_func2(output_tongue,test_tongue)
        loss_lipstongue=loss_func3(output,test_label)
        loss=loss_lipstongue+loss_lips+loss_tongue
        test_loss += float(loss.item()*test_lips.size(0))
        # mae = mean_absolute_error(test_label.cpu().detach().numpy(),output.cpu().detach().numpy())
        # test_mae += float(mae*test_lips.size(0))     
        if step==0:
            # prediction=output.view(-1,128)
            prediction=output
        else:
            prediction=torch.cat((prediction,output),0) #按行竖着接
            # prediction=torch.cat((prediction,output.view(-1,128)),0) #按行竖着接
    print('=====> Average loss: %.4f ' % (test_loss/len(test_datasets)))
    # print('=====> Average loss: %.4f ' % (test_loss/len(test_datasets)), ' | Test mean absolute error: %.4f ' % (test_mae/len(test_datasets)))
    print('[INFO] test complete')

    return prediction

if __name__ == "__main__":
    start=time.perf_counter()
    train_datasets, train_loader, eval_datasets, eval_loader, test_datasets, test_loader = SSIDatasets()
    # train_losses, eval_losses = main()
    train_losses, eval_losses, train_lips_losses, train_tongue_losses, train_lipstongue_losses = main()

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
    plt.plot(train_lips_losses,color='green')
    plt.plot(train_tongue_losses,color='red')
    plt.plot(train_lipstongue_losses,color='blue')
    # plt.plot()
    plt.legend(['train_lips_losses','train_tongue_losses','train_lipstongue_losses'],loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(root+"multiple_train_loss.png")

    # fig3=plt.figure(figsize=(10,8))
    # plt.plot(train_lips_losses,color='green')
    # plt.legend(['autoencoder_train_lips_losses'],loc='upper right')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.grid(True)
    # plt.savefig(root+"train_lips_loss.png")

    # fig4=plt.figure(figsize=(10,8))
    # plt.plot(train_tongue_losses,color='red')
    # # plt.plot()
    # plt.legend(['autoencoder_train_tongue_losses'],loc='upper right')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.grid(True)
    # plt.savefig(root+"multiple_train_loss.png")

    fig5=plt.figure(figsize=(10,8))
    plt.plot(train_lipstongue_losses,color='blue')
    plt.legend(['train_lipstongue_loss(predict)'],loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(root+"train_lipstongue_loss.png")
    
    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))