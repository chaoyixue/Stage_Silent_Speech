import os
from ssi11_input_data_new import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
import matplotlib.pyplot as plt
import sys
import numpy as np 
import time
from pytorchtools import EarlyStopping

# train sur gpu

root = "../out/"
name = 'cnn'
BATCH_SIZE = 32
BASE_LR = 1e-4
NUM_EPOCH = 50
WEIGHT_DECAY =1e-7
MOMENTUM = 0.9
PATIENCE = 5      # earlystoop
DROPOUT = 0.2
plt.switch_backend('agg')
i = 8             # N of input image


def match_image_label(image_data): # 2D
    l = image_data.shape[0]
    image_match = []
    for m in range(l-i+1):
        image_con = np.concatenate((image_data[m:m+i]), axis=-1)  # (batch_size, 64, 64, 6)
        image_match.append(image_con)
    image_match = np.array(image_match)
    
    return image_match


def SSIDatasets():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = load_dataset()
    if i == 1 or i == 2:
        pass
    else:
        train_label = train_label[:-i+2, :]
        test_label = test_label[:-i+2, :]

    # preprocessing
    # train_lips = img_train_normalize(train_lips)
    # test_lips = img_train_normalize(test_lips)
    # train_tongue = img_train_normalize(train_tongue)
    # test_tongue = img_train_normalize(test_tongue)
    train_lips = match_image_label(train_lips)
    train_tongue = match_image_label(train_tongue)
    test_lips = match_image_label(test_lips)
    test_tongue = match_image_label(test_tongue)

    # to torch.tensor
    train_lips = torch.from_numpy(train_lips).float()
    test_lips = torch.from_numpy(test_lips).float()
    train_tongue = torch.from_numpy(train_tongue).float()
    test_tongue = torch.from_numpy(test_tongue).float()
    train_label = torch.from_numpy(train_label).float() 
    test_label = torch.from_numpy(test_label).float()

    # change dimension match: (x,64,64,6) --> (x,6,64,64)    #2D
    train_lips = train_lips.permute(0,3,1,2)  
    test_lips = test_lips.permute(0,3,1,2) 
    train_tongue = train_tongue.permute(0,3,1,2)  
    test_tongue = test_tongue.permute(0,3,1,2)

    # #change dimension match2 (x,1,64,64,2) --> (x,1,2,64,64)  #3D
    # train_lips = train_lips.permute(0,1,4,2,3).cuda()  
    # test_lips = test_lips.permute(0,1,4,2,3).cuda()  
    # train_tongue = train_tongue.permute(0,1,4,2,3).cuda()  
    # test_tongue = test_tongue.permute(0,1,4,2,3).cuda()  

    #set datasets and dataloader
    train_datasets = TensorDataset(train_lips, train_tongue, train_label)
    train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
    eval_datasets = TensorDataset(test_lips, test_tongue, test_label)
    eval_loader = DataLoader(dataset=eval_datasets, batch_size=BATCH_SIZE, shuffle=True)
    test_datasets = TensorDataset(test_lips, test_tongue, test_label)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return train_datasets, train_loader, eval_datasets, eval_loader, test_datasets, test_loader

def adjust_lr(optimizer,epoch):
    if (epoch+1)%10==0:
        for param_group in optimizer.param_groups:
            param_group['lr']=param_group['lr']*0.1
            # print('[INFO] lr:',param_group['lr'])

class CNN(nn.Module):
    def cnn2d(self,in_c, out_c,k_s=3,padd=1):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_s, padding=padd, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=k_s, padding=padd, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_c))
            # nn.Dropout(DROPOUT))
        return conv

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = self.cnn2d(i,16)
        self.conv2 = self.cnn2d(16,32)
        self.conv3 = self.cnn2d(32,64)
        self.conv7 = self.cnn2d(64,128)
        self.conv8 = self.cnn2d(128,256)

        self.conv4 = self.cnn2d(i,16)
        self.conv5 = self.cnn2d(16,32)
        self.conv6 = self.cnn2d(32,64)
        self.conv9 = self.cnn2d(64,128)
        self.conv10 = self.cnn2d(128,256)

        self.dense1 = nn.Sequential(
            nn.Linear(2048, 1024), 
            nn.ReLU(),
            nn.Dropout(DROPOUT))
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 736),
            nn.ReLU())

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
        out = self.dense2(out)

        return out

model = CNN()
model.cuda()
# print('[INFO] cnn model ---------------------------------------')
# print(model)

#optimizer and loss function
# optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY) 
optimizer = optim.Adam(model.parameters(), lr=BASE_LR, betas=(0.9, 0.999),eps=1e-08, weight_decay=WEIGHT_DECAY) # WD: regularization
loss_func = nn.MSELoss()      # reduce=true return scalar, size_average=true return loss.mean

# # multiple optim
# optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY) 
# optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# optimizer = [optimizer_Adam, optimizer_SGD]
# loss_func = nn.MSELoss() 
# losses_history = [[],[]]

def main():
    train_losses, eval_losses=[], []
    early_stopping=EarlyStopping(patience=PATIENCE,verbose=True)
    for epoch in range(NUM_EPOCH):
        print('[INFO] start training ')
        model.train()   # activate batchnormalization and dropout
        train_loss=0.0
        #step_loss=0.0
        for step, (train_lips, train_tongue, train_label) in enumerate(train_loader):
            train_lips, train_tongue, train_label = Variable(train_lips).cuda(), Variable(train_tongue).cuda(), Variable(train_label).cuda()
            optimizer.zero_grad()    
            output = model(train_lips, train_tongue)
            loss = loss_func(output,train_label)
            loss.backward()                    
            optimizer.step()  
            train_loss += float(loss.item()*train_lips.size(0))
            if step%100==99:
                print('Epoch:[%d/%d], Step:[%d/%d], Step loss: %.8f' % (epoch + 1, NUM_EPOCH, step + 1, len(train_datasets) // BATCH_SIZE, loss.item()))
        train_losses.append(train_loss/len(train_datasets))
        print('=====> Epoch:',epoch+1, ' | Average epoch train loss: %.8f' % (train_loss/len(train_datasets)))
        
        adjust_lr(optimizer,epoch) 

        #eval-----------
        print('[INFO] start evaluation')
        model.eval()    # deac batchnormalization and dropout
        with torch.no_grad():
            eval_loss=0.0
            for step,(test_lips, test_tongue, test_label) in enumerate(eval_loader):
                test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
                output = model(test_lips,test_tongue)
                loss = loss_func(output,test_label)
                eval_loss += float(loss.item()*test_lips.size(0))
            eval_losses.append(eval_loss/len(eval_datasets))
            print('=====> Epoch:',epoch+1, ' | Average epoch eval loss: %.8f ' % (eval_loss/len(eval_datasets)))
            print('[INFO] evaluation complete')

        early_stopping(eval_loss/len(test_datasets),model,name)
        if early_stopping.early_stop:
            print('[INFO] early stop')
            break

    return train_losses, eval_losses

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

    np.save(root+"train_losses.npy", np.array(train_losses))
    np.save(root+"eval_losses.npy", np.array(eval_losses))

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))