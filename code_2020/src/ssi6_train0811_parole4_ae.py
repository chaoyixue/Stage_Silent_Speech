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
from pytorchtools_ae import EarlyStopping

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
i=2

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
    train_lips, test_lips, _, _, train_label, test_label = load_dataset()
    if i==1 or i==2:
        pass
    else:
        train_label = train_label[:-i+2,:]
        test_label = test_label[:-i+2,:]

    #preprocessing
    train_lips = match_image_label(train_lips)
    test_lips = match_image_label(test_lips)
    # train_label = train_label.reshape(-1,32,23)
    # test_label = test_label.reshape(-1,32,23)

    #to torch.tensor
    train_lips = torch.from_numpy(train_lips).float()
    test_lips = torch.from_numpy(test_lips).float()
    train_label = torch.from_numpy(train_label).float().unsqueeze(1)
    test_label = torch.from_numpy(test_label).float().unsqueeze(1)

    #set datasets and dataloader
    train_datasets = TensorDataset(train_lips, train_label)
    train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
    eval_datasets = TensorDataset(test_lips, test_label)
    eval_loader = DataLoader(dataset=eval_datasets, batch_size=BATCH_SIZE, shuffle=True)
    test_datasets = TensorDataset(test_lips, test_label)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return train_datasets, train_loader, eval_datasets, eval_loader, test_datasets, test_loader

#Encoder和Decoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(736, 368),
            nn.ReLU(),
            nn.Linear(368, 184),
            nn.ReLU(),
            nn.Linear(184, 30))
        self.decoder = nn.Sequential(
            nn.Linear(30, 184),
            nn.ReLU(),
            nn.Linear(184, 368),
            nn.ReLU(),
            nn.Linear(368, 736),
            nn.ReLU())
    def forward(self, spec):
        out_aeen = self.encoder(spec)
        out_aede = self.decoder(out_aeen)
        return out_aede,out_aeen

autoencoder=AutoEncoder()
autoencoder.cuda()
print(autoencoder)
optimizer = optim.Adam(autoencoder.parameters(), lr=BASE_LR, betas=(0.9, 0.999),eps=1e-08, weight_decay=WEIGHT_DECAY)  # wd正则化
loss_func = nn.MSELoss()  

###自编码器训练
def main():
    train_losses, eval_losses=[], []
    early_stopping=EarlyStopping(patience=PATIENCE,verbose=True)
    for epoch in range(NUM_EPOCH):
        print('[INFO] start training ')
        autoencoder.train()
        train_loss=0.0
        for step, (_,train_label) in enumerate(train_loader):
            train_label = Variable(train_label).cuda()
            optimizer.zero_grad()
            output,out_train= autoencoder(train_label)
            loss=loss_func(output,train_label)
            loss.backward()        
            optimizer.step()
            train_loss += float(loss.item()*train_label.size(0))
            if step==0:
                train_label_ae=out_train.squeeze(1)
            else:
                train_label_ae=torch.cat((train_label_ae,out_train.squeeze(1)),0)
            if step%100==99:
                print('Epoch:[%d/%d], Step:[%d/%d], Step loss: %.4f' % (epoch + 1, NUM_EPOCH, step + 1, len(train_datasets) // BATCH_SIZE, loss.item()))
        train_losses.append(train_loss/len(train_datasets))
        print('=====> Epoch:',epoch+1, ' | Average epoch train loss total: %.4f' % (train_loss/len(train_datasets)))
        
        print('[INFO] start evaluation')
        autoencoder.eval()
        with torch.no_grad():
            eval_loss=0.0
            for step,(_,test_label) in enumerate(eval_loader):
                test_label = Variable(test_label).cuda()
                output,out_test = autoencoder(test_label)
                loss=loss_func(output,test_label)
                eval_loss += float(loss.item()*test_label.size(0))
                if step==0:
                    test_label_ae=out_test.squeeze(1)
                else:
                    test_label_ae=torch.cat((test_label_ae,out_test.squeeze(1)),0)
            eval_losses.append(eval_loss/len(eval_datasets))
            print('=====> Epoch:',epoch+1, ' | Average epoch eval loss: %.4f ' % (eval_loss/len(eval_datasets)))
            print('[INFO] evaluation complete')

        early_stopping(eval_loss/len(test_datasets),autoencoder)
        if early_stopping.early_stop:
            print('[INFO] early stop')
            break

        if early_stopping.counter==0:
            train_label_ae=train_label_ae.cpu().detach().numpy()
            test_label_ae=test_label_ae.cpu().detach().numpy()
            print(train_label_ae.shape,test_label_ae.shape)
            np.save(root+'train_label_ae.npy',train_label_ae)
            np.save(root+'test_label_ae.npy',test_label_ae)   
        else:
            pass

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
    plt.savefig(root+"ae_epoch_loss.png")
    
    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))