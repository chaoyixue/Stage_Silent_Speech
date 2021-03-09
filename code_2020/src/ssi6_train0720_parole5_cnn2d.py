#import struct
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
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
# import librosa
# import librosa.display
import time
#from torch.utils.checkpoint import checkpoint
from pytorchtools_counter import EarlyStopping
# from torchvision.models import AlexNet
# from torchviz import make_dot
# from torchvision import models

root="../out/"

BATCH_SIZE = 32
BASE_LR= 1e-2
NUM_EPOCH = 50
WEIGHT_DECAY=1e-7
MOMENTUM=0.9
PATIENCE=5
DROPOUT=0.2
plt.switch_backend('agg')
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
    if i==1 or i==2:
        pass
    else:
        train_label = train_label[:-i+2,:]
        test_label = test_label[:-i+2,:]

    print(train_label.shape,train_lips.shape,train_tongue.shape)
    print(test_label.shape,test_lips.shape,test_tongue.shape)

    #preprocessing
    # train_lips = img_train_normalize(train_lips)
    # test_lips = img_train_normalize(test_lips)
    # train_tongue = img_train_normalize(train_tongue)
    # test_tongue = img_train_normalize(test_tongue)
    train_lips = match_image_label(train_lips)
    train_tongue = match_image_label(train_tongue)
    test_lips = match_image_label(test_lips)
    test_tongue = match_image_label(test_tongue)
    print(train_label.shape,train_lips.shape,train_tongue.shape)

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

    # #change dimension match2 (x,1,64,64,2) --> (x,1,2,64,64)
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

    # print(len(train_loader)) #100
    # print(len(train_loader.dataset)) #10000

    return train_datasets, train_loader, eval_datasets, eval_loader, test_datasets, test_loader


#output console information-----------
# class Logger(object):
#     def __init__(self,fileN='Default.log'):
#         self.terminal=sys.stdout
#         self.log=open(fileN,'w')
#     def write(self,message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.flush()
#     def flush(self):
#         self.log.flush()

# sys.stdout = Logger(root+'console information.txt')


#调整lr，adjust lr-----------
# def adjust_lr(optimizer,epoch):
#     if (epoch+1)%10==0:
#         for param_group in optimizer.param_groups:
#             param_group['lr']=param_group['lr']*0.1 #每10个epoch lr*0.1
def adjust_lr(optimizer,epoch,eval_losses,counter):
    if epoch==0:
        pass
    elif counter>=3 and eval_losses[-1]>=eval_losses[-2]:
        for param_group in optimizer.param_groups:
            param_group['lr']=param_group['lr']*0.1 #每10个epoch lr*0.1
    else:
        pass


class CNN(nn.Module):
    def cnn2d(self,in_c, out_c,k_s=3,padd=1):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_s, padding=padd, bias=True),#（2*64*64) 若在卷积后加bn，最好bias=False
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=k_s, padding=padd, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2))    #（16*32*32）
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
            nn.Linear(2048, 1024), # 卷积核3*3*16， 64-3+1=62， 输出62*62*16
            nn.ReLU(),
            nn.Dropout(DROPOUT))
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 736), # 卷积核3*3*16， 64-3+1=62， 输出62*62*16
            nn.ReLU())
            # nn.LeakyReLU())
            # nn.Sigmoid())# 旧，jyr
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
        out = self.dense2(out)

        return out

model = CNN()
model.cuda()
print('[INFO] cnn model ---------------------------------------')
print(model)
# inputs = torch.randn(6,2,64,64)
# # g=make_dot(model(lips,tongue))
# g=make_dot(model(inputs), params=dict(model.named_parameters()))
# g.render(root+'cnn_model', view=False)

#优化和损失函数optimizer and loss function----------
# optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)  #随机梯度下降
optimizer = optim.Adam(model.parameters(), lr=BASE_LR, betas=(0.9, 0.999),eps=1e-08, weight_decay=WEIGHT_DECAY)  # wd正则化
loss_func = nn.MSELoss()      #默认reduce=true返回标量，size_average=true返回loss.mean
# loss_func = nn.BCEWithLogitsLoss()

# # multiple optim
# optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)  # wd正则化
# optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)  #随机梯度下降
# optimizer = [optimizer_Adam, optimizer_SGD]
# loss_func = nn.MSELoss() 
# losses_history = [[],[]]


def main():
    #训练test_train-----------
    # print('[INFO] start training ')
    # train_losses, eval_losses, eval_r2s=[], [], []
    train_losses, eval_losses=[], []
    early_stopping=EarlyStopping(patience=PATIENCE,verbose=True)
    for epoch in range(NUM_EPOCH):
        print('[INFO] start training ')
        model.train() #启用batchnormalization和dropout
        train_loss=0.0
        #step_loss=0.0
        for step, (train_lips, train_tongue, train_label) in enumerate(train_loader):
            train_lips, train_tongue, train_label = Variable(train_lips).cuda(), Variable(train_tongue).cuda(), Variable(train_label).cuda()
            optimizer.zero_grad()     #梯度值初始化为0
            output = model(train_lips, train_tongue)
            loss = loss_func(output,train_label)
            loss.backward()           #反向传播                  
            optimizer.step()          #更新参数
            train_loss += float(loss.item()*train_lips.size(0))
            # print('Epoch:[%d/%d], Step:[%d/%d], Step loss: %.4f' % (epoch + 1, NUM_EPOCH, step + 1, len(train_datasets) // BATCH_SIZE, loss.item()))
            if step%100==99:
                print('Epoch:[%d/%d], Step:[%d/%d], Step loss: %.4f' % (epoch + 1, NUM_EPOCH, step + 1, len(train_datasets) // BATCH_SIZE, loss.item()))
                #print('Epoch:[%d/%d], Step:[%d/%d], Average step loss:%.4f' % (epoch + 1, NUM_EPOCH, step + 1, len(train_datasets) // BATCH_SIZE, step_loss/50))
        train_losses.append(train_loss/len(train_datasets))
        print('=====> Epoch:',epoch+1, ' | Average epoch train loss: %.4f' % (train_loss/len(train_datasets)))
        
        # adjust_lr(optimizer,epoch) 

        #eval-----------
        print('[INFO] start evaluation')
        model.eval() #不启用batchnormalization和dropout
        with torch.no_grad():
            # eval_loss,eval_r2 = 0.0, 0.0
            eval_loss=0.0
            for step,(test_lips, test_tongue, test_label) in enumerate(eval_loader):
                test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
                output = model(test_lips,test_tongue)
                loss = loss_func(output,test_label)
                eval_loss += float(loss.item()*test_lips.size(0))
            eval_losses.append(eval_loss/len(eval_datasets))
            print('=====> Epoch:',epoch+1, ' | Average epoch eval loss: %.4f ' % (eval_loss/len(eval_datasets)))
            #print('=====> Epoch:',epoch+1, ' | Average epoch test loss:%.4f ' % (eval_loss/len(test_datasets)), '| average r2 :%.4f ' % (eval_r2/len(test_datasets)))
            print('[INFO] evaluation complete')

        # # early_stopping(train_loss/len(train_datasets),model)
        # early_stopping(eval_loss/len(test_datasets),model)
        # if early_stopping.early_stop:
        #     print('[INFO] early stop')
        #     break
        
        counter=early_stopping(eval_loss/len(test_datasets),model)
        if early_stopping.early_stop:
            print('[INFO] early stop')
            break

        adjust_lr(optimizer,epoch,eval_losses,counter) 

    return train_losses, eval_losses


def test_model():
    model.load_state_dict(torch.load(root+'checkpoint.pt'))
    print('[INFO] start testing, output predict')
    model.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    # mae, test_mae=0.0, 0.0
    for step,(test_lips, test_tongue, test_label) in enumerate(test_loader):
        test_lips, test_tongue, test_label = Variable(test_lips).cuda(), Variable(test_tongue).cuda(), Variable(test_label).cuda()
        output = model(test_lips, test_tongue)
        loss = loss_func(output,test_label)
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
    train_losses, eval_losses = main()
    # prediction = test_model()

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

    # print('[INFO] save model parameters')
    # torch.save(model.state_dict(),root+'test_model_ssi.pth')  #只保存参数，不保存模型
    # print('[INFO] training complete')

    # #保存模型save model：
    # print('[INFO] save model')
    # torch.save(model,'model_ssi.pth')
    # print('training complete')

    # print('[INFO] save test output')
    # spec = prediction.cpu().detach().numpy()
    # # spec = min_max_scaler2.inverse_transform(spec)
    # np.save(root+"test_predict.npy", spec)
    # print('[INFO] finished')

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))