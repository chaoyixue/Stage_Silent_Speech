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
from pytorchtools import EarlyStopping
# from torchvision.models import AlexNet
# from torchviz import make_dot
# from torchvision import models

root="../out/"

BATCH_SIZE = 32
BASE_LR= 1e-4
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
		for num in enumerate(['train_lsf', 'train_f0', 'test_lsf', 'test_f0', 'train_label', 'test_label', 'train_uv', 'test_uv',]):
			path = "../data/new_database/%s.npy"%num[1]
			arr.append(np.load(path))
		train_lsf, test_lsf, train_f0, test_f0, train_label, test_label, train_uv, test_uv = arr[0], arr[2], arr[1], arr[3], arr[4], arr[5], arr[6], arr[7]
	else:
		print("Error")
	return train_lsf, test_lsf, train_f0, test_f0, train_label, test_label, train_uv, test_uv

#数据输入input data----------
def match_image_label(image_data): #2D
    l=image_data.shape[1]
    image_match=[]
    for m in range(l-i+1):
        image_con=np.concatenate((image_data[:,m:m+i,:]),axis=-1) #(batch_size, 64, 64, 6)
        image_match.append(image_con)
    image_match = np.array(image_match)
    
    return image_match

def SSIDatasets():
    print('[INFO] -------------------------------------------------')
    print('[INFO] set datasets')
    train_lsf, test_lsf, train_f0, test_f0, train_label, test_label, train_uv, test_uv = load_dataset()
    # print(train_lsf.shape, test_lsf.shape, train_f0.shape, test_f0.shape, train_label.shape, test_label.shape, train_uv.shape, test_uv.shape )

    #preprocessing
    train_lsf = match_image_label(train_lsf)
    test_lsf = match_image_label(test_lsf)
    train_f0=train_f0[:-1]
    test_f0=test_f0[:-1]
    train_uv=train_uv[:-1]
    test_uv=test_uv[:-1]

    #to torch.tensor
    train_lsf = torch.from_numpy(train_lsf).float()
    test_lsf = torch.from_numpy(test_lsf).float()
    train_f0 = torch.from_numpy(train_f0).float()
    test_f0 = torch.from_numpy(test_f0).float()
    train_uv = torch.from_numpy(train_uv).float()
    test_uv = torch.from_numpy(test_uv).float()
    train_label = torch.from_numpy(train_label).float()
    test_label = torch.from_numpy(test_label).float()

    #set datasets and dataloader
    train_datasets = TensorDataset(train_lsf, train_f0, train_uv, train_label)
    train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
    eval_datasets = TensorDataset(test_lsf, test_f0, test_uv, test_label)
    eval_loader = DataLoader(dataset=eval_datasets, batch_size=BATCH_SIZE, shuffle=True)
    test_datasets = TensorDataset(test_lsf, test_f0, test_uv, test_label)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return train_datasets, train_loader, eval_datasets, eval_loader, test_datasets, test_loader

#调整lr，adjust lr-----------
def adjust_lr(optimizer,epoch):
    if (epoch+1)%10==0:
        for param_group in optimizer.param_groups:
            param_group['lr']=param_group['lr']*0.1 #每10个epoch lr*0.1
# def adjust_lr(optimizer,counter):
#     if counter>=3:
#         for param_group in optimizer.param_groups:
#             param_group['lr']=param_group['lr']*0.5
#             print('[INFO] lr:',param_group['lr'])
#     else:
#         pass

class CNN(nn.Module):
    def cnn1d(self,in_c, out_c,k_s=3,padd=1):
        conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k_s, padding=padd, bias=True),#（2*64*64) 若在卷积后加bn，最好bias=False
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=k_s, padding=padd, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(2))   #（16*32*32）
            # nn.BatchNorm1d(out_c))
            # nn.Dropout(DROPOUT))
        return conv

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = self.cnn1d(i,16)
        self.conv2 = self.cnn1d(16,32)
        self.bn = nn.BatchNorm1d(98)

        self.dense1 = nn.Sequential(
            nn.Linear(98, 736), # 卷积核3*3*16， 64-3+1=62， 输出62*62*16
            nn.ReLU())
            # nn.Dropout(DROPOUT))
        # self.dense2 = nn.Sequential(
        #     nn.Linear(8192, 184*32), # 卷积核3*3*16， 64-3+1=62， 输出62*62*16
        #     nn.ReLU())
            # nn.LeakyReLU())
            # nn.Sigmoid())# 旧，jyr
            # nn.Softmax()#分类
            # nn.Dropout(DROPOUT))

    def forward(self, lsf, f0, uv):
        out1 = self.conv1(lsf)                
        out1 = self.conv2(out1)      #([32, 32, 3])          
        out1 = out1.view(out1.size(0),-1)       # torch.Size([32, 96])
        out = torch.cat((out1, f0, uv),dim=1) #torch.Size([32, 98])
        out = self.bn(out) #torch.Size([32, 98])
        out = self.dense1(out)

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
# loss_func2 = nn.MSELoss()      #默认reduce=true返回标量，size_average=true返回loss.mean
# loss_func3 = nn.BCEWithLogitsLoss()

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
        for step, (train_lsf, train_f0, train_uv, train_label) in enumerate(train_loader):
            train_lsf, train_f0, train_uv, train_label = Variable(train_lsf).cuda(), Variable(train_f0).cuda(), Variable(train_uv).cuda(),Variable(train_label).cuda()
            optimizer.zero_grad()     #梯度值初始化为0
            output = model(train_lsf, train_f0, train_uv)
            loss = loss_func(output,train_label)
            loss.backward()           #反向传播                  
            optimizer.step()          #更新参数
            train_loss += float(loss.item()*train_lsf.size(0))
            if step%100==99:
                print('Epoch:[%d/%d], Step:[%d/%d], Step loss: %.8f' % (epoch + 1, NUM_EPOCH, step + 1, len(train_datasets) // BATCH_SIZE, loss.item()))
        train_losses.append(train_loss/len(train_datasets))
        print('=====> Epoch:',epoch+1, ' | Average epoch train loss: %.8f' % (train_loss/len(train_datasets)))
        
        adjust_lr(optimizer,epoch) 

        #eval-----------
        print('[INFO] start evaluation')
        model.eval() #不启用batchnormalization和dropout
        with torch.no_grad():
            eval_loss=0.0
            for step,(test_lsf, test_f0, test_uv, test_label) in enumerate(eval_loader):
                test_lsf, test_f0, test_uv, test_label = Variable(test_lsf).cuda(), Variable(test_f0).cuda(), Variable(test_uv).cuda(), Variable(test_label).cuda()
                output = model(test_lsf, test_f0, test_uv)
                loss = loss_func(output,test_label)
                eval_loss += float(loss.item()*test_lsf.size(0))
            eval_losses.append(eval_loss/len(eval_datasets))
            print('=====> Epoch:',epoch+1, ' | Average epoch eval loss: %.8f ' % (eval_loss/len(eval_datasets)))
            print('[INFO] evaluation complete')

        # # early_stopping(train_loss/len(train_datasets),model)
        early_stopping(eval_loss/len(test_datasets),model)
        if early_stopping.early_stop:
            print('[INFO] early stop')
            break

        # early_stopping(eval_loss/len(test_datasets),model)
        # adjust_lr(optimizer,early_stopping.counter) 
        # if early_stopping.early_stop:
        #     print('[INFO] early stop')
        #     break
        
    return train_losses, eval_losses


def test_model():
    model.load_state_dict(torch.load(root+'checkpoint.pt'))
    print('[INFO] start testing, output predict')
    model.eval() #不启用batchnormalization和dropout
    test_loss=0.0
    # mae, test_mae=0.0, 0.0
    for step,(test_lsf, test_f0, test_uv, test_label) in enumerate(test_loader):
        test_lsf, test_f0, test_uv, test_label = Variable(test_lsf).cuda(), Variable(test_f0).cuda(), Variable(test_uv).cuda(), Variable(test_label).cuda()
        output = model(test_lsf, test_f0, test_uv)
        loss = loss_func(output,test_label)
        test_loss += float(loss.item()*test_lsf.size(0))
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