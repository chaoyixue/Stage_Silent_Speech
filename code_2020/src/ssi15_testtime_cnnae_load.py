import cv2,os,sys,time,copy,torch,librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import librosa.display
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from scipy import signal
# from sklearn.metrics import r2_score, mean_absolute_error
from ssi6_train0723_parole4_cnn2d import CNN, match_image_label
from ssi6_train0806_parole4_ae import AutoEncoder
from ssi11_input_data_new import *
import matplotlib.pyplot as plt
import IPython.display as ipd

# print(librosa.__version__)
i=8
hop=735
i_spec=2
BATCH_SIZE = 32
MOMENTUM=0.9
ref_spec=np.load('../data/new_database/0624specparole4sr_np2ppc735_refspec.npy')
root1='../data/new_database/testtime/'
root2='../out/0806testtime_'

def TestDatasets():
    print('[INFO] set datasets')
    _, test_lips, _, test_tongue, _, _ = load_dataset()
    # test_lips, test_tongue = input_image()
    test_lips = test_lips[400:800,:,:,:]
    test_tongue = test_tongue[400:800,:,:,:]
    
    test_lips = match_image_label(test_lips)
    test_tongue = match_image_label(test_tongue)

    test_lips = torch.from_numpy(test_lips).float()
    test_tongue = torch.from_numpy(test_tongue).float()

    test_lips = test_lips.permute(0,3,1,2)
    test_tongue = test_tongue.permute(0,3,1,2)

    test_datasets = TensorDataset(test_lips, test_tongue)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)

    return test_datasets, test_loader

def TestDatasets_ae(spec):
    test_label = torch.from_numpy(spec).float().unsqueeze(1)
    test_label2 = torch.from_numpy(spec).float().unsqueeze(1)
    # test_label = spec.unsqueeze(1)
    # test_label2 = spec.unsqueeze(1)
    test_datasets = TensorDataset(test_label,test_label2)
    test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False)
    return test_datasets, test_loader

def test_model(test_datasets, test_loader):
    print('[INFO] start testing')
    model.eval()
    for step,(test_lips, test_tongue) in enumerate(test_loader):
        test_lips, test_tongue= Variable(test_lips).cuda(), Variable(test_tongue).cuda()
        output = model(test_lips, test_tongue)
        if step==0:
            prediction=output
        else:
            prediction=torch.cat((prediction,output),0) #按行竖着接
    print('[INFO] test complete')

    return prediction

def test_autoencoder(test_datasets, test_loader):
    model2.eval()
    for step,(test_label,_) in enumerate(test_loader):
        test_label = Variable(test_label).cuda()
        output = model2(test_label)
        if step==0:
            prediction=output
        else:
            prediction=torch.cat((prediction,output),0) #按行竖着接
    print('[INFO] test complete')

    return prediction
    
if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    # model=CNN()
    # model.cuda()
    # model.load_state_dict(torch.load('../out/checkpoint.pt'))
    # test_datasets, test_loader = TestDatasets()
    # prediction = test_model(test_datasets, test_loader)
    # spec = prediction.cpu().detach().numpy()
    # np.save(root2+"test_predict_me.npy", spec)

    spec=np.load('../out/test_predict_me3.npy')
    prediction = spec

    model2=AutoEncoder()
    model2.cuda()
    model2.load_state_dict(torch.load('../out/ae_checkpoint.pt'))
    test_datasets2, test_loader2 = TestDatasets_ae(prediction)
    prediction2 = test_autoencoder(test_datasets2, test_loader2)
    spec2 = prediction2.cpu().detach().numpy()
    spec2 = np.squeeze(spec2,axis=1)
    np.save(root2+"test_predict_me_ae.npy", spec2)

    print(spec.shape,spec2.shape)

    print('[INFO] save spectrogram picture')
    plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1)
    librosa.display.specshow(spec.T,sr =44100, hop_length=hop, x_axis='s', y_axis='linear')
    plt.title('spectrogram (predict)')
    plt.subplot(2,1,2)
    librosa.display.specshow(spec2.T,sr =44100, hop_length=hop, x_axis='s', y_axis='linear')
    plt.title('spectrogram (predict ae)')
    plt.savefig(root2+'spec_nor_linear.png')

    ref_db,max_db=0,80
    spec = (np.clip(spec, 1e-8, 1) * max_db) - max_db + ref_db # de-norm
    spec2 = (np.clip(spec2, 1e-8, 1) * max_db) - max_db + ref_db # de-norm

    print('[INFO] spec to wav')
    s = librosa.db_to_amplitude(spec.T, ref=ref_spec)
    _wav = librosa.griffinlim(s,win_length=hop*i_spec, hop_length=hop,n_iter=100)
    # _wav = signal.lfilter([1], [1, -0.95], _wav)

    s2 = librosa.db_to_amplitude(spec2.T, ref=ref_spec)
    _wav2 = librosa.griffinlim(s2,win_length=hop*i_spec, hop_length=hop,n_iter=100)

    plt.figure(figsize=(15, 10))
    plt.subplot(2,1,1)
    librosa.display.waveplot(_wav, sr=44100,x_axis= 's')
    plt.title('predict')
    plt.subplot(2,1,2)
    librosa.display.waveplot(_wav2, sr=44100,x_axis= 's')
    plt.title('predict ae')
    plt.savefig(root2+'wav.png')
    librosa.output.write_wav(root2+"predict.wav", _wav, sr=44100)
    librosa.output.write_wav(root2+"predict_ae.wav", _wav2, sr=44100)
    # ipd.Audio(_wav,rate=44100)

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))