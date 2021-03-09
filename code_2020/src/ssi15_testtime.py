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
root2='../out/0803testtime_'

# def count_image(image_path):
#     count = 0
#     for root,dirs,files in os.walk(image_path):
#         for each in files:
#             count += 1 
#     return count

def input_image():
    print('[INFO] image preprocessing')
    test_lips,test_tongue=[],[]
    # count_lips = count_image(root1+"lipster/")
    # count_tongue = count_image(root1+"tonguester/")
    # num=min(count_lips,count_tongue)
    b_num=82100
    num=400
    for i in tqdm(range(b_num,b_num+num)):
        name = str(i)
        for i in range(6-len(name)):
            name = "0"+name
        impath = root1+"lipster/%s.bmp" % name
        lips = cv2.imread(impath)
        lips = lips[:400,100:660,:] #高：宽：rgb  parole4 ROI
        lips = cv2.resize(lips,(64,64),interpolation=cv2.INTER_AREA)
        lips = cv2.cvtColor(lips,cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(root1+"lipster%s.bmp" % name, lips)
        lips = np.array(lips)
        test_lips.append(lips.reshape((lips.shape[0], lips.shape[1], 1)))


        impath = root1+"tonguester/%s.bmp" % name
        tongue = cv2.imread(impath)
        tongue = tongue[40:200,70:250,:] #parole4 ROI
        tongue = cv2.resize(tongue,(64,64),interpolation=cv2.INTER_AREA)
        tongue = cv2.cvtColor(tongue,cv2.COLOR_BGR2GRAY)
        tongue = np.array(tongue)
        test_tongue.append(tongue.reshape((tongue.shape[0], tongue.shape[1], 1)))

    test_lips = np.array(test_lips)
    test_tongue = np.array(test_tongue)

    return test_lips, test_tongue

def TestDatasets():
    print('[INFO] set datasets')
    test_lips, test_tongue = input_image()

    test_lips = match_image_label(test_lips)
    test_tongue = match_image_label(test_tongue)

    test_lips = torch.from_numpy(test_lips).float()
    test_tongue = torch.from_numpy(test_tongue).float()

    test_lips = test_lips.permute(0,3,1,2)
    test_tongue = test_tongue.permute(0,3,1,2)

    test_datasets = TensorDataset(test_lips, test_tongue)
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

if __name__ == "__main__":
    start=time.perf_counter()
    print("[INFO] Load model")
    model=CNN()
    model.cuda()
    model.load_state_dict(torch.load('../out/checkpoint.pt'))

    test_datasets, test_loader = TestDatasets()
    # print('[INFO] begin test')
    prediction = test_model(test_datasets, test_loader)
    spec = prediction.cpu().detach().numpy()
    np.save(root2+"test_predict_me.npy", spec)

    print('[INFO] save spectrogram picture')
    plt.figure(figsize=(15, 6))
    librosa.display.specshow(spec.T,sr =44100, hop_length=hop, x_axis='s', y_axis='linear')
    plt.title('spectrogram (predict)')
    plt.savefig(root2+'spec_nor_linear.png')

    ref_db,max_db=0,80
    spec = (np.clip(spec, 1e-8, 1) * max_db) - max_db + ref_db # de-norm

    print('[INFO] spec to wav')
    s = librosa.db_to_amplitude(spec.T, ref=ref_spec)
    _wav = librosa.griffinlim(s,win_length=hop*i_spec, hop_length=hop,n_iter=100)
    # _wav = signal.lfilter([1], [1, -0.95], _wav)

    plt.figure(figsize=(15, 6))
    librosa.display.waveplot(_wav, sr=44100,x_axis= 's')
    plt.title('predict')
    plt.savefig(root2+'wav.png')
    librosa.output.write_wav(root2+"predict.wav", _wav, sr=44100)
    # ipd.Audio(_wav,rate=44100)

    end=time.perf_counter()
    print('[INFO] running time: %.4s seconds' %(end-start))