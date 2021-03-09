import numpy as np
import os
# import cv2
# import librosa
import sys,os
# from tqdm import tqdm

def _preemphasis(y, rate=.95):
	y = np.append(y[0], y[1:]-rate*y[:-1])
	return y

def spectral_distortion(lsf_true, lsf_pred, N, n0, n1):
    SD = []
    IS16SD = []
    print(len(lsf_true))
    print(len(lsf_pred))
    for frameid in range(len(lsf_true)):
        lpc_true = lpd.lsf2poly(lsf_true[frameid])
        lpc_pred = lpd.lsf2poly(lsf_pred[frameid])

        _, freqResponse_true = signal.freqz(b=1, a=lpc_true, worN=N)
        _, freqResponse_pred = signal.freqz(b=1, a=lpc_pred, worN=N)

        freq_th = freqResponse_true[n0 - 1:n1]
        freq_pred = freqResponse_pred[n0 - 1:n1]

        absoluteRadio = (freq_th.real ** 2 + freq_th.imag ** 2) ** 0.5 / (
                    freq_pred.real ** 2 + freq_pred.imag ** 2) ** 0.5

        logValue = np.log10(absoluteRadio ** 2)
        bigsum = ((10 * logValue) ** 2).sum()
        sd = math.sqrt(1.0 / (n1 - n0)) * bigsum
        IS16sd = math.sqrt(1.0 / (n1 - n0) * bigsum)
        SD.append(sd)
        IS16SD.append(IS16sd)

    return SD, IS16SD, sum(SD) * 1.0 / len(SD), sum(IS16SD) * 1.0 / len(IS16SD)

def load_dataset(dir_path="../data/new_database/"):
	if os.path.exists(dir_path):
		arr = []
		for num in enumerate(['train_lips', 'train_tongue', 'test_lips', 'test_tongue', 'train_label', 'test_label']):
			path = "../data/new_database/%s.npy"%num[1]
			arr.append(np.load(path))
		train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = arr[0], arr[2], arr[1], arr[3], arr[4], arr[5]

	else:
		print("Processing original data...")

		# train_label, test_label = _generate_audio_mel_Lable()

	return train_lips, test_lips, train_tongue, test_tongue, train_label, test_label

def lips_tongue():
	print("Processing original data...")
	lips = []
	tongue = []
	lips_fdir = "./ssi/data/new_data/new_out/resize_lips/lipster%s.bmp"
	tong_fdir = "./ssi/data/new_data/new_out/resize_tongue/tonguester%s.bmp"
	for i in tqdm(range(0, 37905+1)):
		ran = str(i)
		for i in range(6 - len(ran)):
			ran = '0' + ran
		lframe_0 = cv2.imread(lips_fdir % ran, 0)
		lframe_0 = np.array(lframe_0)
		lips.append(lframe_0.reshape((lframe_0.shape[0], lframe_0.shape[1], 1)))

		tframe_0 = cv2.imread(tong_fdir % ran, 0)
		tframe_0 = np.array(tframe_0)
		tongue.append(tframe_0.reshape((tframe_0.shape[0], tframe_0.shape[1], 1)))

	train_lips = np.array(lips[:33000])
	train_tongue = np.array(tongue[:33000])
	test_lips = np.array(lips[33000:])
	test_tongue = np.array(tongue[33000:])

	np.save("./ssi/data/new_data/new_database/train_lips.npy", train_lips)
	np.save("./ssi/data/new_data/new_database/train_tongue.npy", train_tongue)
	np.save("./ssi/data/new_data/new_database/test_lips.npy", test_lips)
	np.save("./ssi/data/new_data/new_database/test_tongue.npy", test_tongue)

	return train_lips, test_lips, train_tongue, test_tongue


def _generate_audio_mel_Lable(dir_path = './ssi/data/new_data/new_audio/audio_denoise.wav'):
	ref_db=0 #max
	max_db=80 #min  #这里
	# max_db=0
	# min_db=-80
	magnitude = []
	i=13  #这里
	_y, sr = librosa.load(dir_path, sr=44100) #(30187008,)
	_y = _preemphasis(_y[:37906*746], 0.95)
	print(_y.shape)
	# _y=librosa.effects.preemphasis(_y, coef=0.95)
	melspec = librosa.feature.melspectrogram(_y,sr=sr,n_fft=746*i, hop_length=746, power=2, n_mels=128) #(64, 68146) 这里
	print(melspec.shape)
	# logmelspec = librosa.amplitude_to_db(melspec[:,6:-6]) #(64, 68135) #20∗log10(Sref)
	# print(logmelspec.max())
	# print(logmelspec.min())
	# logmelspec = np.clip((logmelspec - ref_db + max_db) / max_db, 1e-8, 1)  #power=1,energie
	# magnitude.extend(logmelspec.T)

	melspec=melspec[:,6:-6]
	ref_melspec = melspec.max()
	print(ref_melspec)
	powmelspec = librosa.power_to_db(melspec,ref=np.max) #68146-i+1    #Compute dB relative to peak power
	print(powmelspec.max())
	print(powmelspec.min())
	powmelspec = np.clip((powmelspec - ref_db + max_db) / max_db, 1e-8, 1)
	print(powmelspec.max())
	print(powmelspec.min())
	# powmelspec = np.clip((powmelspec - min_db) / (max_db - min_db), 1e-8, 1)
	magnitude.extend(powmelspec.T)
	print(len(magnitude)) #68146-i+1

	print(':::::')
	train_label, test_label = [], []
	train_label.extend(magnitude[:33000-i+1]) #61331-i+1
	test_label.extend(magnitude[33000:]) #6815-i+1
	print(len(train_label))
	print(len(test_label))

	train_label = np.array(train_label)
	test_label = np.array(test_label) 

	np.save("./ssi/data/new_data/new_database/0528newdenoise_13ppc_128ref.npy", ref_melspec) #这里
	np.save("./ssi/data/new_data/new_database/0528newdenoise_13ppc_128train_label.npy", train_label) #这里
	np.save("./ssi/data/new_data/new_database/0528newdenoise_13ppc_128test_label.npy", test_label)

	return train_label, test_label

def img_train_normalize(image_data):
	l = image_data.shape[0]
	img = np.zeros((64,64,1))
	for i in range(l):
		img += image_data[i]
	img_m = np.mean(img/l)
	img_std = np.std(img/l)
	img_data = []
	for i in range(l):
		tmp = (image_data[i] - img_m)/img_std
		img_data.append(tmp)
	img_data = np.array(img_data)

	return img_data

def target_preprocessing(label):
	l = label.shape[0]
	targets = []
	for i in range(l):
		data = np.reshape(label[i], -1)
		
		targets.append(data)
	targets = np.array(targets)

	return targets

def label_normalize(label_data):
	l = label_data.shape[0]
	label = np.zeros(1025)
	# label = np.zeros((3,1025))
	for i in range(l):
		label += label_data[i]
	label_m = np.mean(label/l)
	label_std = np.std(label/l)
	lab_data = []
	for i in range(l):
		tmp = (label_data[i] - label_m)/label_std
		lab_data.append(tmp)
	lab_data = np.array(lab_data)

	return lab_data

def label_normalize2(label_data):
	l = label_data.shape[0]
	label = np.zeros(1025)
	# label = np.zeros((3,1025))
	for i in range(l):
		label += label_data[i]
	label_m = np.mean(label/l)
	label_std = np.std(label/l)
	lab_data = []
	for i in range(l):
		tmp = (label_data[i] - label_m)/label_std
		lab_data.append(tmp)
	lab_data = np.array(lab_data)

	return lab_data, label_m, label_std

def label_normalize_inverse(label_data,label_m,label_std):
	l = label_data.shape[0]
	label = np.zeros((3,1025))
	for i in range(l):
		label += label_data[i]
	lab_data = []
	for i in range(l):
		# tmp = (label_data[i] - label_m)/label_std
		tmp = label_data[i]*label_std+label_m
		lab_data.append(tmp)
	lab_data = np.array(lab_data)

	return lab_data

# if __name__ == '__main__':
# 	# train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = load_dataset()
# 	# print(train_lips.shape)
# 	# print(train_tongue.shape)
# 	# print(test_lips.shape)
# 	# print(test_tongue.shape)
# 	# print(train_label.shape)
# 	# print(test_label.shape)
# 	# train_lips= img_train_normalize(train_lips)
# 	# # train_tongue, mean_tongue = imgdata_preprocessing(train_tongue)

# 	# train_lips, test_lips, train_tongue, test_tongue = lips_tongue()
# 	# print(train_lips.shape)	#33000
# 	# print(train_tongue.shape)
# 	# print(test_lips.shape)
# 	# print(test_tongue.shape) #4906
# 	train_label, test_label = _generate_audio_mel_Lable('./ssi/data/new_data/new_audio/audio_denoise.wav')
# 	print(train_label.shape) #(32988, 128)
# 	print(test_label.shape) #(4894, 128)
