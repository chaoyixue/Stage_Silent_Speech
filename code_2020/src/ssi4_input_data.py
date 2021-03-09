import numpy as np
import os
import cv2
# import librosa
import sys,os

def _get_frames_data(file_path):
	frame_data = []
	for root, dirnames, filenames in os.walk(file_path):
		for i in range(len(filenames)):
			image_name = file_path+'/'+str(filenames[i])
			img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
			# img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)  #activer seulement pour produire les validation exemples
			img = np.array(img)
			frame_data.append(img.reshape((img.shape[0], img.shape[1], 1)))
	return frame_data

def _preemphasis(y, rate=.95):
	y = np.append(y[0], y[1:]-rate*y[:-1])
	return y

def _get_spectrograms(y, sr=44100, n_fft=2048, win_length=1024, hop_length=512):
	linear = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
	mag = np.abs(linear)  # magnitude
	return mag

def _get_melspectrograms(y,sr,n_fft,n_mels,max_db,ref_db):
	# mel = librosa.feature.melspectrogram(_y,sr=sr,n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels) #从audio提取！！！！
	mel0 = librosa.filters.mel(sr, n_fft, n_mels)
	mel = 20 * np.log10(np.maximum(1e-5, np.dot(mel0, y)))# to decibel
	mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1) # normalize，clip函数将大于max改为max，小于min改为min
	mel = mel.T.astype(np.float32)  # (T, n_mels) # Transpose 
	return mel

def label_feature(label, step=3):
	labels = []
	feature_frame = []
	for i in range(0,label.shape[0],step):
		feature_frame = np.vstack((label[i], label[i+1]))
		feature_frame = np.vstack((feature_frame, label[i+2]))
		labels.append(feature_frame)
	labels = np.array(labels)
	return labels 

def load_dataset(dir_path="../data/database/"):
	if os.path.exists(dir_path):
		arr = []
		for num in enumerate(['train_lips', 'train_tongue', 'test_lips', 'test_tongue', 'train_label', 'test_label']):
			path = "../data/database/%s.npy"%num[1]
			arr.append(np.load(path))
		train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = arr[0], arr[2], arr[1], arr[3], arr[4], arr[5]

	else:
		print("Processing original data...")
		os.mkdir(dir_path)
		path_to_lips = "./ssi/data/out/resize_lips"
		path_to_tongue = "./ssi/data/out/resize_tongue"
		lips_data = _get_frames_data(path_to_lips)
		tongue_data = _get_frames_data(path_to_tongue)
		train_lips, test_lips, train_tongue, test_tongue = [], [], [], []
		train_lips.extend(lips_data[:61331])
		train_tongue.extend(tongue_data[:61331])
		test_lips.extend(lips_data[61330:])
		test_tongue.extend(tongue_data[61330:])

		train_lips = np.array(train_lips, dtype=int)
		train_tongue = np.array(train_tongue, dtype=int)
		test_lips = np.array(test_lips, dtype=int)
		test_tongue = np.array(test_tongue, dtype=int)

		np.save("./ssi/data/database/train_lips.npy", train_lips)
		np.save("./ssi/data/database/train_tongue.npy", train_tongue)
		np.save("./ssi/data/database/test_lips.npy", test_lips)
		np.save("./ssi/data/database/test_tongue.npy", test_tongue)

		train_label, test_label = _generate_audio_mel_Lable()

	return train_lips, test_lips, train_tongue, test_tongue, train_label, test_label

def _generate_audio_mel_Lable(dir_path = './ssi/data/Songs_Audio_matched/out01234.wav'):
	step = 1
	magnitude = []
	_y, sr = librosa.load(dir_path, sr=44100) #(50087310,)
	melspec = librosa.feature.melspectrogram(_y,sr=sr,n_fft=2048, hop_length=735, win_length=1470, n_mels=64) #(64, 68147)
	logmelspec = librosa.amplitude_to_db(melspec[:,1:-1],ref=np.max) #(64, 68145)
	magnitude.extend(logmelspec.T)
	print(len(magnitude)) #50087310/hop+1

	print(':::::')
	train_label, test_label = [], []
	train_label.extend(magnitude[:61330])
	test_label.extend(magnitude[61330:])
	print(len(train_label))
	print(len(test_label))

	train_label = np.array(train_label)	#(61330, 64)
	test_label = np.array(test_label) #(6815, 64)

	np.save("./ssi/data/database/train_label.npy", train_label)
	np.save("./ssi/data/database/test_label.npy", test_label)

	return train_label, test_label

def _generate_audio_Lable(dir_path = './ssi/data/Songs_Audio_matched/'):
	#sr=8820
	sr=44100
	preempha=0.95
	n_fft=2048 #513
	win_length=735
	hop_length=247
	max_db=100
	ref_db=0
	# hop_length=245
	step = int(win_length/hop_length)
	magnitude = []
	for i in range(5):
		file = dir_path + 'out%d.wav'%i
		_y, sr = librosa.load(file, sr=sr, mono=1)
		_y = _preemphasis(_y, preempha)
		_mag = _get_spectrograms(_y, sr, n_fft, win_length, hop_length)
		# print(_mag.shape)   #(1025, ~)
		_mag_D = librosa.amplitude_to_db(_mag, ref=np.max)
		_mag_D = np.clip((_mag_D - ref_db + max_db) / max_db, 1e-8, 1)
		magnitude.extend(_mag_D.T)
		# print(np.array(magnitude).shape)  #(~, 1025)

	cliprate = 0.1
	length = np.array([10667790, 7055265, 13640865, 5317725, 13405665])   # 44.1KHz
	# length = np.array([7055265, 10667056, 13640866, 5317726, 13406400])   # 44.1KHz
	#length = np.array([10669995, 7050120, 13645275, 5311110, 13410810])   # 44.1KHz
	#length = np.array([2133999, 1410024, 2729055, 1062222, 2682162])   # 8820Hz
	#length = (length / 44100 * 60)
	length = (length / sr * 60)
	test = length * cliprate
	start_test = (length.cumsum() - test).astype(int)*step    #[13065 23149 40817 49177 66321]*3未修改
	end_test = (length.cumsum()).astype(int)*step    #[14517 24109 42674 49900 68146]*3未修改
	train_label, test_label = [], []

	for i in range(len(start_test)):
		test_label.extend(magnitude[start_test[i]:end_test[i]])

		if i == 0:
			first = 0
		else:
			first = end_test[i-1]
		train_label.extend(magnitude[first:start_test[i]])

	train_label = np.array(train_label)	#(183990, 1025)
	test_label = np.array(test_label) #(20448, 1025)

	# train_label = label_feature(train_label)
	# test_label = label_feature(test_label)

	np.save("./ssi/data/database/train_label.npy", train_label)
	np.save("./ssi/data/database/test_label.npy", test_label)

	return train_label, test_label

def _generate_audio_mel_Lable1(dir_path = './ssi/data/Songs_Audio_matched/out01234.wav'):
	sr=44100
	preempha=0.95
	n_fft=2048
	win_length = 1470
	hop_length=735
	step = 1
	n_mels=64
	max_db = 100
	ref_db = 0
	magnitude = []
	_y, sr = librosa.load(dir_path, sr=sr)
	# melspec = librosa.feature.melspectrogram(_y,sr=sr,n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
	# logmelspec = librosa.amplitude_to_db(melspec)
	# magnitude.extend(logmelspec.T)
	print(_y.shape)
	_y = _preemphasis(_y, preempha)
	_mag = _get_spectrograms(_y, sr, n_fft, win_length, hop_length)
	mel = _get_melspectrograms(_mag,sr=sr,n_fft=n_fft,n_mels=n_mels,max_db=max_db,ref_db=ref_db)
	magnitude.extend(mel[1:-1])
	print(mel.shape)
	print(len(magnitude)) #50087310/hop+1

	print(':::::')
	train_label, test_label = [], []
	train_label.extend(magnitude[:61330])
	test_label.extend(magnitude[61330:])
	print(len(train_label))
	print(len(test_label))

	train_label = np.array(train_label)	#(61330, 64)
	test_label = np.array(test_label) #(6816, 64)

	# train_label = label_feature(train_label)
	# test_label = label_feature(test_label)

	np.save("./ssi/data/database/train_label.npy", train_label)
	np.save("./ssi/data/database/test_label.npy", test_label)

	return train_label, test_label

def _generate_audio_mel_Lable2(dir_path = './ssi/data/Songs_Audio_matched/'):
	sr=44100
	preempha=0.95
	n_fft=1470
	win_length=1470
	hop_length=735
	# step = int(win_length/hop_length)
	step = 1
	n_mels=64
	max_db = 100
	ref_db = 0
	magnitude = []
	for i in range(5):
		file = dir_path + 'out%d.wav'%i
		_y, sr = librosa.load(file, sr=sr, mono=1)
        # melspec = librosa.feature.melspectrogram(_y,sr=sr,n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
        # logmelspec = librosa.amplitude_to_db(melspec)
        # magnitude.extend(logmelspec.T)

		_y = _preemphasis(_y, preempha)
		_mag = _get_spectrograms(_y, sr, n_fft, win_length,hop_length)
		mel = _get_melspectrograms(_mag,sr=sr,n_fft=n_fft,n_mels=n_mels,max_db=max_db,ref_db=ref_db)
		magnitude.extend(mel[:-1])
		print(mel.shape)
		print(len(magnitude))  
	print(len(magnitude)) #50087313/hop+1

	cliprate = 0.1
	length = np.array([10667790, 7055265, 13640865, 5317725, 13405665])   # 44.1KHz
	#length = np.array([10669995, 7050120, 13645275, 5311110, 13410810])   # 44.1KHz
	#length = np.array([2133999, 1410024, 2729055, 1062222, 2682162])   # 8820Hz
	#length = (length / 44100 * 60)
	length = (length / sr * 60)
	test = length * cliprate
	start_test = (length.cumsum() - test).astype(int)*step  #[13062 23153 40816 49183 66322]
	end_test = (length.cumsum()).astype(int)*step #[14514 24113 42672 49907 68146]
	train_label, test_label = [], []
	print(start_test)
	print(end_test)

	for i in range(len(start_test)):
		test_label.extend(magnitude[start_test[i]:end_test[i]])
		if i == 0:
			first = 0
		else:
			first = end_test[i-1]
		train_label.extend(magnitude[first:start_test[i]])
		print(len(test_label))

	train_label = np.array(train_label)	#(61330, 64)
	test_label = np.array(test_label) #(6816, 64)

	# train_label = label_feature(train_label)
	# test_label = label_feature(test_label)

	np.save("./ssi/data/database/train_label.npy", train_label)
	np.save("./ssi/data/database/test_label.npy", test_label)

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
	# train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = load_dataset()
	# print(train_lips.shape)
	# print(train_tongue.shape)
	# print(test_lips.shape)
	# print(test_tongue.shape)
	# print(train_label.shape)
	# print(test_label.shape)
	# train_lips= img_train_normalize(train_lips)
	# # train_tongue, mean_tongue = imgdata_preprocessing(train_tongue)

	##########注意以下两函数勿同时使用############
	# train_label, test_label = _generate_audio_Lable("./ssi/data/Songs_Audio_matched/")
	# train_label, test_label = _generate_audio_mel_Lable('./ssi/data/Songs_Audio_matched/out01234.wav')
	# print(train_label.shape)
	# print(test_label.shape)

	# val_lips = _get_frames_data("../data/VowelsData/LipsSeq")
	# val_tongue = _get_frames_data("../data/VowelsData/TongueSeq")
	# val_lips = np.array(val_lips)
	# val_tongue = np.array(val_tongue)
	# print(val_lips.shape)
	# print(val_tongue.shape)
	# np.save("../data/data_val/val_lips.npy", val_lips)
	# np.save("../data/data_val/val_tongue.npy", val_tongue)







