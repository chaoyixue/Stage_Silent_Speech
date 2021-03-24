import numpy as np
import librosa
import sys,os, cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display

# prepare datasets (local)

root1 = './ssi/data/database/'
root2 = './ssi/data/Songs_Audio_matched/Songs_Audio_matched/out01234.wav'   # audio


def load_dataset(dir_path=root1):
	if os.path.exists(dir_path):
		arr = []
		for num in enumerate(['train_lips', 'train_tongue', 'test_lips', 'test_tongue', 'train_label', 'test_label']):
			path = root1+"%s.npy" % num[1]
			arr.append(np.load(path))
		train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = arr[0], arr[2], arr[1], arr[3], arr[4], arr[5]

	else:
		train_lips, test_lips, train_tongue, test_tongue = lips_tongue()
		train_label, test_label = _generate_audio_Lable()

	return train_lips, test_lips, train_tongue, test_tongue, train_label, test_label


def lips_tongue():
	print("Processing original data...")
	lips = []
	tongue = []
	lips_fdir = "./ssi/data/out/resize_lips/%s.tif"
	tong_fdir = "./ssi/data/out/resize_tongue/%s.tif"
	for i in tqdm(range(1, 68147)):  # parole4 : (84679+1) , parole5 : (147923+1)
		ran = str(i)
		for i in range(6 - len(ran)):
			ran = '0' + ran
		lframe_0 = cv2.imread(lips_fdir % ran, 0)
		lframe_0 = np.array(lframe_0)
		lips.append(lframe_0.reshape((lframe_0.shape[0], lframe_0.shape[1], 1)))

		tframe_0 = cv2.imread(tong_fdir % ran, 0)
		tframe_0 = np.array(tframe_0)
		tongue.append(tframe_0.reshape((tframe_0.shape[0], tframe_0.shape[1], 1)))

	train_lips = np.array(lips[:61331])		# song : train(:61331) , parole4 : train(:72611)
	train_tongue = np.array(tongue[:61331])
	test_lips = np.array(lips[61331:])
	test_tongue = np.array(tongue[61331:])

	# test_lips = np.array(lips[:14793])	# parole5 : test(:14793)
	# train_lips = np.array(lips[14793:])

	np.save(root1+"train_lips.npy", train_lips)
	np.save(root1+"train_tongue.npy", train_tongue)
	np.save(root1+"test_lips.npy", test_lips)
	np.save(root1+"test_tongue.npy", test_tongue)

	return train_lips, test_lips, train_tongue, test_tongue


def _preemphasis(y, rate=.95):
	y = np.append(y[0], y[1:]-rate*y[:-1])
	return y


def _generate_audio_Lable(dir_path=root2):
	hop=735		# hop_size
	i=2			# win_size = i * hop
	ref_db=0 	# max: 0 dB
	max_db=80 	# min: -80 dB
	magnitude = []

	_y, sr = librosa.load(dir_path, sr=44100)
	# _y = _preemphasis(_y, 0.95)
	spec=librosa.stft(_y, n_fft=hop*i, hop_length=hop, center=False)  # win_size=n_fft
	# spec=spec[:,1:-1] 	# if center = True
	spec=np.abs(spec) 		# magnitude
	ref_spec = spec.max()
	spec = librosa.amplitude_to_db(spec, ref=np.max)  
	spec = np.clip((spec - ref_db + max_db) / max_db, 1e-8, 1)		# normalization
	magnitude.extend(spec.T)

	train_label, test_label = [], []
	train_label.extend(magnitude[:61331-i+1])  # song: 61331, parole4: 76211
	test_label.extend(magnitude[61331:])
	
	# train_label.extend(magnitude[14793:]) 	# parole5
	# test_label.extend(magnitude[:14793-i+1])

	train_label = np.array(train_label)
	test_label = np.array(test_label) 

	np.save(root1+"refspec.npy", ref_spec)  # spec.max
	np.save(root1+"train_label.npy", train_label)
	np.save(root1+"test_label.npy", test_label)

	return train_label, test_label

##### mel spectrogram 1


def _generate_audio_mel_Lable(dir_path=root2):
	ref_db=0 	# max: 0 dB
	max_db=80 	# min: -80 dB
	# max_db,min_db=0,-80
	magnitude = []
	hop, i = 735, 2
	_y, sr = librosa.load(dir_path, sr=44100)
	_y = _preemphasis(_y, 0.95)

	melspec = librosa.feature.melspectrogram(_y, sr=sr, n_fft=hop*i, hop_length=hop, power=2, n_mels=128, center=False)
	# melspec=melspec[:,1:-1] 		# if center = True
	ref_melspec = melspec.max()
	powmelspec = librosa.power_to_db(melspec, ref=np.max)   					# compute dB relative to peak power
	powmelspec = np.clip((powmelspec - ref_db + max_db) / max_db, 1e-8, 1)	# normalization
	# powmelspec = np.clip((powmelspec - min_db) / (max_db - min_db), 1e-8, 1)	# normalization
	magnitude.extend(powmelspec.T)

	# # # if power=1, energie
	# melspec = librosa.feature.melspectrogram(_y,sr=sr,n_fft=hop*i, hop_length=hop, power=1, n_mels=128,center=False)
	# logmelspec = librosa.amplitude_to_db(melspec) 
	# ref_melspec = logmelspec.max() 
	# logmelspec = np.clip((logmelspec - ref_db + max_db) / max_db, 1e-8, 1)
	# magnitude.extend(logmelspec.T)
	# # #

	train_label, test_label = [], []
	train_label.extend(magnitude[:61331-i+1])
	test_label.extend(magnitude[61331:])
	train_label = np.array(train_label)
	test_label = np.array(test_label) 

	np.save(root1+"128ref.npy", ref_melspec)
	np.save(root1+"128train_label.npy", train_label)
	np.save(root1+"128test_label.npy", test_label)

	return train_label, test_label

##### mel spectrogram 2


def _get_spectrograms(y, sr=44100, n_fft=2048, win_length=1024, hop_length=512):
	linear = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
	mag = np.abs(linear)  # magnitude
	return mag


def _get_melspectrograms(y,sr,n_fft,n_mels,max_db,ref_db):
	mel0 = librosa.filters.mel(sr, n_fft, n_mels)
	mel = 20 * np.log10(np.maximum(1e-5, np.dot(mel0, y)))		# to decibel
	mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1) 	# normalization
	mel = mel.T.astype(np.float32)  							# (T, n_mels) transpose 
	return mel


def _generate_audio_mel_Lable2(dir_path=root2):
	sr=44100
	preempha=0.95
	hop_length,i=735,2
	n_fft=hop_length*i
	win_length = hop_length*i
	n_mels=128
	max_db, ref_db = 100, 0
	magnitude = []
	_y, sr = librosa.load(dir_path, sr=sr)
	_y = _preemphasis(_y, preempha)
	_mag = _get_spectrograms(_y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
	mel = _get_melspectrograms(_mag,sr=sr,n_fft=n_fft, n_mels=n_mels, max_db=max_db, ref_db=ref_db)
	magnitude.extend(mel[1:-1])

	train_label, test_label = [], []
	train_label.extend(magnitude[:61331-i+1])
	test_label.extend(magnitude[61331:])
	train_label = np.array(train_label)
	test_label = np.array(test_label)

	np.save(root1+"128train_label.npy", train_label)
	np.save(root1+"128test_label.npy", test_label)

	return train_label, test_label

##### 


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


def match_image_label3D(image_data): #3D
	image_data=np.expand_dims(image_data, axis=1)
	l=image_data.shape[0]
	image_match=[]
	for m in range(l-i+1):
		image_con=np.concatenate((image_data[m:m+i]), axis=-1) #(batch_size, 64, 64, 6)
		image_match.append(image_con)
	image_match = np.array(image_match)

	return image_match


def match_image_label2D(image_data): #2D
	l=image_data.shape[0]
	image_match=[]
	for m in range(l-i+1):
		image_con=np.concatenate((image_data[m:m+i]),axis=-1) #(batch_size, 64, 64, 6)
		image_match.append(image_con)
	image_match = np.array(image_match)

	return image_match


if __name__ == '__main__':
	train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = load_dataset()
	print(train_lips.shape)
	print(train_tongue.shape)
	print(test_lips.shape)
	print(test_tongue.shape)
	print(train_label.shape)
	print(test_label.shape)
	
	#####

	# train_lips, test_lips, train_tongue, test_tongue = lips_tongue()
	# train_label, test_label = _generate_audio_Lable(root2)
	# # train_label, test_label = _generate_audio_mel_Lable(root2)






