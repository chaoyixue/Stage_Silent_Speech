import numpy as np
import os
import cv2
import librosa
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
		path_to_lips = "../data/out/resize_lips"
		path_to_tongue = "../data/out/resize_tongue"
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

		np.save("../data/database/train_lips.npy", train_lips)
		np.save("../data/database/train_tongue.npy", train_tongue)
		np.save("../data/database/test_lips.npy", test_lips)
		np.save("../data/database/test_tongue.npy", test_tongue)

		train_label, test_label = _generate_audio_mel_Lable()

	return train_lips, test_lips, train_tongue, test_tongue, train_label, test_label

def _generate_audio_mel_Lable(dir_path = '../data/Songs_Audio_matched/out01234.wav'):
	step = 1
	ref_db=1e-1
    max_db=80
	magnitude = []
	_y, sr = librosa.load(dir_path, sr=44100) #(50087310,)
	melspec = librosa.feature.melspectrogram(_y,sr=sr,n_fft=2048, hop_length=735, win_length=1470, power=1.2, n_mels=64) #(64, 68147)
	logmelspec = librosa.amplitude_to_db(melspec[:,1:-1],ref=np.max) #(64, 68145)
	logmelspec = np.clip((logmelspec - ref_db + max_db) / max_db, 1e-8, 1)
	magnitude.extend(logmelspec.T)
	# print(len(magnitude)) #50087310/hop+1

	train_label, test_label = [], []
	train_label.extend(magnitude[:61330])
	test_label.extend(magnitude[61330:])

	train_label = np.array(train_label)	#(61330, 64)
	test_label = np.array(test_label) #(6815, 64)

	np.save("../data/database/train_label.npy", train_label)
	np.save("../data/database/test_label.npy", test_label)

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

# if __name__ == '__main__':
# 	train_lips, test_lips, train_tongue, test_tongue, train_label, test_label = load_dataset()
# 	print(train_lips.shape)
# 	print(train_tongue.shape)
# 	print(test_lips.shape)
# 	print(test_tongue.shape)
# 	print(train_label.shape)
# 	print(test_label.shape)

	# train_label, test_label = _generate_audio_mel_Lable('./ssi/data/Songs_Audio_matched/out01234.wav')
	# print(train_label.shape)
	# print(test_label.shape)