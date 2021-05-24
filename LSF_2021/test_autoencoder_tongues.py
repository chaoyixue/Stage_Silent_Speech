import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, concatenate, Flatten, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model


def lips_AE_disp(img_in, img_out, img_idx):

    num_img = len(img_idx)
    plt.figure(figsize=(18, 4))

    for i, image_idx in enumerate(img_idx):
        # 显示输入图像
        ax = plt.subplot(2, num_img, i + 1)
        plt.imshow(img_in[image_idx].reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 显示输出图像
        ax = plt.subplot(2, num_img, num_img + i + 1)
        plt.imshow(img_out[image_idx].reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    # input images
    X = np.load("C:/Users/chaoy/Desktop/StageSilentSpeech/data_npy_one_image/tongues_all_chapiters.npy")
    nb_images_chapiter7 = 15951

    # normalisation
    X = X / 255.0

    # ch1-ch6
    X_train = X[:-15951, :]
    X_test = X[-15951:, :]

    autoencoder = keras.models.load_model(
        "C:/Users/chaoy/Desktop/StageSilentSpeech/results/week_0517/autoencoder_tongues-73-0.06114350.h5")

    # 挑选十个随机的图片

    num_images = 5
    np.random.seed(42)
    random_test_images = np.random.randint(X_test.shape[0], size=num_images)
    # 预测输出图片
    decoded_img = autoencoder.predict(X_test)
    # 显示并对比输入与输出图片
    lips_AE_disp(X_test, decoded_img, random_test_images)
