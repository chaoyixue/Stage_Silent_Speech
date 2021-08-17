import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    lips_all_chapiter = np.load("../../data_npy_one_image/lips_all_chapiters.npy")
    tongues_all_chapiter = np.load("../../data_npy_one_image/tongues_all_chapiters.npy")
    lips_chapiter7 = lips_all_chapiter[-15951:]
    tongues_chapiter7 = tongues_all_chapiter[-15951:]
    # pick fist kappa
    lips_debut = lips_chapiter7[408:438]
    tongues_debut = tongues_chapiter7[408:438]

    f = plt.figure()
    for n in range(len(lips_debut)):
        f.add_subplot(1, len(lips_debut), n+1)
        plt.imshow(lips_debut[n], 'gray')
        plt.axis('off')
        plt.xticks([])  # 不显示x轴
        plt.yticks([])
        f.add_subplot(2, len(tongues_debut), n+1)
        plt.imshow(tongues_debut[n], 'gray')
        plt.axis('off')
        plt.xticks([])  # 不显示x轴
        plt.yticks([])

    plt.show()