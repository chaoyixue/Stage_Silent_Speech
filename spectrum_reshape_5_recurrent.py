import numpy as np
import librosa


if __name__ == "__main__":
    test = np.load("../data_npy_one_image/spectrogrammes_all_chapitre_corresponding.npy")
    print(test.shape)
    result = np.zeros((736, 84651))
    result[:, :10050] = test[:, 2:10052]
    result[:, 10050:24487] = test[:, 10056:24493]
    result[:, 24487:33368] = test[:, 24497:33378]
    result[:, 33368:48985] = test[:, 33382:48999]
    result[:, 48985:63534] = test[:, 49003:63552]
    result[:, 63534:68704] = test[:, 63556:68726]
    result[:, 68704:84651] = test[:, 68730:84677]
    print(result.shape)
    np.save("../five_recurrent_image_npy/spectrum_recurrent_all.npy", result)