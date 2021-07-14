import numpy as np
from matplotlib import pyplot as plt


def plot_lsf_values():
    test_lsf = np.load("../../LSF_data/lsp_all_chapiter.npy")
    print(test_lsf.shape)
    plt.figure()
    plt.plot()


if __name__ == "__main__":
    plot_lsf_values()