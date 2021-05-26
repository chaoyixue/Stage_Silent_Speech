from scipy.io import loadmat
import numpy as np

if __name__ == "__main__":
    dict_mat = loadmat("lsp_all_chapiter.mat")
    print(type(dict_mat))
    npy_lsp = dict_mat["lsp_all_chapiter"]
    npy_lsp = np.matrix.transpose(npy_lsp)
    print(npy_lsp.shape)
    np.save("lsp_all_chapiter.npy", npy_lsp)
