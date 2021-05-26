from scipy.io import loadmat

if __name__ == "__main__":
    dict_mat = loadmat("lsp_all_chapiter.mat")
    print(type(dict_mat))
    npy_lsp = dict_mat["lsp_all_chapiter"]
    print(npy_lsp.shape)