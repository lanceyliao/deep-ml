import numpy as np

def make_diagonal(x):
    # r = np.eye(len(x))
    # r = r * np.array(x)
    # return r
    return np.diag(x)