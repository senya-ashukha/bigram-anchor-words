import numpy as np

def norn_mtx(x, y, axis):
    mtx = np.random.random((x, y))
    for row in mtx.T if axis == 'x' else mtx: 
        row /= row.sum()
    return mtx

def norm_vec(size):
    mtx = np.random.random(size)
    return mtx / mtx.sum()