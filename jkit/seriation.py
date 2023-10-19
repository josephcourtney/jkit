import numpy as np

from .tsp import tsp


def seriation(data, d_func, symmetric = False):
    dmat = np.zeros((data.shape[0]+1, data.shape[0]+1))
    dmat[:-1,:-1] = d_func(data)
    path, path_length = tsp(dmat + 1e-12, initialization = 'greedy')
    idx = path.index(data.shape[0])
    path = path[idx+1:] + path[:idx]

    if symmetric:
        data = data[path,:][:,path]
        return data, path, path_length
    else:
        data = np.swapaxes(data[path,:], 1, 0)

        dmat = np.zeros((data.shape[1]+1, data.shape[1]+1))
        dmat[:-1,:-1] = d_func(data)
        path, path_length = tsp(dmat + 1e-12, initialization = 'greedy')
        idx = path.index(data.shape[1])
        path = path[idx+1:] + path[:idx]

        data = np.swapaxes(data[path,:], 1, 0)
        return data, path, path_length
