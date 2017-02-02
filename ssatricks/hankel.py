import numpy as np
import scipy as sc

from numpy.fft import rfft, irfft


def trajectory_matrix(x, L=None):
    N = x.shape[0]
    if L is None:
        L = N // 2
    K = N - L + 1
    hh = np.zeros((L, K))
    for j in range(K):
        hh[:, j] = x[j:L + j]
    return hh


def hankelization_r1(u, v, s):
    L = u.shape[0]
    K = v.shape[0]
    N = K + L - 1
    u = np.pad(u, (0, K - 1), mode='constant', constant_values=0)
    v = np.pad(v, (0, L - 1), mode='constant', constant_values=0)

    g = irfft(rfft(u) * rfft(v))
    a1 = np.ones(N)
    a1[:L] = np.arange(1, L + 1)
    a1[L:K - 1] = np.ones(K - L - 1) * L
    a1[K - 1:] = np.arange(L, 0, -1)
    return (s / a1) * g
