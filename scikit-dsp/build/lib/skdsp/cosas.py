import numpy as np


def cconv(x1, x2, N=None):
    from math import ceil
    from scipy.signal import convolve
    # linear convolution
    y = convolve(x1, x2)
    # if N not specified, return full linear convolution
    if N is None:
        return y
    x1l = x1.shape[0]
    x2l = x2.shape[0]
    S = max(x1l, x2l)
    if N < S:
        raise ValueError('N ({0}) must be at least the largest length of ' +
                         'x1 ({0}) and x2 ({1})'.format(N, x1l, x2l))
    M = y.shape[0]
    R = ceil(M/N)
    ye = np.r_[y, [0]*(R*N-M)].reshape((R, N))
    return np.sum(ye, axis=0)


def kk():
    from sympy import fourier_series
    from sympy.abc import t
    from sympy.functions.special.delta_functions import Heaviside
    s = fourier_series(Heaviside(t) - Heaviside(t-1/4), (t, 0, 1))
    iter_ = s.truncate(None)
    cosine_coeffs = []
    sine_coeffs = [0]
    for k in range(0, 4):
        term = next(iter_)
        cosine_coeffs.append(term.subs(t, 0))
        if k > 0:
            sine_coeffs.append(term.subs(t, 1/(4*k)))
