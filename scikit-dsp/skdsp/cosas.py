import numpy as np
import sympy as sp


class _DiscreteDelta(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg.is_nonzero:
                return sp.S.Zero
            else:
                return sp.S.One

    @staticmethod
    def _imp_(n):
        return np.equal(n, 0).astype(np.float_)


class _DiscreteStep(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if arg is sp.S.NaN:
            return sp.S.NaN
        elif arg.is_negative:
            return sp.S.Zero
        elif arg.is_zero or arg.is_positive:
            return sp.S.One

    @staticmethod
    def _imp_(n):
        return np.greater_equal(n, 0).astype(np.float_)


class _DiscreteRamp(sp.Function):

    nargs = 1
    is_finite = False
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if arg is sp.S.NaN:
            return sp.S.NaN
        elif arg.is_negative:
            return sp.S.Zero
        elif arg.is_zero or arg.is_positive:
            return arg

    @staticmethod
    def _imp_(n):
        return n*np.greater_equal(n, 0).astype(np.float_)


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
