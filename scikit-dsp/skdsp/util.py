from math import ceil
from numbers import Integral, Number
import numpy as np


def buffer(x, N, P=0, opt=None, order='C', *args):
    """
    Pretende funcionar como buffer de matlab pero por filas en lugar de
    por columnas, salvo que order sea 'F'

    Parameters
    ----------
    x : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or
        column-major (Fortran-style) memory representation.
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray.  If `a` is a subclass of ndarray, a base
        class ndarray is returned.

    Examples
    --------
    Convert a list into an array:
    >>> a = [1, 2]
    >>> np.asarray(a)
    array([1, 2])

    """
    if not isinstance(N, Integral):
        raise ValueError('N must be an integer')
    if not isinstance(P, Integral) or P >= N:
        raise ValueError('P must be an integer less than N')
    if order not in ('C', 'F'):
        raise ValueError('allowed values for order are "C" or "F"')
    L = len(x)
    if P > 0:
        delay = np.array([])
        if opt is not None:
            if isinstance(opt, str):
                if opt == 'nodelay':
                    nf = 1 + ceil((L - N) / (N - P))
                else:
                    raise ValueError('unexpected opt string')
            else:
                if hasattr(opt, '__len__') and len(opt) == P:
                    delay = np.asarray(opt)
                    nf = ceil(L / (N - P))
                elif isinstance(opt, Number) and P == 1:
                    delay = np.array([opt])
                    nf = ceil(L / (N - P))
                else:
                    raise ValueError('opt must be an array like of length P')
        else:
            delay = np.r_[[0.0]*P]
            nf = ceil(L / (N - P))
        xf = np.r_[delay, x.copy().flatten(),
                   [0.0]*((nf * N) - L - len(delay))]
    else:
        skip = 0
        if opt is not None:
            if not isinstance(opt, Integral) or opt < 0 or opt > -P:
                raise ValueError('opt must be an integer between 0 and -P')
            skip = opt
        nf = ceil(L / (N - P))
        xf = np.r_[x.copy().flatten(), [0.0]*((nf * (N - P)) - L)][skip:]
    y = np.zeros((nf, N))
    for f in range(nf):
        s = f * (N - P)
        y[f, :] = xf[s:s+N]
    if order == 'C':
        return y
    else:
        return y.T
