from math import ceil
from numbers import Integral, Number
import numpy as np
import matplotlib as plt


__all__ = ['buffer', 'stem_init', 'stem_update']


def stem_init(x, y, iax=None, color='navy', yaxis=False, grid=True, xlim=True,
              ylim=True, xlabel=r'$n$', ylabel=None, title=None):
    """
    Representa los pares (x, y) con stem
    """
    if iax is None:
        ax = plt.pyplot.gca()
    else:
        ax = iax
    lines = ax.stem(x, y, markerfmt='o', linefmt='-', basefmt='-')
    lines[0].set_color(color)
    lines[2].set_color(color)
    for l in lines[1]:
        l.set_color(color)
    ax.axhline(0, color=color, lw=1.5)
    if yaxis:
        ax.axvline(0, color=color, lw=1.5)
    if xlim:
        ax.set_xlim(x[0]-0.5, x[-1]+0.5)
    if ylim:
        max_ = np.max(y)
        min_ = np.min(y)
        d = (max_ - min_)*0.1
        ax.set_ylim(min_-d, max_+d)
    ax.grid(grid)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14, position=(1, 1), labelpad=-10,
                      horizontalalignment='right')
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)
    if title is not None:
        txt = ax.set_title(title, fontsize=14, loc='left')
    return lines, ax, txt


def stem_update(st, y, title=None):
    """
    Actualiza los datos de una representación stem
    """
    st[0][0].set_ydata(y)
    for k, line in enumerate(st[0][1]):
        line.set_ydata([0, y[k]])
    if title is not None:
        st[2].set_text(title)


def buffer(x, N, P=0, opt=None, order='C', *args):
    """
    Pretende funcionar como buffer de matlab pero por filas en lugar de
    por columnas, salvo que order sea 'F'

    Args:
    x (array_like): Input data, in any form that can be converted to an array.
    This includes lists, lists of tuples, tuples, tuples of tuples, tuples of
    of lists and ndarrays.
    order (str): {'C', 'F'}, optional. Whether to use row-major (C-style) or
    column-major (Fortran-style) memory representation. Defaults to 'C'.

    Returns:
    str: bla, bla

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
