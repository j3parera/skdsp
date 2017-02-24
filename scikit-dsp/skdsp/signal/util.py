from skdsp.signal.discrete import _DiscreteMixin
import numpy as np
import sympy as sp

__all__ = ['is_discrete',
           'is_continuous',
           'is_real',
           'is_complex',
           'xvar']


def discrete_xvar():
    '''
    Return the deafult xvar for discrete signals
    '''
    return _DiscreteMixin._default_xvar()


def is_discrete(s):
    '''
    Checks if `s` is a discrete signal
    :param s: signal to check
    :return: True if `s` is a discrete signal
    '''
    return isinstance(s, _DiscreteMixin)


def is_continuous(s):
    return not isinstance(s, _DiscreteMixin)


def is_real(s):
    return (s._dtype == np.float_)


def is_complex(s):
    return (s._dtype == np.complex_)
