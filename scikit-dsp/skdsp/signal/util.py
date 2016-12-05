from skdsp.signal.discrete import _DiscreteMixin
import numpy as np
import sympy as sp

__all__ = ['is_discrete', 'is_continuous', 'is_real', 'is_complex']


def is_real_scalar(x):
    """ Checks if argument is a real valued scalar.

    Args:
        x (Python scalar, numpy scalar or sympy scalar expression):
            object to be checked.

    Returns:
        bool: True for success, False otherwise.
    """
    ok = True
    if isinstance(x, sp.Expr):
        ok = x.is_number and x.is_real
    else:
        ok = np.isscalar(x) and np.isrealobj(x)
    return ok


def is_integer_scalar(x):
    """ Checks if argument is an integer valued scalar.

    Args:
        x (Python scalar, numpy scalar or sympy scalar expression):
            object to be checked.

    Returns:
        bool: True for success, False otherwise.
    """
    ok = True
    if isinstance(x, sp.Expr):
        ok = x.is_number and x.is_integer
    else:
        ok = np.isscalar(x) and isinstance(x, np.integer)
    return ok


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
    return (s._dtype == float_)


def is_complex(s):
    return (s._dtype == complex_)
