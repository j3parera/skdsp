from numpy import float_, complex_

from skdsp.signal.discrete import _DiscreteMixin

__all__ = ['is_discrete', 'is_continuous', 'is_real', 'is_complex']


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
