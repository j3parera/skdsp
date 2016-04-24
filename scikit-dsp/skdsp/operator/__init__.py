from sympy.core.numbers import Integer


__all__ = ['shift', 'flip']


def shift(s, k):
    """Returns the signal `s` shifted `k` units to the right; ie s[n-k] if
    discrete, otherwise s(t-k).

    Parameters
    ----------
    s : signal object
        The signal to be shifted.
    k : real or integer
        The shift amount. If `s` is discrete must be an integral type;
        otherwise it could be a real type

    Returns
    -------
    y : signal object
        Returns a new signal object copied from the input signal `s` but
        shifted `k` units to the right. `k` could be negative in which case
        the shift is to the left.

    Notes
    -----
    .. versionadded:: 0.0.1
    The shift is applied to any x axis variable, not necessarily time.

    Examples
    --------
    >>> s1 = sg.Delta()
    >>> s2 = sg.shift(s1, 2)
    >>> print(s2.eval(np.arange(-3, 3)))
    >>> [0., 0., 0., 0., 0., 1., 0.]
    >>> s3 = sg.shift(s1, -2)
    >>> print(s2.eval(np.arange(-3, 3)))
    >>> [0., 1., 0., 0., 0., 0., 0.]
    """
    return s.shift(k)


def flip(s):
    """Returns the signal `s` reversed in time; ie s[-n] if discrete,
    otherwise s(-t).

    Parameters
    ----------
    s : signal object
        The signal to be reversed.

    Returns
    -------
    y : signal object
        Returns a new signal object copied from the input signal `s` but
        reversed in time.

    Notes
    -----
    .. versionadded:: 0.0.1
    The reversion is applied to any x axis variable, not necessarily time.

    Examples
    --------
    >>> s1 = sg.Delta()
    >>> s2 = sg.reverse(sg.shift(s1, 2))
    >>> print(s2.eval(np.arange(-3, 3)))
    >>> [0., 1., 0., 0., 0., 0., 0.]
    """
    return s.flip()
