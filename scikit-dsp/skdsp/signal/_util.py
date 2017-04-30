from sympy.functions.elementary.trigonometric import _pi_coeff
import numpy as np
import sympy as sp

__all__ = [s for s in dir() if not s.startswith('_')]


def _extract_omega(x):
    px = sp.arg(x)
    pc = _pi_coeff(px)
    if pc is not None:
        return sp.S.Pi*pc
    # última posibilidad para algunos caso raros
    # siempre y cuando la fase quede como (pi +) atan(algo) y
    # se haya pasado x como a*exp(sp.I*omega0)
    pisub = False
    if px.func == sp.Add:
        pisub = True
        if px.args[0].is_constant():
            pisubarg = px.args[0]
            px -= px.args[0]  # +- pi, supuestamente
    if px.func == sp.atan:
        if isinstance(x, sp.Expr):
            exponent = None
            if x.func == sp.exp:
                exponent = x.args[0]
            elif x.func == sp.Mul and x.args[1].func == sp.exp:
                exponent = x.args[1].args[0]
            if exponent is not None:
                expoverj = exponent/sp.I
                pc = _pi_coeff(expoverj)
                if pc is not None:
                    return sp.S.Pi*pc
    if pisub:
        px += pisubarg
    return px.evalf()


def _is_real_scalar(x):
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


def _is_complex_scalar(x):
    """ Checks if argument is a complex valued scalar.

    Args:
        x (Python scalar, numpy scalar or sympy scalar expression):
            object to be checked.

    Returns:
        bool: True for success, False otherwise.
    """
    ok = True
    if isinstance(x, sp.Expr):
        ok = x.is_number and sp.im(x) != 0
    else:
        ok = np.isscalar(x) and np.iscomplexobj(x)
    return ok


def _is_integer_obj(x):
    try:
        dtype = x.dtype
    except AttributeError:
        dtype = np.asarray(x).dtype
    try:
        return issubclass(dtype.type, np.integer)
    except AttributeError:
        return False


def _is_integer_scalar(x):
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
        ok = np.isscalar(x) and _is_integer_obj(x)
    return ok


def _latex_mode(s, mode):
    if mode == 'inline':
        return r'$' + s + '$'
    else:
        return s
