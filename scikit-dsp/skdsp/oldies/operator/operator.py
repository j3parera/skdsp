from abc import abstractmethod

import sympy as sp
from sympy.core.cache import cacheit

__all__ = [s for s in dir() if not s.startswith("_")]


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


class _Operator(sp.Basic):
    """ Clase base para todos los operadores.
    * Un operador está definido en un dominio entero (señales discretas,DFTs),
    real (señales continuas, TF) o complejo (Laplace o Z).
    * También hay operadores genéricos como la suma, el producto o el producto
    por un escalar, que hacen la misma transformación en cualquier dominio.
    * Los operadores pueden ser unitarios, si operan sobre una señal (función),
    o binarios, si operan con dos.
    * Un operador transforma una señal o función en otra señal o función; es,
    por tanto un mapeo funcional (MAP) o
    * Un operador puede hacer una operación de reducción generando un
    escalar a partir de una señal (p.e. media) o dos (producto escalar)
    (REDUCE).
    """

    is_Unary = None
    is_Map = None
    __instance = None

    @staticmethod
    def apply(var, expr, *args):
        """ Aplica el operador a la expresión expr(var)

        Parámetros
        ----------
        var: variable independiente
        expr: expresión con la variable independiente expr(var)
        args: parámetro(s) adicional(es) definido(s) por cada operador

        Devuelve
        ----------
        expresión modificada

        Ejemplo
        -------
        Operador retardo de 2 unidades aplicado con
        (var = n, expr = n-1) -> xexpr = n-3
        (var = n, expr = exp(-0.5*(n-1))) -> expr = exp(-0.5*(n-3)))
        """
        pass

    def is_unary(self):
        return self.is_Unary

    def is_binary(self):
        return not self.is_Unary

    def is_map(self):
        return self.is_Map

    def is_reduce(self):
        return not self.is_Map


# ==============================================================================
#    Operadores unitarios que cambian la variable independiente
# ==============================================================================
@cacheit
class FlipOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(var, expr, *_args):
        """ Invierte la variable independiente:
        expr(var) -> expr(-var)
        """
        return expr.xreplace({var: -var})


@cacheit
class ShiftOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(var, expr, *args):
        """ Retrasa la variable independiente args[0] unidades:
        expr(var) -> expr(var - args[0])
        """
        s = args[0]
        return expr.xreplace({var: (var - s)})


@cacheit
class ScaleOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(var, expr, *args):
        """ Escala la variable independiente args[0] unidades:
        expr(var) -> expr(args[0]*var)
        """
        if args[1] is True:
            return expr.xreplace({var: var * args[0]})
        else:
            return expr.xreplace({var: var / args[0]})


# class CircularShiftOperator(Operator, UnitaryOperator):
#
#     def __init__(self, k, N):
#         super().__init__(k)
#         self._k = k
#         if not isinstance(N, Real):
#             raise TypeError('modulo length must be real')
#         self._N = N
#
#     def apply(self, x):
#         x0 = np.roll(np.intersect1d(np.arange(0, self._N), x, True), self._k)
#         i0 = np.where(x == 0)[0][0]
#         iN = i0 + len(x0)
#         x[i0:iN] = x0
#         return x

# ==============================================================================
#    Operadores unitarios que NO cambian la variable independiente
# ==============================================================================


@cacheit
class GainOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(_var, expr, *args):
        """ Amplifica la expresión por args[0]:
        expr(var) -> args[0]*expr(var)
        """
        return args[0] * expr


@cacheit
class AbsOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(_var, expr, *_args):
        """ Toma valor absoluto de la expresión:
        expr(var) -> abs(expr(var))
        """
        return sp.Abs(expr)


@cacheit
class ConjugateOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(_var, expr, *_args):
        """ Devuelve el conjugado de la expresión:
        expr(var) -> conj(expr(var))
        """
        if expr.is_real:
            return expr
        return sp.conjugate(expr)


@cacheit
class RealPartOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(_var, expr, *_args):
        """ Devuelve la parte real de la expresión:
        expr(var) -> Re(expr(var))
        """
        (real, _) = expr.as_real_imag(evaluate=True)
        return real


@cacheit
class ImaginaryPartOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(_var, expr, *_args):
        """ Devuelve la parte imaginaria de la expresión:
        expr(var) -> Im(expr(var))
        """
        (_, imag) = expr.as_real_imag(evaluate=True)
        return imag


@cacheit
class HermitianOperator(_Operator):

    is_Unary = True
    is_Map = True

    @staticmethod
    def apply(var, expr, *_args):
        """ Devuelve el conjugado de la expresión invertida:
        expr(var) -> conj(expr(-var))
        """
        if expr.is_real:
            s = expr
        else:
            s = sp.conjugate(expr)
        return s.xreplace({var: -var})


# ==============================================================================
#    Operadores binarios que NO cambian la variable independiente
# ==============================================================================
