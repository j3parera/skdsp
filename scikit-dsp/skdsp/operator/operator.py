import sympy as sp
# from skdsp.signal._util import _is_integer_scalar


__all__ = [s for s in dir() if not s.startswith('_')]


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


class Operator(object):
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


class UnaryOperatorMixin(object):
    """ Mixin para indicar que el operador es unitario; es decir,que opera
    sobre una única señal o función.
    """
    pass


class BinaryOperatorMixin(object):
    """ Mixin para indicar que el operador es binario; es decir,que opera
    sobre dos señales o funciones.
    """
    pass

# ==============================================================================
#    Operadores unitarios que cambian la variable independiente
# ==============================================================================


class FlipOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
        """ Invierte la variable independiente:
        expr(var) -> expr(-var)
        """
        return expr.xreplace({var: -var})


class ShiftOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
        """ Retrasa la variable independiente args[0] unidades:
        expr(var) -> expr(var - args[0])
        """
        s = args[0]
#         if _is_integer_scalar(s):
#             s = sp.Integer(args[0])
        return expr.xreplace({var: (var - s)})


class ScaleOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
        """ Escala la variable independiente args[0] unidades:
        expr(var) -> expr(args[0]*var)
        """
        return expr.xreplace({var: args[0]*var})

# class ExpandOperator(Operator, UnitaryOperator):
#
#     def __init__(self, beta):
#         super().__init__(beta)
#         self._beta = beta
#
#     def compose(self, op, var):
#         # TODO: cuidadito con señales discretas
#         return op.xreplace({var: var / self._beta})
#
#
# class CompressOperator(Operator, UnitaryOperator):
#
#     def __init__(self, alpha):
#         super().__init__(alpha)
#         self.alpha = alpha
#
#     def compose(self, op, var):
#         return op.xreplace({var: var * self._alpha})
#
#
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


class GainOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
        """ Amplifica la expresión por args[0]:
        expr(var) -> args[0]*expr(var)
        """
        return args[0]*expr


class AbsOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
        """ Toma valor absoluto de la expresión:
        expr(var) -> abs(expr(var))
        """
        return sp.Abs(expr)


class ConjugateOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
        """ Devuelve el conjugado de la expresión:
        expr(var) -> conj(expr(var))
        """
        if expr.is_real:
            return expr
        return sp.conjugate(expr)


class RealPartOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
        """ Devuelve la parte real de la expresión:
        expr(var) -> Re(expr(var))
        """
        return sp.re(expr)


class ImaginaryPartOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
        """ Devuelve la parte imaginaria de la expresión:
        expr(var) -> Im(expr(var))
        """
        return sp.im(expr)


class HermitianOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def apply(var, expr, *args):
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
