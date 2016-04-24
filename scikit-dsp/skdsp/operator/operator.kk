from numbers import Real, Number, Integral
import sympy as sp
from sympy.physics.quantum.operator import UnitaryOperator
from skdsp.transform.discrete import ZTransform, FourierTransform


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
    def apply(xvar, xexpr, yexpr, *args):
        """ Aplica el operador

        Parámetros
        ----------
        xvar: variable independiente
        xexpr: expresión de la variable independiente xexpr(xvar)
        yexpr: expresión de la variable dependiente yexpr(xexpr(xvar))
        args: parámetro(s) adicional(es) definido(s) por cada operador

        Devuelve
        ----------
        Tupla con (expresión modificada de la variable independiente,
        expresión modificada de la variable dependiente)

        Ejemplo
        -------
        Operador retardo de 2 unidades aplicado con
        xvar = n, xexpr = n-1, yexpr = exp(-0.5*(n-1))
        --> (xexpr = n-3, yexpr = exp(-0.5*(n-3)))
        """
        pass

    @staticmethod
    def _is_real(value):
        if not isinstance(value, Real):
            raise TypeError("parameter '{0}' must be real".format(value))

    @staticmethod
    def _is_integer(value):
        if not isinstance(value, Integral):
            raise TypeError("parameter '{0}' must be integer".format(value))


class UnaryOperatorMixin(object):
    """ Mixin para indicar que el operador es unitario; es decir,que opera
    sobre una única señal o función.
    """
    pass


class BinaryOperatorMixin(object):
    """ Mixin para indicar que el operador es binario; es decir,que opera
    sobre dos señales o funciones.
    """

# ==============================================================================
#    Operadores unitarios
# ==============================================================================


class FlipOperator(Operator, UnitaryOperator):

    @staticmethod
    def time_apply(var, expr):
        """ Aplica el operador en el dominio del tiempo a la expresión `expr`,
        con variable temporal `var`.
        Parámetros
        ----------
        var : variable temporal
        expr : expresión en el dominio temporal
        Ambos parámetros componen expr(var)
        Devuelve
        -------
        expr : expr(-var)
        """
        return expr.xreplace({var: -var})

    @staticmethod
    def z_apply(zt):
        """ Aplica el operador en el dominio transformado z a la transformada
        z `zt`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        roc : región de convergencia
        Devuelve
        -------
        ZTransform(zt.var, zt.expr(1/zt.var), 1/zt.roc)
        """
        newexpr = zt._expr.xreplace({zt._var: zt._var**(-1)})
        right = abs(1/zt._roc.left)
        left = abs(1/zt._roc.right)
        newroc = sp.Interval(left, right, zt._roc.left_open,
                             zt._roc.right_open)
        return ZTransform(zt._var, newexpr, newroc)

    @staticmethod
    def dtf_apply(tf):
        """ Aplica el operador en el dominio de la frecuencia a la transformada
        de Fourier `tf`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        Devuelve
        -------
        FourierTransform(zt.var, zt.expr(-var))
        """
        return FourierTransform(tf._expr.xreplace({tf.var: -tf.var}))


class ShiftOperator(Operator, UnitaryOperator):

    @staticmethod
    def time_apply(var, expr, *args):
        """ Aplica el operador en el dominio del tiempo a la expresión `expr`,
        con variable temporal `var`.
        Parámetros
        ----------
        var : variable temporal
        expr : expresión en el dominio temporal
        Ambos parámetros componen expr(var)
        args[0] : retardo
        Devuelve
        -------
        expr : expr(var - args[0])
        """
        Operator.check_is_real(args[0])
        return expr.xreplace({var: (var - args[0])})

    @staticmethod
    def z_apply(zt, *args):
        """ Aplica el operador en el dominio transformado z a la transformada
        z `zt`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        args[0] : retardo
        roc : región de convergencia
        Devuelve
        -------
        Ztransform(zt.var, zt.var**(-args[0])*zt.expr(zt.var), zt.roc +- {0})
        """
        Operator.check_is_real(args[0])
        expr = (zt._var**(-args[0]))*zt._expr
        zero = True if args[0] >= 0 else False
        inf = True if args[0] <= 0 else False
        roc = zt.roc.intersect(sp.Interval(0, sp.oo, zero, inf))
        return ZTransform(zt._var, expr, roc)

    @staticmethod
    def dtf_apply(ft, *args):
        """ Aplica el operador en el dominio de la frecuencia a la transformada
        de Fourier `tf`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        args[0] : retardo
        Devuelve
        -------
        FourierTransform(ft.var, exp(-j*var*args[0])*ft.expr(ft.var))
        """
        return FourierTransform(sp.exp(-sp.numbers.I*ft._var*args[0])*ft._expr)


# class TimeScaleOperator(Operator, UnitaryOperator):
#
#     def __init__(self, alpha):
#         super().__init__(alpha)
#         self._alpha = alpha
#
#     def compose(self, op, var):
#         return op.xreplace({var: var * self._alpha})
#
#
# class ExpandOperator(Operator, UnitaryOperator):
#
#     def __init__(self, beta):
#         super().__init__(beta)
#         self._beta = beta
#
#     def compose(self, op, var):
#         # TODO cuidadito con señales discretas
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


class GainOperator(Operator, UnaryOperatorMixin):

    @staticmethod
    def time_apply(var, expr, *args):
        """ Aplica el operador en el dominio del tiempo a la expresión `expr`,
        con variable temporal `var`.
        Parámetros
        ----------
        var : variable temporal
        expr : expresión en el dominio temporal
        Ambos parámetros componen expr(var)
        args[0] : ganancia
        Devuelve
        -------
        expr : args[0]*expr(var)
        """
        return args[0]*expr

    @staticmethod
    def z_apply(zt, *args):
        """ Aplica el operador en el dominio transformado z a la transformada
        z `zt`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        roc : región de convergencia
        args[0] : ganancia
        Devuelve
        -------
        Ztransform(zt.var, args[0]*zt.expr(zt.var), zt.roc)
        """
        return ZTransform(zt._var, args[0]*zt._expr, zt._roc)

    @staticmethod
    def dtf_apply(zt, *args):
        """ Aplica el operador en el dominio de la frecuencia a la transformada
        de Fourier `tf`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        args[0] : ganancia
        Devuelve
        -------
        FourierTransform(zt.var, args[0]*zt.expr(zt.var))
        """
        return FourierTransform(zt._var, args[0]*zt._expr)

# ==============================================================================
#    Operadores binarios
# ==============================================================================

# TODO KK de la VACA. Sólo es válido el operador definido en el dominio del tiempo

class AddOperator(Operator, BinaryOperatorMixin):

    @staticmethod
    def time_apply(var, expr, *args):
        """ Aplica el operador en el dominio del tiempo a la expresión `expr`,
        con variable temporal `var`.
        Parámetros
        ----------
        var : variable temporal
        expr : expresión en el dominio temporal
        Ambos parámetros componen expr(var)
        args[0] : adendo
        Devuelve
        -------
        expr : args[0] + expr(var)
        """
        return args[0] + expr

    @staticmethod
    def z_apply(zt, *args):
        """ Aplica el operador en el dominio transformado z a la transformada
        z `zt`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        roc : región de convergencia
        args[0] : adendo
        Devuelve
        -------
        si adendo es escalar:
            Ztransform(zt.var, addendo + zt.expr(zt.var), zt.roc)
        si adendo es z-transform
            Ztransform(zt.var, addendo(zt.var) + zt.expr(zt.var),
            zt.roc intersect addendo.roc)
        """
        if not isinstance(args[0], (Number, ZTransform)):
            raise TypeError('addendum must be number or z transform')
        if isinstance(args[0], Number):
            return ZTransform(zt.var, args[0] + zt.expr, zt.roc)
        else:
            # TODO verificar si la ROC crece
            return ZTransform(zt.var, args[0].var(zt.var) + zt.expr,
                              zt.roc.intersect(args[0].roc))

    @staticmethod
    def dtf_apply(ft, *args):
        """ Aplica el operador en el dominio de la frecuencia a la transformada
        de Fourier `ft`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        args[0] : adendo
        Devuelve
        -------
        si adendo es escalar:
            FourierTransform(ft.var, addendo + ft.expr(ft.var))
        si adendo es fourier-transform
            Ztransform(ft.var, addendo(ft.var) + ft.expr(ft.var))
        """
        if not isinstance(args[0], (Number, FourierTransform)):
            raise TypeError('addendum must be number or fourier transform')
        if isinstance(args[0], Number):
            return FourierTransform(ft.var, args[0] + ft.expr)
        else:
            return FourierTransform(ft.var, args[0].var(ft.var) + ft.expr)


class MultiplyOperator(Operator, BinaryOperatorMixin):

    @staticmethod
    def time_apply(var, expr, *args):
        """ Aplica el operador en el dominio del tiempo a la expresión `expr`,
        con variable temporal `var`.
        Parámetros
        ----------
        var : variable temporal
        expr : expresión en el dominio temporal
        Ambos parámetros componen expr(var)
        args[0] : multiplicando
        Devuelve
        -------
        expr : args[0]*expr(var)
        """
        return args[0]*expr

    @staticmethod
    def z_apply(zt, *args):
        """ Aplica el operador en el dominio transformado z a la transformada
        z `zt`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        roc : región de convergencia
        args[0] : multiplicando
        Devuelve
        -------
        si adendo es escalar:
            Ztransform(zt.var, multiplicando*zt.expr(zt.var), zt.roc)
        si adendo es z-transform
            Ztransform(zt.var, multiplicando(zt.var)*zt.expr(zt.var),
            zt.roc algo addendo.roc)
        """
        if not isinstance(args[0], (Number, ZTransform)):
            raise TypeError('addendum must be number or z transform')
        if isinstance(args[0], Number):
            return ZTransform(zt.var, args[0]*zt.expr, zt.roc)
        else:
            # TODO algo con la ROC
            return ZTransform(zt.var, args[0].var(zt.var) + zt.expr,
                              zt._roc.intersect(args[0].roc))

    @staticmethod
    def dtf_apply(ft, *args):
        """ Aplica el operador en el dominio de la frecuencia a la transformada
        de Fourier `ft`.
        Parámetros
        ----------
        var : variable en el dominio transformado
        expr : expresión en el dominio transformado
        Ambos parámetros componen expr(var)
        args[0] : multiplicando
        Devuelve
        -------
        si adendo es escalar:
            FourierTransform(ft.var, multiplicando*ft.expr(ft.var))
        si adendo es fourier-transform
            Ztransform(ft.var, multiplicando(ft.var)*ft.expr(ft.var))
        """
        if not isinstance(args[0], (Number, FourierTransform)):
            raise TypeError('addendum must be number or fourier transform')
        if isinstance(args[0], Number):
            return FourierTransform(ft.var, args[0]*ft.expr)
        else:
            return FourierTransform(ft.var, args[0].var(ft.var)*ft.expr)
