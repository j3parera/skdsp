from numbers import Number
from skdsp.operator.operator import ShiftOperator
from skdsp.signal.signal import FunctionSignal, ConstantSignal

import numpy as np
import sympy as sp

__all__ = ['ContinuousFunctionSignal']


class ContinuousMixin(object):

    @staticmethod
    def _default_xvar():
        return sp.symbols('t', real=True)

    def __init__(self):
        pass

    def _copy_to(self, other):
        pass

    def shift(self, tau):
        if not self._check_is_real(tau):
            raise TypeError('delay/advance must be real')
        return FunctionSignal.shift(self, tau)

    def delay(self, tau):
        return self.shift(tau)

    def __rshift__(self, tau):
        return ContinuousMixin.shift(self, tau)

    __irshift__ = __rshift__

    def __lshift__(self, tau):
        return ContinuousMixin.shift(self, -tau)

    __ilshift__ = __lshift__

    def scale(self, alpha):
        # Scale permite cualquier valor de alpha, no necesariamente entero
        if not self._check_is_real(alpha):
            raise TypeError('scale must be real')
        return FunctionSignal.scale(self, alpha)

    def __add__(self, other):
        if not isinstance(other, (ContinuousMixin, Number)):
            raise TypeError("can't add {0}".format(str(other)))
        if other == 0 or other == Constant(0):
            return self
        if self == 0 or self == Constant(0):
            if isinstance(other, Number):
                return Constant(other)
            else:
                return other
        s = ContinuousFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__add__(s, o)

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, (ContinuousMixin, Number)):
            raise TypeError("can't sub {0}".format(str(other)))
        if other == 0 or other == Constant(0):
            return self
        if self == 0 or self == Constant(0):
            if isinstance(other, Number):
                return Constant(-other)
            else:
                return -other
        s = ContinuousFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__sub__(s, o)

    def __rsub__(self, other):
        if not isinstance(other, (ContinuousMixin, Number)):
            raise TypeError("can't sub {0}".format(str(other)))
        if other == 0 or other == Constant(0):
            return -self
        if self == 0 or self == Constant(0):
            if isinstance(other, Number):
                return Constant(other)
            else:
                return other
        s = ContinuousFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__rsub__(o, s)

    __isub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, sp.Expr):
            if other.is_number:
                other = Constant(other)
        if not isinstance(other, (ContinuousMixin, Number)):
            raise TypeError("can't multiply {0}".format(str(other)))
        if other == 0 or other == Constant(0):
            return Constant(0)
        if self == 0 or self == Constant(0):
            return Constant(0)
        if other == 1 or other == Constant(1):
            return self
        if self == 1 or self == Constant(1):
            if isinstance(other, Number):
                return Constant(other)
            else:
                return other
        if isinstance(other, (Constant, Number)):
            s = self.__class__._factory(self)
        else:
            s = ContinuousFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__mul__(s, o)

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, (ContinuousMixin, Number)):
            raise TypeError("can't divide {0}".format(str(other)))
        if other == 1 or other == Constant(1):
            return self
        if isinstance(other, (Constant, Number)):
            s = self.__class__._factory(self)
        else:
            s = ContinuousFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__truediv__(s, o)

    __itruediv__ = __truediv__

    def __rtruediv__(self, other):
        if not isinstance(other, (ContinuousMixin, Number)):
            raise TypeError("can't divide {0}".format(str(other)))
        if other == 1 or other == Constant(1):
            return Constant(1)/self
        s = ContinuousFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__rtruediv__(o, s)


class ContinuousFunctionSignal(ContinuousMixin, FunctionSignal):
    # Las especializaciones discretas deben ir antes que las funcionales
    @staticmethod
    def _factory(other):
        s = ContinuousFunctionSignal(other._yexpr)
        other._copy_to(s)
        return s

    def __init__(self, expr):
        FunctionSignal.__init__(self, expr)
        ContinuousMixin.__init__(self)
        self._laplace_transform = None
        self._fourier_transform = None

    def _copy_to(self, other):
        other._laplace_transform = self._laplace_transform
        other._fourier_transform = self._fourier_transform
        ContinuousMixin._copy_to(self, other)
        FunctionSignal._copy_to(self, other)

    @property
    def laplace(self):
        return self._laplace_transform

    @property
    def fourier(self):
        return self._fourier_transform

#     @property
#     def period(self):
#         # Si la señal es suma y cada una es periódica, el periodo es el
#         # mínimo común múltiplo
#         # Algo así, pero bien
#         # if isinstance(self._yexpr, sp.Add):
#         #    Ns = []
#         #    for s0 in self._yexpr.args:
#         #        Ns.append(DiscreteFunctionSignal(s0).period)
#         #    N = mcm(Ns)

    def eval(self, x):
        is_scalar = False
        if not isinstance(x, np.ndarray):
            is_scalar = True
            x = np.array([x])
        y = FunctionSignal.eval(self, x)
        if is_scalar:
            y = np.asscalar(y)
        return y

    def __abs__(self):
        s = ContinuousFunctionSignal._factory(self)
        return FunctionSignal.__abs__(s)

    def magnitude(self, dB=False):
        m = abs(self)
        if dB:
            mdb = ContinuousFunctionSignal._factory(self)
            mdb._yexpr = 20*sp.log(m._yexpr, 10)
            return mdb
        return m


class Constant(ContinuousFunctionSignal, ConstantSignal):

    @staticmethod
    def _factory(other, cte):
        s = Constant(cte)
        if other:
            other._copy_to(s)
        return s

    def __init__(self, c):
        ContinuousFunctionSignal.__init__(self, sp.sympify(c))
        ConstantSignal.__init__(self, c, self._xvar)

    def max(self):
        return self._yexpr

    def min(self):
        return self._yexpr


class Delta(ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Delta()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, delay=0):
        expr = sp.DiracDelta(self._default_xvar())
        ContinuousFunctionSignal.__init__(self, expr)
        # delay
        if delay != 0:
            self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
            self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    def max(self):
        return sp.oo

    def min(self):
        return 0

    def _print(self):
        return 'd({0})'.format(str(self._xexpr))


class Step(ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Step()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, delay=0):
        expr = sp.Heaviside(self._default_xvar())
        ContinuousFunctionSignal.__init__(self, expr)
        if delay != 0:
            self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
            self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    def max(self):
        return 1

    def min(self):
        return 0

    def _print(self):
        return 'u({0})'.format(str(self._xexpr))


class Ramp(ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Ramp()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, delay=0):
        expr = self._default_xvar()
        ContinuousFunctionSignal.__init__(self, expr)
        # delay
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    def _print(self):
        return 'r({0})'.format(str(self._xexpr))

    def max(self):
        return np.inf

    def min(self):
        return -np.inf
