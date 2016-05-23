from numbers import Number
from skdsp.operator.operator import ShiftOperator, ScaleOperator
from skdsp.signal.signal import FunctionSignal, ConstantSignal, Signal

import numpy as np
import sympy as sp
from sympy.core.evaluate import evaluate


__all__ = ['ContinuousFunctionSignal']


class _ContinuousMixin(object):

    @staticmethod
    def _default_xvar():
        return sp.symbols('t', real=True)

    def __init__(self):
        pass

    def _copy_to(self, other):
        pass

#     def _has_ramp(self):
#         # CUIDADITO, si esto da más problemas quizás sea mejor quitarlo
#         # aunque las rampas no se usan mucho que digamos
#         dohas = False
#         if isinstance(self, (Ramp, Ramp._ContinuousRamp)):
#             dohas = True
#         else:
#             for arg in sp.preorder_traversal(self._yexpr):
#                 if isinstance(arg, Ramp._ContinuousRamp):
#                     dohas = True
#                     break
#         return dohas

    def flip(self):
        s = FunctionSignal.flip(self)
        return s

    __reversed__ = flip

    def shift(self, tau):
        if not self._check_is_real(tau):
            raise TypeError('delay/advance must be real')
        # esto evita que r(t-k) se convierta en t-k (sin la r)
        # doeval = not self._has_ramp()
        # with evaluate(doeval):
        with evaluate(True):
            s = FunctionSignal.shift(self, tau)
        return s

    def delay(self, tau):
        return self.shift(tau)

    def __rshift__(self, tau):
        return _ContinuousMixin.shift(self, tau)

    __irshift__ = __rshift__

    def __lshift__(self, tau):
        return _ContinuousMixin.shift(self, -tau)

    __ilshift__ = __lshift__

    def scale(self, alpha):
        # Scale permite cualquier valor de alpha, no necesariamente entero
        if not self._check_is_real(alpha):
            raise TypeError('scale must be real')
        return FunctionSignal.scale(self, alpha)

    def __add__(self, other):
        if not isinstance(other, (_ContinuousMixin, Number)):
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
        if not isinstance(other, (_ContinuousMixin, Number)):
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
        if not isinstance(other, (_ContinuousMixin, Number)):
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
        if not isinstance(other, (_ContinuousMixin, Number)):
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
        if not isinstance(other, (_ContinuousMixin, Number)):
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
        if not isinstance(other, (_ContinuousMixin, Number)):
            raise TypeError("can't divide {0}".format(str(other)))
        if other == 1 or other == Constant(1):
            return Constant(1)/self
        s = ContinuousFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__rtruediv__(o, s)


class ContinuousFunctionSignal(_ContinuousMixin, FunctionSignal):
    # Las especializaciones discretas deben ir antes que las funcionales
    @staticmethod
    def _factory(other):
        s = ContinuousFunctionSignal(other._yexpr)
        other._copy_to(s)
        return s

    def __init__(self, expr):
        FunctionSignal.__init__(self, expr)
        _ContinuousMixin.__init__(self)
        self._laplace_transform = None
        self._fourier_transform = None

    def _copy_to(self, other):
        other._laplace_transform = self._laplace_transform
        other._fourier_transform = self._fourier_transform
        _ContinuousMixin._copy_to(self, other)
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
        raise ValueError("delta(t) hasn't maximum")

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
        n = self._default_xvar()
        expr = n*sp.Heaviside(n)
        ContinuousFunctionSignal.__init__(self, expr)
        # delay
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    def _print(self):
        return 'r({0})'.format(str(self._xexpr))

    def max(self):
        return np.inf

    def min(self):
        return 0


class Rect(ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Rect()
        if other:
            other._copy_to(s)
        return s

    def _copy_to(self, other):
        ContinuousFunctionSignal._copy_to(self, other)
        other._width = self._width

    def __init__(self, width=16, delay=0):
        n = self._default_xvar()
        expr = sp.Piecewise((1, sp.Abs(n) < width/2), (0, True))
        ContinuousFunctionSignal.__init__(self, expr)
        self._width = width
        # delay
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    @property
    def width(self):
        return self._width

    def _print(self):
        return 'Pi({0}, {1})'.format(str(self._xexpr), self._width)

    def max(self):
        return 1

    def min(self):
        return 0


class Triang(ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Triang()
        if other:
            other._copy_to(s)
        return s

    def _copy_to(self, other):
        ContinuousFunctionSignal._copy_to(self, other)
        other._width = self._width

    def __init__(self, width=16, delay=0):
        n = self._default_xvar()
        expr = sp.Piecewise((1.0 - 2.0*sp.Abs(n)/width, sp.Abs(n) < width/2),
                            (0, True))
        ContinuousFunctionSignal.__init__(self, expr)
        self._width = width
        # delay
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    @property
    def width(self):
        return self._width

    def _print(self):
        return 'Delta({0}, {1})'.format(str(self._xexpr), self._width)

    def max(self):
        return 1

    def min(self):
        return 0


class _SinCosCExpMixin(object):

    def __init__(self, omega0, phi0):
        self._omega0 = sp.simplify(omega0)
        self._phi0 = self._reduce_phase(phi0)

    def _copy_to(self, other):
        other._omega0 = self._omega0
        other._phi0 = self._phi0

    def _compute_period(self):
        # si omega0 es cero, se puede considerar periodo N = 1
        if self._omega0 == 0:
            return sp.S.One
        return 2*sp.S.Pi/self._omega0

    def _reduce_phase(self, phi):
        ''' Reduce la fase, módulo 2*pi en el intervalo [-pi, pi)
        '''
        phi0 = sp.Mod(phi, 2*sp.S.Pi)
        if phi0 >= sp.S.Pi:
            phi0 -= 2*sp.S.Pi
        return phi0

    @property
    def angular_frequency(self):
        return self._omega0

    @property
    def frequency(self):
        return self._omega0/(2*sp.S.Pi)

    @property
    def phase_offset(self):
        return self._phi0

    @property
    def period(self):
        if self._period is not None:
            return self._period
        self._period = self._compute_period()
        return self._period

    def as_euler(self):
        eu = ContinuousFunctionSignal(self._yexpr.rewrite(sp.exp))
        eu._dtype = np.complex_
        return eu


class Cosine(_SinCosCExpMixin, ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Cosine()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, omega0=1, phi0=0):
        expr = sp.cos(self._default_xvar())
        ContinuousFunctionSignal.__init__(self, expr)
        _SinCosCExpMixin.__init__(self, omega0, phi0)
        # delay (negativo, OJO)
        delay = -self._phi0
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)
        # escalado
        self._xexpr = ScaleOperator.apply(self._xvar, self._xexpr,
                                          self._omega0)
        self._yexpr = ScaleOperator.apply(self._xvar, self._yexpr,
                                          self._omega0)

    def _copy_to(self, other):
        ContinuousFunctionSignal._copy_to(self, other)
        _SinCosCExpMixin._copy_to(self, other)

    def max(self):
        return 1

    def min(self):
        return -1


class Sine(_SinCosCExpMixin, ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Sine()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, omega0=1, phi0=0):
        expr = sp.sin(self._default_xvar())
        ContinuousFunctionSignal.__init__(self, expr)
        _SinCosCExpMixin.__init__(self, omega0, phi0)
        # delay (negativo, OJO)
        delay = -self._phi0
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)
        # escalado
        self._xexpr = ScaleOperator.apply(self._xvar, self._xexpr,
                                          self._omega0)
        self._yexpr = ScaleOperator.apply(self._xvar, self._yexpr,
                                          self._omega0)

    def _copy_to(self, other):
        ContinuousFunctionSignal._copy_to(self, other)
        _SinCosCExpMixin._copy_to(self, other)

    def max(self):
        return 1

    def min(self):
        return -1


class Sinusoid(Cosine):

    @staticmethod
    def _factory(other):
        s = Sinusoid()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, A=1, omega0=1, phi=0):
        Cosine.__init__(self, omega0, phi)
        self._peak_amplitude = A
        self._yexpr *= A

    def _copy_to(self, other):
        other._peak_amplitude = self._peak_amplitude
        Cosine._copy_to(self, other)

    @property
    def peak_amplitude(self):
        return self._peak_amplitude

    @peak_amplitude.setter
    def peak_amplitude(self, value):
        self._yexpr /= self._peak_amplitude
        self._peak_amplitude = value
        self._yexpr *= value

    def max(self):
        return self._peak_amplitude

    def min(self):
        return -self._peak_amplitude

    @property
    def in_phase(self):
        A1 = Constant(self._peak_amplitude * sp.cos(self._phi0))
        return A1*Cosine(self._omega0)

    @property
    def I(self):
        return self.in_phase

    @property
    def in_quadrature(self):
        A2 = Constant(-self._peak_amplitude * sp.sin(self._phi0))
        return A2*Sine(self._omega0)

    @property
    def Q(self):
        return self.in_quadrature



class Exponential(_SinCosCExpMixin, ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Exponential()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, base=1):
        expr = sp.Pow(base, self._default_xvar())
        ContinuousFunctionSignal.__init__(self, expr)
        _SinCosCExpMixin.__init__(self, self._extract_omega(base), 0)
        self._base = sp.sympify(base)
        pb = sp.arg(self._base)
        if pb != sp.nan:
            if pb != 0:
                self._dtype = np.complex_

    def _copy_to(self, other):
        other._base = self._base
        ContinuousFunctionSignal._copy_to(self, other)
        _SinCosCExpMixin._copy_to(self, other)

    @property
    def base(self):
        return self._base

    def is_periodic(self):
        mod1 = sp.Abs(self._base) == 1
        return mod1 and Signal.is_periodic(self)


class ComplexSinusoid(_SinCosCExpMixin, ContinuousFunctionSignal):

    @staticmethod
    def _factory(other):
        s = ComplexSinusoid()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, A=1, omega0=1, phi0=0):
        expr = A*sp.exp(sp.I*self._default_xvar())
        ContinuousFunctionSignal.__init__(self, expr)
        _SinCosCExpMixin.__init__(self, omega0, phi0)
        self._dtype = np.complex_
        # delay (negativo, OJO)
        delay = -self._phi0
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)
        # escalado
        self._xexpr = ScaleOperator.apply(self._xvar, self._xexpr,
                                          self._omega0)
        self._yexpr = ScaleOperator.apply(self._xvar, self._yexpr,
                                          self._omega0)

    def _copy_to(self, other):
        ContinuousFunctionSignal._copy_to(self, other)
        _SinCosCExpMixin._copy_to(self, other)
