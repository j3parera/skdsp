from numbers import Integral, Number, Real
from skdsp.operator.operator import ShiftOperator, ScaleOperator
from skdsp.signal.signal import FunctionSignal, ConstantSignal, Signal
import numpy as np
import sympy as sp
from sympy.functions.elementary.trigonometric import _pi_coeff

__all__ = ['DiscreteMixin', 'DiscreteFunctionSignal', 'DataSignal'
           'Constant', 'Delta', 'Step', 'Ramp',
           'Cosine', 'Sine', 'Sinusoid'
           'ComplexExponential', 'Exponential',
           'Sawtooth']


class DiscreteMixin(object):

    @staticmethod
    def _is_integer(x):
        return np.all(np.equal(np.mod(x, 1), 0))

    @staticmethod
    def _check_eval(x):
        if not DiscreteMixin._is_integer(x):
            raise TypeError('discrete signal defined only for integer indexes')

    @staticmethod
    def _check_is_real(x):
        ok = True
        if isinstance(x, sp.Expr):
            x = x.evalf()
            if not isinstance(x, sp.Float):
                ok = False
        elif not isinstance(x, Real):
            ok = False
        return ok

    @staticmethod
    def _check_is_integer(x):
        ok = True
        if isinstance(x, sp.Expr):
            x = x.evalf()
            if not isinstance(x, sp.Integer):
                ok = False
        elif not isinstance(x, Integral):
            ok = False
        return ok

    @staticmethod
    def _default_xvar():
        return sp.symbols('n', integer=True)

    def __init__(self):
        pass

    def _copy_to(self, other):
        pass

    def flip(self):
        o = FunctionSignal.flip(self)
        return o

    __reversed__ = flip

    def shift(self, k):
        if not self._check_is_integer(k):
            raise TypeError('delay/advance must be integer')
        o = FunctionSignal.shift(self, k)
        return o

    def delay(self, d):
        # versión de retardo no necesariamente entero
        if not self._check_is_real(d):
            raise TypeError('delay/advance must be real')
        o = FunctionSignal.shift(self, d)
        return o

    def __rshift__(self, k):
        return DiscreteMixin.shift(self, k)

    __irshift__ = __rshift__

    def __lshift__(self, k):
        return DiscreteMixin.shift(self, -k)

    __ilshift__ = __lshift__

    def scale(self, v):
        # Nota: scale permite cualquier valor de v, no necesariamente entero
        if not self._check_is_real(v):
            raise TypeError('scale must be real')
        o = FunctionSignal.scale(self, v)
        return o

#     def compress(self, alpha):
#         if not isinstance(alpha, Integral):
#             raise TypeError('compress factor must be integer')
#         return Signal.compress(self, alpha)
#
#     def expand(self, beta):
#         if not isinstance(beta, Integral):
#             raise TypeError('expand factor must be integer')
#         return Signal.expand(self, beta)
#
#     def ccshift(self, k, N):
#         if not isinstance(k, Integral):
#             raise TypeError('delay/advance must be integer')
#         if not isinstance(N, Integral):
#             raise TypeError('modulo length must be integer')
#         return Signal.cshift(self, k, N)

    def __add__(self, other):
        if not isinstance(other, (DiscreteMixin, Number)):
            raise TypeError("can't add {0}".format(str(other)))
        if other == 0 or other == Constant(0):
            return self
        if self == 0 or self == Constant(0):
            if isinstance(other, Number):
                return Constant(other)
            else:
                return other
        s = DiscreteFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__add__(s, o)

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, (DiscreteMixin, Number)):
            raise TypeError("can't sub {0}".format(str(other)))
        if other == 0 or other == Constant(0):
            return self
        if self == 0 or self == Constant(0):
            if isinstance(other, Number):
                return Constant(-other)
            else:
                return -other
        s = DiscreteFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__sub__(s, o)

    def __rsub__(self, other):
        if not isinstance(other, (DiscreteMixin, Number)):
            raise TypeError("can't sub {0}".format(str(other)))
        if other == 0 or other == Constant(0):
            return -self
        if self == 0 or self == Constant(0):
            if isinstance(other, Number):
                return Constant(other)
            else:
                return other
        s = DiscreteFunctionSignal._factory(self)
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
        if not isinstance(other, (DiscreteMixin, Number)):
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
            s = DiscreteFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__mul__(s, o)

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, (DiscreteMixin, Number)):
            raise TypeError("can't divide {0}".format(str(other)))
        if other == 1 or other == Constant(1):
            return self
        if isinstance(other, (Constant, Number)):
            s = self.__class__._factory(self)
        else:
            s = DiscreteFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__truediv__(s, o)

    __itruediv__ = __truediv__

    def __rtruediv__(self, other):
        if not isinstance(other, (DiscreteMixin, Number)):
            raise TypeError("can't divide {0}".format(str(other)))
        if other == 1 or other == Constant(1):
            return Constant(1)/self
        s = DiscreteFunctionSignal._factory(self)
        if isinstance(other, Number):
            o = Constant(other)
        else:
            o = other
        return FunctionSignal.__rtruediv__(o, s)

    def __neg__(self):
        o = FunctionSignal.__neg__(self)
        return o

    def __eq__(self, other):
        # TODO: ¿es correcto?
        return str(self).__eq__(str(other))
#         if isinstance(other, FunctionSignal):
#             return self._yexpr == other._yexpr
#         d = self._yexpr - other
#         if (sp.expand(d) == 0) or \
#            (sp.simplify(d) == 0) or \
#            (sp.trigsimp(d) == 0):
#             return True
#         else:
#             return False

    def dfs(self, P=None, force=False, symbolic=False):
        # es difícil determinar si una señal es periódica, así que suponemos
        # que si se quiere calcular la DFS es porque se sabe que es periódica
        # de periodo P (que no debe ser None)
        if not force and not self.is_periodic():
            raise TypeError('cant compute DFS of non periodic signal')
        if force:
            N = P
        else:
            # periodo de la señal
            N = int(self._period)
            if P is not None:
                # P debe ser un múltiplo de N
                if P % N != 0:
                    raise ValueError('P must be a multiple of N = {0}'
                                     .format(N))
        X = np.zeros(N, np.complex_)
        if symbolic:
            n = self._xvar
            for k0 in np.arange(0, N):
                X[k0] = sp.Sum(self._yexpr*sp.exp(-sp.I*2*sp.S.Pi*k0*n/N),
                               (n, 0, N-1))
        else:
            n = np.arange(0, N)
            x = self.eval(n)
            for k0 in np.arange(0, N):
                X[k0] = np.sum(x*np.exp(-1j*2*np.pi*k0*n/N))
        return X


class DataSignal(Signal, DiscreteMixin):

    def __init__(self, data, span):
        super.__init__()
        self._data = data
        self._span = span


class DiscreteFunctionSignal(DiscreteMixin, FunctionSignal):
    # Las especializaciones discretas deben ir antes que las funcionales
    @staticmethod
    def _factory(other):
        s = DiscreteFunctionSignal(other._yexpr)
        other._copy_to(s)
        return s

    def __init__(self, expr):
        FunctionSignal.__init__(self, expr)
        DiscreteMixin.__init__(self)
        self._zt = None
        self._dtft = None

    def _copy_to(self, other):
        other._zt = self._zt
        other._dtft = self._dtft
        DiscreteMixin._copy_to(self, other)
        FunctionSignal._copy_to(self, other)

    @property
    def zt(self):
        return self._zt

    @property
    def dtft(self):
        return self._dtft

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

    def eval(self, x, force=False):
        is_scalar = False
        if not isinstance(x, np.ndarray):
            is_scalar = True
            x = np.array([x])
        if not force:
            DiscreteMixin._check_eval(x)
        y = FunctionSignal.eval(self, x)
        if is_scalar:
            y = np.asscalar(y)
        return y

    def __abs__(self):
        s = DiscreteFunctionSignal._factory(self)
        return FunctionSignal.__abs__(s)

    def magnitude(self, dB=False):
        m = abs(self)
        if dB:
            mdb = DiscreteFunctionSignal._factory(self)
            mdb._yexpr = 20*sp.log(m._yexpr, 10)
            return mdb
        return m

    def dynamic_range(self, dB=False):
        dr = self.max() - self.min()
        if dB:
            return 20*sp.log(dr, 10)
        return dr


class Constant(DiscreteFunctionSignal, ConstantSignal):

    @staticmethod
    def _factory(other, cte):
        s = Constant(cte)
        if other:
            other._copy_to(s)
        return s

    def __init__(self, c):
        DiscreteFunctionSignal.__init__(self, sp.sympify(c))
        ConstantSignal.__init__(self, c, self._xvar)

    def max(self):
        return self._yexpr

    def min(self):
        return self._yexpr


class Delta(DiscreteFunctionSignal):

    class _DiscreteDelta(sp.Function):

        @classmethod
        def eval(cls, arg):
            arg = sp.sympify(arg)
            if arg is sp.S.NaN:
                return sp.S.NaN
            elif arg.is_negative or arg.is_positive:
                return sp.S.Zero
            elif arg.is_zero:
                return sp.S.One

        @staticmethod
        def _imp_(n):
            # i es 0, siempre
            return np.equal(n, 0).astype(np.float_)

    @staticmethod
    def _factory(other):
        s = Delta()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, delay=0):
        expr = Delta._DiscreteDelta(self._default_xvar())
        DiscreteFunctionSignal.__init__(self, expr)
        # delay
        if not isinstance(delay, Integral):
            raise TypeError('delay/advance must be integer')
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    def max(self):
        return sp.S.One

    def min(self):
        return sp.S.Zero


class Step(DiscreteFunctionSignal):

    class _DiscreteStep(sp.Function):

        @classmethod
        def eval(cls, arg):
            arg = sp.sympify(arg)
            if arg is sp.S.NaN:
                return sp.S.NaN
            elif arg.is_negative:
                return sp.S.Zero
            elif arg.is_zero or arg.is_positive:
                return sp.S.One

        @staticmethod
        def _imp_(n):
            return np.greater_equal(n, 0).astype(np.float_)

    @staticmethod
    def _factory(other):
        s = Step()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, delay=0):
        expr = Step._DiscreteStep(self._default_xvar())
        DiscreteFunctionSignal.__init__(self, expr)
        # delay
        if not isinstance(delay, Integral):
            raise TypeError('delay/advance must be integer')
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    def max(self):
        return sp.S.One

    def min(self):
        return sp.S.Zero


class Ramp(DiscreteFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Ramp()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, delay=0):
        expr = self._default_xvar()
        DiscreteFunctionSignal.__init__(self, expr)
        # delay
        if not isinstance(delay, Integral):
            raise TypeError('delay/advance must be integer')
        self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
        self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)

    def _print(self):
        return 'r[{0}]'.format(str(self._xexpr))

    def max(self):
        return sp.oo

    def min(self):
        return sp.S.Zero


class SinCosCExpMixin(object):

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
        # trata de simplificar 2*pi/omega como racional
        sNk = sp.nsimplify(2*sp.S.Pi/self._omega0, rational=True)
        if sp.ask(sp.Q.rational(sNk)):
            # si es racional, el numerador es el periodo
            return sp.Rational(sNk).p
        return np.Inf

    def _reduce_phase(self, phi):
        ''' Reduce la fase, módulo 2*pi en el intervalo [-pi, pi)
        '''
        phi0 = sp.Mod(phi, 2*sp.S.Pi)
        if phi0 >= sp.S.Pi:
            phi0 -= 2*sp.S.Pi
        return phi0

    @property
    def frequency(self):
        return self._omega0

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
        dfs = DiscreteFunctionSignal(self._yexpr.rewrite(sp.exp))
        dfs._dtype = np.complex_
        return dfs


class Cosine(SinCosCExpMixin, DiscreteFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Cosine()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, omega0=1, phi0=0):
        expr = sp.cos(self._default_xvar())
        DiscreteFunctionSignal.__init__(self, expr)
        SinCosCExpMixin.__init__(self, omega0, phi0)
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
        DiscreteFunctionSignal._copy_to(self, other)
        SinCosCExpMixin._copy_to(self, other)

    def max(self):
        return 1

    def min(self):
        return -1


class Sine(SinCosCExpMixin, DiscreteFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Sine()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, omega0=1, phi0=0):
        expr = sp.sin(self._default_xvar())
        DiscreteFunctionSignal.__init__(self, expr)
        SinCosCExpMixin.__init__(self, omega0, phi0)
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
        DiscreteFunctionSignal._copy_to(self, other)
        SinCosCExpMixin._copy_to(self, other)

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

    def in_phase(self):
        A1 = Constant(self._peak_amplitude * sp.cos(self._phi0))
        return A1*Cosine(self._omega0)

    def in_quadrature(self):
        A2 = Constant(-self._peak_amplitude * sp.sin(self._phi0))
        return A2*Sine(self._omega0)


class Exponential(SinCosCExpMixin, DiscreteFunctionSignal):

    @staticmethod
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

    @staticmethod
    def _factory(other):
        s = Exponential()
        if other:
            other._copy_to(s)
        return s

    def __init__(self, base=1):
        expr = sp.Pow(base, self._default_xvar())
        DiscreteFunctionSignal.__init__(self, expr)
        SinCosCExpMixin.__init__(self, self._extract_omega(base), 0)
        self._base = sp.sympify(base)
        pb = sp.arg(self._base)
        if pb != sp.nan:
            if pb != 0:
                self._dtype = np.complex_

    def _copy_to(self, other):
        other._base = self._base
        DiscreteFunctionSignal._copy_to(self, other)
        SinCosCExpMixin._copy_to(self, other)

    @property
    def base(self):
        return self._base

    def is_periodic(self):
        mod1 = sp.Abs(self._base) == 1
        return mod1 and Signal.is_periodic(self)

    def culo(self):
        return self._period


# class ComplexSinusoid(SinCosCExpMixin):
#
#     @staticmethod
#     def _factory(other):
#         s = ComplexSinusoid()
#         if other:
#             other._copy_to(s)
#         return s
#
#     def __init__(self, A=1, omega0=1, phi0=0):
#         SinCosCExpMixin.__init__(self, omega0, phi0)
#         self._dtype = np.complex_
#         self._yexpr = A*sp.exp(sp.I*self._xvar)
#          delay (negativo, OJO)
#         delay = -self._phi0
#         self._xexpr = ShiftOperator.apply(self._xvar, self._xexpr, delay)
#         self._yexpr = ShiftOperator.apply(self._xvar, self._yexpr, delay)
#          escalado
#         self._xexpr = ScaleOperator.apply(self._xvar, self._xexpr,
#                                           self._omega0)
#         self._yexpr = ScaleOperator.apply(self._xvar, self._yexpr,
#                                           self._omega0)
#
#     def _copy_to(self, other):
#         DiscreteFunctionSignal._copy_to(self, other)
#         SinCosCExpMixin._copy_to(self, other)

class Sawtooth(DiscreteFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Sawtooth()
        if other:
            other._copy_to(s)
        return s

    def _copy_to(self, other):
        DiscreteFunctionSignal._copy_to(self, other)
        other._period = self._period
        other._width = self._width

    def __init__(self, N=16, width=None):
        if width is None:
            width = N
        if N <= 0:
            raise ValueError('N must be greater than 0')
        if width > N:
            raise ValueError('width must be less than N')
        nm = sp.Mod(self._default_xvar(), N)
        expr = sp.Piecewise((-1+(2*nm)/width, nm < width),
                            (1-2*(nm-width)/(N-width), nm < N))
        DiscreteFunctionSignal.__init__(self, expr)
        self._period = N
        self._width = width

    def max(self):
        return 1

    def min(self):
        return -1

    def _print(self):
        return 'saw[{0}, {1}, {2}]'.format(str(self._xexpr), self._period,
                                           self._width)


class Square(DiscreteFunctionSignal):

    @staticmethod
    def _factory(other):
        s = Square()
        if other:
            other._copy_to(s)
        return s

    def _copy_to(self, other):
        DiscreteFunctionSignal._copy_to(self, other)
        other._period = self._period
        other._width = self._width

    def __init__(self, N=16, width=None):
        if width is None:
            width = N
        if N <= 0:
            raise ValueError('N must be greater than 0')
        if width > N:
            raise ValueError('width must be less than N')
        nm = sp.Mod(self._default_xvar(), N)
        expr = sp.Piecewise((-1, nm < width), (1, nm < N))
        DiscreteFunctionSignal.__init__(self, expr)
        self._period = N
        self._width = width

    def max(self):
        return 1

    def min(self):
        return -1

    def _print(self):
        return 'square[{0}, {1}, {2}]'.format(str(self._xexpr), self._period,
                                              self._width)
