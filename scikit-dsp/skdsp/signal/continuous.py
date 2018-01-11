from ._signal import _Signal, _FunctionSignal, _SignalOp
from ._util import _extract_omega, _extract_phase
from ._util import _is_real_scalar, _is_complex_scalar
from numbers import Number
from sympy.core.compatibility import iterable
from sympy.core.evaluate import evaluate, global_evaluate
import numpy as np
import sympy as sp


__all__ = [s for s in dir() if not s.startswith('_')]

# frequently used symbols
t, tau = sp.symbols('t, tau', real=True)


class _ContinuousMixin(object):

    is_Continuous = True
    _default_xvar = sp.Symbol('t', real=True)

    def __init__(self):
        """
        Mixin class initialization.
        """
        pass

    @classmethod
    def default_xvar(cls):
        """
        Default discrete time variable: n
        """
        return cls._default_xvar

    @property
    def period(self):
        """
        ¿Se puede hacer algo?
        """
        return super().period

    @period.setter
    def period(self, value):
        v = sp.sympify(value)
        if _is_real_scalar(v) or v.equals(sp.oo) \
                or isinstance(v, sp.Symbol):
            self._period = v
        else:
            raise ValueError("'period' must be a real scalar")

    @property
    def imag(self):
        s = super().imag
        if s is None:
            return Constant(0)
        return s

    @classmethod
    def _sympify_xexpr(cls, expr):
        expr = sp.sympify(expr)
        if len(expr.free_symbols) == 0:
            raise ValueError('xexpr must be have at least a variable')
        return expr

    def _check_indexes(self, x):
        """
        Checks if all values in x are reals.
        """
        try:
            xlambda = sp.lambdify(self._xvar, self.xexpr, 'numpy')
            x = xlambda(x)
            if not hasattr(x, "__len__"):
                # workaround para issue #5642 de sympy. Cuando yexpr es una
                # constante, se devuelve un escalar aunque la entrada sea un
                # array
                x = np.full(x.shape, x, self.dtype)
        except (NameError, ValueError):
            # sympy no ha podido hacer una función lambda
            # (o hay algún problema de cálculo, p.e 2^(-1) enteros)
            # así que se procesan los valores uno a uno
            x = np.zeros_like(x, self.dtype)
            for k, x0 in enumerate(x):
                try:
                    x[k] = self.xexpr.xreplace({self._xvar: x0})
                except TypeError:
                    x[k] = np.nan

        if not np.all(np.isreal(x)):
            raise ValueError('continuous signals are only defined' +
                             'for real indexes')

    def _post_op(self, other, xexpr, yexpr):
        if yexpr.is_constant():
            return Constant(yexpr)
        cmplx = self.dtype == np.complex_ or other.dtype == np.complex_
        return ContinuousFunctionSignal(xexpr, yexpr, cmplx=cmplx)

    def _add(self, other):
        if other.yexpr == sp.S.Zero:
            return self
        if self.yexpr == sp.S.Zero:
            return other
        if isinstance(self, _FunctionSignal):
            if isinstance(other, _FunctionSignal):
                yexpr1, xvar1 = self.yexpr, self.xvar
                yexpr2, xvar2 = other.yexpr, other.xvar
                yexpr = yexpr1 + yexpr2.subs(xvar2, xvar1)
                return self._post_op(other, xvar1, yexpr)

    def _mul(self, other):
        if other.yexpr == sp.S.One:
            return self
        if self.yexpr == sp.S.One:
            return other
        if self.yexpr == sp.S.Zero or other.yexpr == sp.S.Zero:
            return Constant(0)
        if isinstance(self, _FunctionSignal):
            if isinstance(other, _FunctionSignal):
                yexpr1, xvar1 = self.yexpr, self.xvar
                yexpr2, xvar2 = other.yexpr, other.xvar
                yexpr = yexpr1 * yexpr2.subs(xvar2, xvar1)
                return self._post_op(other, xvar1, yexpr)

    # --- independent variable operations -------------------------------------
    def shift(self, tau):
        """
        Delays (or advances) the signal k (integer) units.
        """
        if not _is_real_scalar(tau):
            raise TypeError('delay/advance must be real')
        with evaluate(True):
            s = _FunctionSignal.shift(self, tau)
        return s

    def delay(self, tau):
        return self.shift(tau)

    def scale(self, alpha):
        """ Scales the independent variable by `s`."""
        if not _is_real_scalar(alpha):
            raise TypeError('scale factor must be real')
        r = sp.sympify(alpha)
        if r == sp.S.One:
            return self
        elif r < sp.S.One:
            return self.compress(r)
        else:
            return self.expand(r)

    def compress(self, alpha):
        if alpha == sp.S.One:
            return self
        return _Signal.scale(self, alpha, mul=False)

    def expand(self, alpha):
        if alpha == sp.S.One:
            return self
        return _Signal.scale(self, alpha, mul=True)

    def __rshift__(self, k):
        return _ContinuousMixin.shift(self, k)

    __irshift__ = __rshift__

    def __lshift__(self, k):
        return _ContinuousMixin.shift(self, -k)

    __ilshift__ = __lshift__

    # --- math wrappers -------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _ContinuousMixin):
            raise TypeError("can't add continuous signal and {0}"
                            .format(type(other)))
        return ContinuousSignalAdd(self, other)

    def __radd__(self, other):
        return self + other

    __iadd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _ContinuousMixin):
            raise TypeError("can't add continuous signal and {0}"
                            .format(type(other)))
        return ContinuousSignalAdd(self, -other)

    def __rsub__(self, other):
        return (-self) + other

    __isub__ = __sub__

    def __neg__(self):
        with evaluate(True):
            return Constant(-1) * self

    def __mul__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _ContinuousMixin):
            raise TypeError("can't mul continuous signal and {0}"
                            .format(type(other)))
        return ContinuousSignalMul(self, other)

    def __rmul__(self, other):
        return self * other

    __imul__ = __mul__

    def __pow__(self, other):
        raise _SignalOp.OperationNotDefined

    __ipow__ = __pow__

    def __truediv__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _ContinuousMixin):
            raise TypeError("can't div continuous signal and {0}"
                            .format(type(other)))
        return ContinuousSignalMul(self, pow(other, -1))

    def __rtruediv__(self, other):
        return other * pow(self, -1)

    __itruediv__ = __truediv__


class _ContinuousSignalOp(_ContinuousMixin, _SignalOp):
    is_Continuous = True


class ContinuousSignalAdd(_ContinuousSignalOp):

    def __init__(self, *_args):
        _Signal.__init__(self)
        s0 = self.args[0]
        s1 = self.args[1]
        s1.xvar = s0.xvar
        self.period = sp.lcm(s0.period, s1.period)

    def __new__(cls, *args):
        evaluate = global_evaluate[0]

        # flatten inputs
        args = list(args)

        # adapted from sequences.SeqAdd
        def _flatten(arg):
            if isinstance(arg, _Signal):
                if isinstance(arg, ContinuousSignalAdd):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            if iterable(arg):
                return sum(map(_flatten, arg), [])
            raise TypeError("Input must be signals or "
                            " iterables of signals")

        args = _flatten(args)

        # reduce using known rules
        if evaluate:
            return ContinuousSignalAdd.reduce(args)

        return _Signal.__new__(cls, *args)

    @staticmethod
    def reduce(args):
        """
        Simplify :class:`SignalAdd` using known rules.
        """
        new_args = True
        while(new_args):
            for id1, s in enumerate(args):
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    new_sgn = s._add(t)
                    # This returns None if s does not know how to add
                    # with t. Returns the newly added signal otherwise
                    if new_sgn is not None:
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_sgn)
                        break
                if new_args:
                    args = new_args
                    break

        if len(args) == 1:
            return args.pop()
        else:
            return ContinuousSignalAdd(args)

    def eval(self, r):
        return sum(a.eval(r) for a in self.args)


class ContinuousSignalMul(_ContinuousSignalOp):

    def __init__(self, *_args):
        _Signal.__init__(self)
        s0 = self.args[0]
        s1 = self.args[1]
        s1.xvar = s0.xvar
        self.period = sp.lcm(s0.period, s1.period)

    def __new__(cls, *args):
        evaluate = global_evaluate[0]

        # flatten inputs
        args = list(args)

        # adapted from sequences.SeqMul
        def _flatten(arg):
            if isinstance(arg, _Signal):
                if isinstance(arg, ContinuousSignalMul):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            if iterable(arg):
                return sum(map(_flatten, arg), [])
            raise TypeError("Input must be signals or "
                            " iterables of signals")

        args = _flatten(args)

        # reduce using known rules
        if evaluate:
            return ContinuousSignalMul.reduce(args)

        return _Signal.__new__(cls, *args)

    @staticmethod
    def reduce(args):
        """
        Simplify :class:`SignalMul` using known rules.
        """
        new_args = True
        while(new_args):
            for id1, s in enumerate(args):
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    new_sgn = s._mul(t)
                    # This returns None if s does not know how to add
                    # with t. Returns the newly added signal otherwise
                    if new_sgn is not None:
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_sgn)
                        break
                if new_args:
                    args = new_args
                    break

        if len(args) == 1:
            return args.pop()
        else:
            return ContinuousSignalMul(args)

    def eval(self, r):
        val = 1
        for a in self.args:
            val *= a.eval(r)
        return val


class ContinuousFunctionSignal(_ContinuousMixin, _FunctionSignal):
    def __new__(cls, _xexpr, _yexpr, **_kwargs):
        return object.__new__(cls)

    def __init__(self, xexpr, yexpr, **kwargs):
        _FunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        _ContinuousMixin.__init__(self)
        self._laplace_transform = None
        self._fourier_transform = None

    @property
    def laplace(self):
        return self._laplace_transform

    @property
    def fourier(self):
        return self._fourier_transform

    def eval(self, x):
        is_scalar = False
        if isinstance(x, np.ndarray):
            if x.size == 1:
                is_scalar = True
        else:
            is_scalar = True
            x = np.array([x])
        self._check_indexes(x)
        y = _FunctionSignal.eval(self, x)
        if is_scalar:
            y = np.asscalar(y)
        return y


class Constant(ContinuousFunctionSignal):

    """
    Continuous constant signal. Not a degenerate case for constant functions
    such as `A*cos(0)`, `A*sin(pi/2)`, `A*exp(0*n)`, althought it could be.
    """
    is_finite = True

    def __new__(cls, const=0, **_kwargs):
        return _Signal.__new__(cls, const)

    def __init__(self, const=0, **kwargs):
        const = sp.sympify(const)
        if not const.is_constant():
            raise ValueError('const value is not constant')
        ContinuousFunctionSignal.__init__(self,
                                          _ContinuousMixin.default_xvar(),
                                          const, **kwargs)
        if _is_complex_scalar(const):
            self.dtype = np.complex_
        # period
        self._period = sp.oo
        # TODO transformadas

    def __str__(self, *_args, **_kwargs):
        return str(self._yexpr)

    def __repr__(self):
        return 'Constant({0})'.format(self._yexpr)


class Delta(ContinuousFunctionSignal):
    """
    Continuous unit impulse signal.
    """

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, xexpr=_ContinuousMixin._default_xvar, **_kwargs):
        return _Signal.__new__(cls, xexpr)

    def __init__(self, xexpr=_ContinuousMixin._default_xvar, **kwargs):
        xexpr = _ContinuousMixin._sympify_xexpr(xexpr)
        yexpr = sp.DiracDelta(xexpr)
        ContinuousFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self._period = sp.oo
        # TODO transformadas

    def __str__(self, *_args, **_kwargs):
        return 'delta({0})'.format(str(self.xexpr))

    def __repr__(self, *_args, **_kwargs):
        return 'Delta({0})'.format(str(self.xexpr))


class Step(ContinuousFunctionSignal):
    """
    Unit step signal.
    """

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, xexpr=_ContinuousMixin._default_xvar, **_kwargs):
        return _Signal.__new__(cls, xexpr)

    def __init__(self, xexpr=_ContinuousMixin._default_xvar, **kwargs):
        xexpr = _ContinuousMixin._sympify_xexpr(xexpr)
        yexpr = sp.Heaviside(xexpr)
        ContinuousFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self._period = sp.oo
        # TODO transformadas

    def __str__(self):
        return 'u({0})'.format(str(self.xexpr))

    def __repr__(self):
        return 'Step({0})'.format(str(self.xexpr))

