from ._signal import _Signal, _FunctionSignal, _SignalOp
from ._util import _extract_omega, _extract_phase
from ._util import _is_complex_scalar, _is_integer_scalar
from numbers import Number
from sympy.core.compatibility import iterable
from sympy.core.evaluate import evaluate, global_evaluate
import numpy as np
import sympy as sp

# frequently used symbols
n, m, k = sp.symbols('n, m, k', integer=True)


# functions
class UnitDelta(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_zero:
            return sp.S.One
        elif arg.is_nonzero:
            return sp.S.Zero

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((1, sp.Eq(arg, 0)), (0, True))

    def _eval_rewrite_as_UnitStep(self, arg):
        return UnitStep(arg) - UnitStep(arg-1)

    @staticmethod
    def _imp_(n):
        return np.equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\delta'


class UnitStep(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_negative:
            return sp.S.Zero
        elif arg.is_nonnegative:
            return sp.S.One

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((1, arg >= 0), (0, True))

    def _eval_rewrite_as_UnitDelta(self, arg):
        k = sp.Dummy(integer=True)
        return sp.Sum(UnitDelta(k), (k, -sp.oo, arg))

    def _eval_rewrite_as_UnitRamp(self, arg):
        return UnitRamp(arg+1) - UnitRamp(arg)

    @staticmethod
    def _imp_(n):
        return np.greater_equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\u'


class UnitRamp(sp.Function):

    nargs = 1
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_negative:
            return sp.S.Zero
        elif arg.is_nonnegative:
            return arg

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((arg, arg >= 0), (0, True))

    def _eval_rewrite_as_UnitStep(self, arg):
        return arg*UnitStep(arg)

    def _eval_rewrite_as_Max(self, arg):
        return sp.Max(0, arg)

    def _eval_rewrite_as_Abs(self, arg):
        return sp.S.Half*(arg + sp.Abs(arg))

    @staticmethod
    def _imp_(n):
        return n*np.greater_equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\r'


# functions
class UnitDelta(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_zero:
            return sp.S.One
        elif arg.is_nonzero:
            return sp.S.Zero

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((1, sp.Eq(arg, 0)), (0, True))

    def _eval_rewrite_as_UnitStep(self, arg):
        return UnitStep(arg) - UnitStep(arg-1)

    @staticmethod
    def _imp_(n):
        return np.equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\delta'


class UnitStep(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_negative:
            return sp.S.Zero
        elif arg.is_nonnegative:
            return sp.S.One

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((1, arg >= 0), (0, True))

    def _eval_rewrite_as_UnitDelta(self, arg):
        k = sp.Dummy(integer=True)
        return sp.Sum(UnitDelta(k), (k, -sp.oo, arg))

    def _eval_rewrite_as_UnitRamp(self, arg):
        return sp.Sum(UnitRamp(arg+1) - UnitRamp(arg))

    @staticmethod
    def _imp_(n):
        return np.greater_equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\u'


class UnitRamp(sp.Function):

    nargs = 1
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_negative:
            return sp.S.Zero
        elif arg.is_nonnegative:
            return arg

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((arg, arg >= 0), (0, True))

    def _eval_rewrite_as_UnitStep(self, arg):
        return arg*UnitStep(arg)

    def _eval_rewrite_as_Max(self, arg):
        return sp.Max(0, arg)

    def _eval_rewrite_as_Abs(self, arg):
        return sp.S.Half*(arg + sp.Abs(arg))

    @staticmethod
    def _imp_(n):
        return n*np.greater_equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\r'


# functions
class UnitDelta(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_zero:
            return sp.S.One
        elif arg.is_nonzero:
            return sp.S.Zero

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((1, sp.Eq(arg, 0)), (0, True))

    def _eval_rewrite_as_UnitStep(self, arg):
        return UnitStep(arg) - UnitStep(arg-1)

    @staticmethod
    def _imp_(n):
        return np.equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\delta'


class UnitStep(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_negative:
            return sp.S.Zero
        elif arg.is_nonnegative:
            return sp.S.One

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((1, arg >= 0), (0, True))

    def _eval_rewrite_as_UnitDelta(self, arg):
        k = sp.Dummy(integer=True)
        return sp.Sum(UnitDelta(k), (k, -sp.oo, arg))

    def _eval_rewrite_as_UnitRamp(self, arg):
        return sp.Sum(UnitRamp(arg+1) - UnitRamp(arg))

    @staticmethod
    def _imp_(n):
        return np.greater_equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\u'


class UnitRamp(sp.Function):

    nargs = 1
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if sp.logic.fuzzy_not(arg.is_integer):
            raise ValueError('Function defined only for integer values')
        if arg.is_negative:
            return sp.S.Zero
        elif arg.is_nonnegative:
            return arg

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((arg, arg >= 0), (0, True))

    def _eval_rewrite_as_UnitStep(self, arg):
        return arg*UnitStep(arg)

    def _eval_rewrite_as_Max(self, arg):
        return sp.Max(0, arg)

    def _eval_rewrite_as_Abs(self, arg):
        return sp.S.Half*(arg + sp.Abs(arg))

    @staticmethod
    def _imp_(n):
        return n*np.greater_equal(n, 0).astype(np.float_)

    @staticmethod
    def _latex_no_arg(_printer):
        return r'\r'


class _DiscreteMixin(object):
    """
    Base mixin class for discrete signals.
    Defines all common operations for this kind of signals
    and other useful stuff.
    """

    is_Discrete = True
    _default_xvar = sp.Symbol('n', integer=True)

    @property
    def period(self):
        """
        ¿Se puede hacer algo?
        """
        return super().period

    @period.setter
    def period(self, value):
        v = sp.sympify(value)
        if _is_integer_scalar(v) or v.equals(sp.oo) \
                or isinstance(v, sp.Symbol):
            self._period = v
        else:
            raise ValueError("'period' must be an integer scalar")

    @property
    def imag(self):
        s = super().imag
        if s is None:
            return Constant(0)
        return s

    @classmethod
    def _sympify_xexpr(cls, expr):
        expr = sp.sympify(expr)
        if len(expr.free_symbols) == 0 or not expr.is_integer:
            raise ValueError('xexpr must be an integer expression ' +
                             'with at least a variable')
        return expr

    def _check_indexes(self, x):
        """
        Checks if all values in x are integers (including floats
        without fractional part (e.g. 1.0, 2.0).
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

        if not np.all(np.equal(np.mod(np.asanyarray(x), 1), 0)):
            raise ValueError('discrete signals are only defined' +
                             'for integer indexes')

    @classmethod
    def default_xvar(cls):
        """
        Default discrete time variable: n
        """
        return cls._default_xvar

    def __init__(self):
        """
        Mixin class initialization.
        """
        pass

    def _post_op(self, other, xexpr, yexpr):
        if yexpr.is_constant():
            return Constant(yexpr)
        cmplx = self.dtype == np.complex_ or other.dtype == np.complex_
        return DiscreteFunctionSignal(xexpr, yexpr, cmplx=cmplx)

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

    def _pow(self, other):
        if other.yexpr == sp.S.One:
            return self
        if other.yexpr == sp.S.Zero:
            return Constant(0)
        if isinstance(self, _FunctionSignal):
            if isinstance(other, _FunctionSignal):
                yexpr1, xvar1 = self.yexpr, self.xvar
                yexpr2, xvar2 = other.yexpr, other.xvar
                yexpr = yexpr1 ** yexpr2.subs(xvar2, xvar1)
                return self._post_op(other, xvar1, yexpr)

    # --- independent variable operations -------------------------------------
    def shift(self, k):
        """
        Delays (or advances) the signal k (integer) units.
        """
        if not _is_integer_scalar(k):
            raise ValueError('delay/advance must be integer')
        if isinstance(self, _FunctionSignal):
            o = _FunctionSignal.shift(self, k)
        elif isinstance(self, DataSignal):
            o = DataSignal.shift(self, k)
        return o

    def delay(self, k):
        """
        Quizás D/C - retardo continuo -- C/D
        """
        return self.shift(k)

    def scale(self, s):
        """ Scales the independent variable by `s`."""
        r = sp.nsimplify(s, rational=True)
        if not isinstance(r, sp.Rational):
            raise ValueError('expansion/compression value not rational')
        return self.expand(r.p).compress(r.q)

    def compress(self, n):
        if not _is_integer_scalar(n):
            raise ValueError('compress factor must be integer')
        if n == sp.S.One:
            return self
        return _Signal.scale(self, n, mul=False)

    def expand(self, n):
        if not _is_integer_scalar(n):
            raise ValueError('expand factor must be integer')
        if n == sp.S.One:
            return self
        return _Signal.scale(self, n, mul=True)

    def __rshift__(self, k):
        return _DiscreteMixin.shift(self, k)

    __irshift__ = __rshift__

    def __lshift__(self, k):
        return _DiscreteMixin.shift(self, -k)

    __ilshift__ = __lshift__

    # --- math wrappers -------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _DiscreteMixin):
            raise TypeError("can't add discrete signal and {0}"
                            .format(type(other)))
        return DiscreteSignalAdd(self, other)

    def __radd__(self, other):
        return self + other

    __iadd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _DiscreteMixin):
            raise TypeError("can't add discrete signal and {0}"
                            .format(type(other)))
        return DiscreteSignalAdd(self, -other)

    def __rsub__(self, other):
        return (-self) + other

    __isub__ = __sub__

    def __neg__(self):
        with evaluate(True):
            return Constant(-1) * self

    def __mul__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _DiscreteMixin):
            raise TypeError("can't mul discrete signal and {0}"
                            .format(type(other)))
        return DiscreteSignalMul(self, other)

    def __rmul__(self, other):
        return self * other

    __imul__ = __mul__

    def __pow__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _DiscreteMixin):
            raise TypeError("can't pow discrete signal and {0}"
                            .format(type(other)))
        return DiscreteSignalPow(self, other)

    __ipow__ = __pow__

    def __truediv__(self, other):
        if isinstance(other, Number):
            other = Constant(other)
        if not isinstance(other, _DiscreteMixin):
            raise TypeError("can't div discrete signal and {0}"
                            .format(type(other)))
        return DiscreteSignalMul(self, pow(other, -1))

    def __rtruediv__(self, other):
        return other * pow(self, -1)

    __itruediv__ = __truediv__

    # --- other operations ----------------------------------------------------
#     def ccshift(self, k, N):
#         if not isinstance(k, Integral):
#             raise ValueError('delay/advance must be integer')
#         if not isinstance(N, Integral):
#             raise ValueError('modulo length must be integer')
#         return _Signal.cshift(self, k, N)

    def dfs(self, P=None, force=False, symbolic=False):
        if not force and not self.is_periodic():
            raise TypeError('cant compute DFS of non periodic signal')
        # es difícil determinar si una señal es periódica, así que suponemos
        # que si se quiere calcular la DFS es porque se sabe que es periódica
        # de periodo P (que no debe ser None)
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
        if symbolic:
            n = self._xvar
            k = sp.symbols('k', integer=True)
            Xs = sp.Sum(self._yexpr*sp.exp(-sp.I*2*sp.S.Pi*k*n/N), (n, 0, N-1))
            Xl = [Xs.xreplace({k: k0}) for k0 in range(0, N)]
            X = np.array(Xl)
        else:
            X = np.zeros(N, np.complex_)
            n = np.arange(0, N)
            x = self.eval(n)
            for k0 in np.arange(0, N):
                X[k0] = np.sum(x*np.exp(-1j*2*np.pi*k0*n/N))
        return X


class _DiscreteSignalOp(_DiscreteMixin, _SignalOp):
    is_Discrete = True


class DiscreteSignalAdd(_DiscreteSignalOp):

    def __init__(self, *args):
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
                if isinstance(arg, DiscreteSignalAdd):
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
            return DiscreteSignalAdd.reduce(args)

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
            return DiscreteSignalAdd(args)

    def eval(self, r):
        return sum(a.eval(r) for a in self.args)


class DiscreteSignalMul(_DiscreteSignalOp):

    def __init__(self, *args):
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
                if isinstance(arg, DiscreteSignalMul):
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
            return DiscreteSignalMul.reduce(args)

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
            return DiscreteSignalMul(args)

    def eval(self, r):
        val = 1
        for a in self.args:
            val *= a.eval(r)
        return val


class DiscreteSignalPow(_DiscreteSignalOp):

    def __init__(self, *args):
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
                if isinstance(arg, DiscreteSignalAdd):
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
            return DiscreteSignalPow.reduce(args)

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
                    new_sgn = s._pow(t)
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
            return DiscreteSignalAdd(args)

    def eval(self, r):
        # two posible cases
        y = self.args[0].eval(r)
        if len(self.args) == 1:
            return y
        e = self.args[1].eval(r)
        return y ** e


class DataSignal(_Signal, _DiscreteMixin):

    is_DataSignal = True

    def __init__(self, data, span):
        super.__init__()
        self._data = data
        self._span = span


class DiscreteFunctionSignal(_DiscreteMixin, _FunctionSignal):

    def __new__(cls, _xexpr, _yexpr, **_kwargs):
        return object.__new__(cls)

    def __init__(self, xexpr, yexpr, **kwargs):
        _FunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        _DiscreteMixin.__init__(self)
        self._z_transform = None
        self._dt_fourier_transform = None

    @property
    def z_transform(self):
        return self._z_transform

    @property
    def dtft(self):
        return self._dt_fourier_transform

    def generate(self, s0=0, step=1, size=1, overlap=0):
        if step != 1:
            raise ValueError('discrete signals are only defined' +
                             'for integer indexes')
        return super().generate(s0, step, size, overlap)

    def eval(self, x, force=False):
        is_scalar = False
        if isinstance(x, np.ndarray):
            if x.size == 1:
                is_scalar = True
        else:
            is_scalar = True
            x = np.array([x])
        if not force:
            self._check_indexes(x)
        y = _FunctionSignal.eval(self, x)
        if is_scalar:
            y = np.asscalar(y)
        return y


class Constant(DiscreteFunctionSignal):
    """
    Discrete constant signal. Not a degenerate case for constant functions
    such as `A*cos(0)`, `A*sin(pi/2)`, `A*exp(0*n)`, althought it could be.
    """
    is_finite = True

    def __new__(cls, const=0, **_kwargs):
        return _Signal.__new__(cls, const)

    def __init__(self, const=0, **kwargs):
        const = sp.sympify(const)
        if not const.is_constant():
            raise ValueError('const value is not constant')
        DiscreteFunctionSignal.__init__(self, _DiscreteMixin.default_xvar(),
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


class Delta(DiscreteFunctionSignal):
    """
    Discrete unit impulse signal.
    """

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, **_kwargs):
        return _Signal.__new__(cls, xexpr)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, **kwargs):
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        # yexpr = sp.Piecewise((1, sp.Eq(xexpr, 0)), (0, True))
        yexpr = UnitDelta(xexpr)
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self._period = sp.oo
        # TODO transformadas

    def __str__(self, *_args, **_kwargs):
        return 'delta[{0}]'.format(str(self.xexpr))

    def __repr__(self, *_args, **_kwargs):
        return 'Delta({0})'.format(str(self.xexpr))


class Step(DiscreteFunctionSignal):
    """
    Discrete unit step signal.
    """

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, **_kwargs):
        return _Signal.__new__(cls, xexpr)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, **kwargs):
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        # yexpr = sp.Piecewise((1, xexpr >= 0), (0, True))
        yexpr = UnitStep(xexpr)
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self._period = sp.oo
        # TODO transformadas

    def __str__(self):
        return 'u[{0}]'.format(str(self.xexpr))

    def __repr__(self):
        return 'Step({0})'.format(str(self.xexpr))


class Ramp(DiscreteFunctionSignal):
    """
    Discrete unit ramp signal.
    """

    is_finite = False
    is_integer = True
    is_nonnegative = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, **_kwargs):
        return _Signal.__new__(cls, xexpr)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, **kwargs):
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        # yexpr = sp.Piecewise((xexpr, xexpr >= 0), (0, True))
        yexpr = UnitRamp(xexpr)
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self._period = sp.oo

    def __str__(self):
        return 'r[{0}]'.format(str(self.xexpr))

    def __repr__(self):
        return 'Ramp({0})'.format(str(self.xexpr))


class RectPulse(DiscreteFunctionSignal):

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, width=16, **_kwargs):
        return _Signal.__new__(cls, xexpr, width)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, width=16, **kwargs):
        width = sp.sympify(width)
        if not width.is_integer or not width.is_nonnegative:
            raise ValueError('width must be a non-negative integer')
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        yexpr = sp.Piecewise((1, sp.Abs(xexpr) <= width), (0, True))
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self._period = sp.oo

    @property
    def width(self):
        return self.args[1]

    def __str__(self):
        return 'Pi{0}[{1}]'.format(self.width, self.xexpr)

    def __repr__(self):
        return 'RectPulse({0}, {1})'.format(self.xexpr, self.width)


class TriangPulse(DiscreteFunctionSignal):

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, width=16, **_kwargs):
        return _Signal.__new__(cls, xexpr, width)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, width=16, **kwargs):
        width = sp.sympify(width)
        if not width.is_integer or not width.is_nonnegative:
            raise ValueError('width must be a non-negative integer')
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        yexpr = sp.Max(0, 1 - sp.Abs(xexpr/width))
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self._period = sp.oo

    @property
    def width(self):
        return self.args[1]

    def __str__(self):
        return 'Delta{0}[{1}]'.format(self.width, str(self.xexpr))

    def __repr__(self):
        return 'TriangPulse({0}, {1})'.format(self.xexpr, self.width)


class _TrigMixin(object):

    def __init__(self, omega0, phi0):
        self._omega0 = sp.simplify(omega0)
        self._phi0 = self._reduce_phase(phi0)

    def _compute_period(self):
        # si omega0 es cero, periodo infinito
        if self.frequency == 0:
            return sp.oo
        # frecuencia positiva
        omega = sp.Mod(self.frequency, 2*sp.S.Pi)
        # trata de simplificar 2*pi/omega como racional
        sNk = sp.nsimplify(2*sp.S.Pi/omega, tolerance=1e-12, rational=True)
        if sp.ask(sp.Q.rational(sNk)):
            # si es racional, el periodo es el producto numerador x denominador
            r = sp.Rational(sNk)
            return r.p * r.q
        return sp.oo

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
    def phase(self):
        return self._phi0

    @property
    def period(self):
        if self._period is not None:
            return self._period
        self._period = self._compute_period()
        return self._period

    def as_euler(self):
        eu = DiscreteFunctionSignal(self._xexpr, self._yexpr.rewrite(sp.exp))
        eu.dtype = np.complex_
        return eu


class Sinusoid(_TrigMixin, DiscreteFunctionSignal):

    is_finite = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, A=1, omega0=1,
                phi0=0, **_kwargs):
        return _Signal.__new__(cls, xexpr, A, omega0, phi0)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, A=1, omega0=1,
                 phi0=0, **kwargs):
        A = sp.sympify(A)
        omega0 = sp.sympify(omega0)
        phi0 = sp.sympify(phi0)
        _TrigMixin.__init__(self, omega0, phi0)
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        yexpr = A*sp.cos(omega0*xexpr + phi0)
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)

    @property
    def amplitude(self):
        return self.args[1]

    @property
    def in_phase(self):
        A = self.amplitude * sp.cos(self.phase)
        if A == 0:
            return Constant(0)
        return Sinusoid(self.xexpr, A, self.frequency)

    @property
    def i(self):
        return self.in_phase

    @property
    def in_quadrature(self):
        A = -self.amplitude * sp.sin(self.phase)
        if A == 0:
            return Constant(0)
        return Sinusoid(self.xexpr, A, self.frequency, -sp.S.Pi/2)

    @property
    def q(self):
        return self.in_quadrature

    def __str__(self):
        s = '{0}*cos({1}*{2}'.format(str(self.amplitude), str(self.frequency),
                                     self.str_xexpr)
        phi = self.phase
        if phi != 0:
            s += ' {0} {1}'.format('-' if phi.is_negative else '+',
                                   str(abs(phi)))
        s += ')'
        return s

    def __repr__(self):
        return 'Sinusoid({0}, {1}, {2}, {3})'.format(str(self.xexpr),
                                                     str(self.amplitude),
                                                     str(self.frequency),
                                                     str(self.phase))


class Exponential(_TrigMixin, DiscreteFunctionSignal):

    is_real = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, A=1, alpha=1,
                **_kwargs):
        return _Signal.__new__(cls, xexpr, A, alpha)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, A=1, alpha=1,
                 **kwargs):
        A = sp.sympify(A)
        alpha = sp.sympify(alpha)
        _TrigMixin.__init__(self, _extract_omega(alpha), _extract_phase(A))
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        yexpr = A*sp.Pow(alpha, xexpr)
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        _, Ai = A.as_real_imag()
        _, ai = alpha.as_real_imag()
        if Ai.is_nonzero or ai.is_nonzero:
            self.dtype = np.complex_
            self.is_complex = True

    @property
    def amplitude(self):
        return self.args[1]

    @property
    def base(self):
        return self.args[2]

    @property
    def is_periodic(self):
        mod1 = sp.Abs(self.base) == sp.S.One
        return mod1 and super().is_periodic

    @property
    def phasor(self):
        # TODO verificar
        ph = sp.Abs(self.amplitude)*sp.exp(sp.I*self.phase)
        if ph.is_constant:
            return Constant(ph)
        return DiscreteFunctionSignal(ph)

    @property
    def carrier(self):
        # TODO verificar
        o = self.frequency
        c, _ = self._xexpr.as_coeff_mul(self.xvar)
        if c.is_negative:
            o = -o
        s = Exponential(self.xvar, 1, sp.exp(sp.I*o))
        if s.yexpr.is_constant():
            s = Constant(s.yexpr)
        return s

    def __str__(self):
        # s = '{0}'.format(str(self.yexpr))
        return str(self.yexpr)

    def __repr__(self):
        return 'Exponential({0}, {1}, {2})'.format(self.xexpr,
                                                   self.amplitude,
                                                   self.base)


class RealExponential(Exponential):

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, A=1, alpha=1,
                **_kwargs):
        return Exponential.__new__(cls, xexpr, A, alpha, **_kwargs)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, A=1, alpha=1,
                 **kwargs):
        Exponential.__init__(self, xexpr, A, alpha, **kwargs)
        if self.dtype_is_complex:
            raise ValueError('amplitude and base must be real')


class DeltaTrain(DiscreteFunctionSignal):
    """
    Discrete delta train signal.
    """
    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, N=16, **_kwargs):
        return _Signal.__new__(cls, xexpr, N)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, N=16, **kwargs):
        N = sp.sympify(N)
        if N <= 0:
            raise ValueError('N must be greater than 0')
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        nm = sp.Eq(sp.Mod(xexpr, N), 0)
        yexpr = sp.Piecewise((1, nm), (0, True))
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self.period = N

    def __str__(self, *_args, **_kwargs):
        return 'delta[(({0})){1}]'.format(self.xexpr, self.period)

    def __repr__(self):
        return 'DeltaTrain({0}, {1})'.format(self.xexpr, self.period)


class Square(DiscreteFunctionSignal):

    is_finite = True
    is_integer = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, N=16,
                width=8, **_kwargs):
        return _Signal.__new__(cls, xexpr, N, width)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, N=16,
                 width=8, **kwargs):
        N = sp.sympify(N)
        width = sympify(width)
        if isinstance(N, sp.Symbol):
            if not N.is_integer or not N.is_positive:
                raise ValueError('period must a positive integer')
        if isinstance(width, sp.Symbol):
            if not width.is_integer or not width.is_positive:
                raise ValueError('width must a positive integer')
        if N.is_number:
            if N <= 0:
                raise ValueError('N must be greater than 0')
            if width.is_number:
                if width >= N:
                    raise ValueError('width must be less than N')
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        nm = sp.Mod(xexpr, N)
        yexpr = sp.Piecewise((1, nm < width), (-1, nm < N))
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self.period = N

    @property
    def width(self):
        return self.args[2]

    def __str__(self, *_args, **_kwargs):
        return 'square[(({0})){1}/{2}]'.format(self.xexpr, self.period,
                                               self.width)

    def __repr__(self):
        return 'Square({0}, {1}, {2})'.format(self.xexpr, self.period,
                                              self.width)


class Sawtooth(DiscreteFunctionSignal):

    is_finite = True
    is_integer = False
    is_real = True

    def __new__(cls, xexpr=_DiscreteMixin._default_xvar, N=16,
                width=8, **_kwargs):
        return _Signal.__new__(cls, xexpr, N, width)

    def __init__(self, xexpr=_DiscreteMixin._default_xvar, N=16,
                 width=8, **kwargs):
        N = sp.sympify(N)
        width = sympify(width)
        if isinstance(N, sp.Symbol):
            if not N.is_integer or not N.is_positive:
                raise ValueError('period must a positive integer')
        if isinstance(width, sp.Symbol):
            if not width.is_integer or not width.is_positive:
                raise ValueError('width must a positive integer')
        if N.is_number:
            if N <= 0:
                raise ValueError('N must be greater than 0')
            if width.is_number:
                if width >= N:
                    raise ValueError('width must be less than N')
        xexpr = _DiscreteMixin._sympify_xexpr(xexpr)
        nm = sp.Mod(xexpr, N)
        yexpr = sp.Piecewise((-1+(2*nm)/width, nm < width),
                             (1-2*(nm-width)/(N-width), nm < N))
        DiscreteFunctionSignal.__init__(self, xexpr, yexpr, **kwargs)
        self.period = N

    @property
    def width(self):
        return self.args[2]

    def __str__(self, *_args, **_kwargs):
        return 'saw[(({0})){1}/{2}]'.format(self.xexpr, self.period,
                                            self.width)

    def __repr__(self):
        return 'Sawtooth({0}, {1}, {2})'.format(self.xexpr, self.period,
                                                self.width)
