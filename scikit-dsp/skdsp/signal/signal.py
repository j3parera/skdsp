import copy
from numbers import Number

import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain, periodicity
from sympy.core.decorators import call_highest_priority
from sympy.core.evaluate import global_evaluate
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.utilities.iterables import flatten, is_sequence, iterable

from skdsp.signal.functions import UnitDelta, deltasimp, stepsimp


class Signal(sp.Basic):

    _op_priority = 16
    is_conmutative = True

    def __new__(
        cls, amplitude, iv=None, domain=None, codomain=None, **kwargs
    ):
        # amplitude
        amplitude = sp.sympify(amplitude)
        # iv (independent variable)
        if iv is None:
            free = amplitude.free_symbols
            if len(free) == 0 or len(free) > 1:
                raise ValueError("The independent variable must be supplied explicitly")
            iv = list(free)[0]
        else:
            if not isinstance(iv, sp.Symbol):
                raise ValueError("The independent variable is not a valid symbol")
        # free symbols with the same name as iv must be the same
        for s in amplitude.free_symbols:
            if s.name == iv.name and s != iv:
                raise ValueError(
                    "The independent variable is not the same as that of the expression"
                )
        # domain
        if domain is None:
            domain = sp.S.Integers if iv.is_integer else sp.S.Reals
        else:
            domain = sp.sympify(domain)
            if domain not in [sp.S.Reals, sp.S.Integers]:
                raise ValueError("The domain is not valid")
        # codomain
        if codomain is None:
            _, yim = amplitude.as_real_imag()
            codomain = sp.S.Complexes if yim != 0 else sp.S.Reals
        else:
            codomain = sp.sympify(codomain)
            if codomain not in [sp.S.Reals, sp.S.Complexes]:
                raise ValueError("The codomain is not valid")
        # undefs
        if isinstance(amplitude, AppliedUndef):
            if hasattr(amplitude, "period") and hasattr(amplitude, "duration"):
                raise ValueError("Period and Duration are incompatible fetures.")
        # period
        period = kwargs.pop("period", None)
        if period is not None:
            period = sp.sympify(period)
            # known to be non periodic
            if period == sp.S.Zero:
                period = None
        else:
            period = Ellipsis
        # assumptions
        assumptions = kwargs.pop("assume", None)
        # create basic object (NO kwargs)
        if kwargs:
            raise ValueError("Unrecognized keyword argument(s): {0}".format(*kwargs))
        obj = sp.Basic.__new__(cls, amplitude, iv, domain, codomain)
        # other attributes
        obj._period = period
        obj._assumptions = assumptions
        return obj

    @staticmethod
    def _domain_periodicity(amp, period, domain):
        if domain == sp.S.Integers:
            # sympy returns 0 for constants
            if period.is_zero:
                if amp.is_constant():
                    period = sp.S.One
                else:
                    period = None
            elif period.is_rational:
                period = period.p
            else:
                period = None
        else:
            # sympy returns 0 for constants
            if period.is_zero:
                period = None
        return period

    @staticmethod
    def _periodicity(amp, iv, domain):
        amp = sp.expand_mul(amp)
        # 0.- Let's suppose that a summation is not periodic
        if amp.has(sp.Sum):
            return None
        # 1.- Undef explicitly set
        if isinstance(amp, AppliedUndef):
            if hasattr(amp, "period"):
                return amp.period
            return None
        # 2.- amplitude has its own method (trig functions...)
        # ... but it doesn't work with cos(x-pi) because it evaluates to -cos(x)
        # ... and then amp becomes Mul
        try:
            period = amp.period(iv)
            if period is not None:
                return Signal._domain_periodicity(amp, period, domain)
        except:
            pass
        # 3.- maybe sympy is smart enough. Usually it does correctly but sometimes fails,
        # ... for instance, exp(1j*3*pi*x/4). But it works well if exp(I*pi*x*Rational(3,4))
        period = periodicity(amp, iv)
        if period is not None:
            return Signal._domain_periodicity(amp, period, domain)
        # 4.- But fails for A*exp(I*pi*(x-3)*Rational(3,4)) because of pi!!
        om = sp.Wild("om")
        A = sp.Wild("A")
        phi = sp.Wild("phi")
        k = sp.Wild("k")
        patterns = [
            (A * sp.exp(om * (iv - k) + phi), sp.exp),
            (A * sp.sin(om * (iv - k) + phi), sp.sin),
            (A * sp.cos(om * (iv - k) + phi), sp.cos),
        ]
        for pattern in patterns:
            d = amp.match(pattern[0])
            if d is not None:
                a, b = d[om].as_independent(sp.S.Pi)
                if b != 0:
                    period = periodicity(pattern[1](a * iv), iv)
                    if period is not None:
                        period /= sp.S.Pi
                        return Signal._domain_periodicity(amp, period, domain)
        # 5.- sympy also fails with (-1)**n (n = integer)
        if domain == sp.S.Integers and sp.simplify(amp - (-1) ** iv) == 0:
            return sp.S(2)
        # TODO 6.- sympy also fails with delta[((n-k))M] and maybe other modulus
        return period

    def __getstate__(self):
        return {'_assumptions': self._assumptions}

    def _upclass(self):
        if self.__class__ == Signal:
            return self.__class__
        mro = self.__class__.__mro__
        idx = mro.index(Signal)
        return mro[idx - 1]

    def clone(self, cls, amplitude, **kwargs):
        if cls == None:
            cls = self._upclass()
        args = {
            "iv": self.iv,
            "domain": self.domain,
            "codomain": self.codomain,
            "period": self._period,
        }
        for key, value in kwargs.items():
            args[key] = value
        # pylint: disable-msg=redundant-keyword-arg
        obj = Signal.__new__(cls, amplitude, **args)
        # pylint: enable-msg=redundant-keyword-arg
        # pylint: disable-msg=no-member
        if hasattr(cls, "_transmute"):
            cls._transmute(obj)
        if hasattr(self, "_clone_extra"):
            self._clone_extra(obj)
        # pylint: enable-msg=no-member
        return obj

    def _hashable_content(self):
        return sp.Basic._hashable_content(self)

    def as_coeff_Mul(self, *args, **kwargs):
        return self.amplitude.as_coeff_Mul(*args, **kwargs)

    def latex(self, **settings):
        from skdsp.signal.printer import SignalLatexPrinter

        printer = SignalLatexPrinter(
            settings={
                "imaginary_unit": "rj",
                "fold_frac_powers": True,
                "order": "lex",
                "gothic_re_im": True,
            }
        )
        return printer.print_signal(self)

    @property
    def amplitude(self):
        return self.args[0]

    @property
    def iv(self):
        return self.args[1]

    def _solve_func_arg(self, func, value):
        f = list(self.amplitude.atoms(func))[0]
        iv = f.func_arg
        sols = sp.solve_linear(iv, value, [self.iv])
        return sols[1]

    @property
    def period(self):
        if self._period == Ellipsis:
            try:
                self._period = self._periodicity(self.amplitude, self.iv, self.domain)
            except:
                self._period = None
        return self._period

    @property
    def domain(self):
        return self.args[2]

    @property
    def codomain(self):
        return self.args[3]

    @property
    def is_periodic(self):
        return self.period != None

    @property
    def is_continuous(self):
        return self.domain == sp.S.Reals

    @property
    def is_discrete(self):
        return self.domain == sp.S.Integers

    @property
    def is_real(self):
        return self.codomain == sp.S.Reals

    @property
    def is_complex(self):
        return self.codomain == sp.S.Complexes

    @property
    def free_symbols(self):
        """
        Return lexically ordered list of free symbols excluding independent variable
        """
        free = self.amplitude.free_symbols
        return sorted(list(free.difference((self.iv,))), key=sp.default_sort_key)

    @property
    def support(self):
        if self.is_periodic:
            return self.domain
        expr = self.amplitude
        # 1.- Undef explicitly set
        if isinstance(expr, AppliedUndef):
            if hasattr(expr, "duration"):
                if self.is_discrete:
                    supp = sp.Range(0, expr.duration)
                else:
                    supp = sp.Interval(0, expr.duration)
            else:
                supp = self.domain
        # 2.- delta's
        elif expr.has(sp.KroneckerDelta):
            if expr.is_Add:
                supp = sp.S.EmptySet
                for arg in expr.args:
                    supp0 = sp.solveset(arg, self.iv, sp.S.Reals).complement(sp.S.Reals)
                    supp = sp.Union(supp, supp0)
            else:
                supp = sp.solveset(expr, self.iv, sp.S.Reals).complement(sp.S.Reals)
            # always discrete
            supp = sp.Range(supp.inf, supp.sup + 1)
        # 3.- otherwise
        else:
            supp = continuous_domain(expr, self.iv, sp.S.Reals)
            if supp.inf == sp.S.NegativeInfinity and supp.sup == sp.S.Infinity:
                supp = self.domain
            else:
                if self.is_discrete:
                    supp = sp.Range(supp.inf, supp.sup + 1)
        return supp

    @property
    def duration(self):
        supp = self.support
        duration = None
        if isinstance(supp, sp.Range):
            if supp.start >= 0:
                duration = supp.stop
        elif isinstance(supp, sp.Interval):
            if supp.inf == 0:
                duration = supp.sup
        return duration

    @property
    def as_real_imag(self):
        return self.amplitude.as_real_imag()

    @property
    def real_part(self):
        r, _ = self.as_real_imag
        if r is None:
            return sp.S.Zero
        return r

    @property
    def real(self):
        return self.clone(None, self.real_part, period=None, codomain=sp.S.Reals)

    @property
    def imag_part(self):
        _, i = self.as_real_imag
        if i is None:
            return sp.S.Zero
        return i

    @property
    def imag(self):
        return self.clone(None, self.imag_part, period=None, codomain=sp.S.Reals)

    @property
    def is_even(self):
        return (self.amplitude + self.amplitude.subs({self.iv: -self.iv})) == 0

    @property
    def is_odd(self):
        return (self.amplitude - self.amplitude.subs({self.iv: -self.iv})) == 0

    @property
    def even_part(self):
        if self.is_even:
            return self
        if self.is_odd:
            return sp.S.Zero
        return (self + self.flip().conjugate) * sp.S.Half

    @property
    def odd_part(self):
        if self.is_odd:
            return self
        if self.is_even:
            return sp.S.Zero
        return (self - self.flip().conjugate) * sp.S.Half

    @property
    def peak(self):
        # TODO tests
        return sp.Max(self.amplitude)

    @property
    def is_energy(self):
        E = self.energy()
        return E is not None and E.is_finite

    @property
    def is_power(self):
        P = self.mean_power()
        return P is not None and P.is_finite and not P.is_zero

    @property
    def is_causal(self):
        raise NotImplementedError
    
    @property
    def abs(self):
        return self.clone(None, sp.Abs(self.amplitude), period=None)

    @property
    def square_abs(self):
        return self.clone(None, sp.Abs(self.amplitude) ** 2, period=None)

    @property
    def conjugate(self):
        cls = self.__class__ if self.imag == 0 else None
        return self.clone(cls, sp.conjugate(self.amplitude))

    def magnitude(self, dB=False):
        m = sp.Abs(self.amplitude)
        if dB:
            m = sp.S(20) * sp.log(m, sp.S(10))
        return m

    def subs(self, *args, **kwargs):
        # TODO tests
        if self.iv in args or isinstance(args[0], dict) and self.iv in args[0].keys():
            raise ValueError("Use 'subs_iv' to change the independent variable.")
        amp = self.amplitude.subs(*args, **kwargs)
        return self.clone(self.__class__, amp)

    def subs_iv(self, newiv):
        if (self.is_discrete and not newiv.is_integer) or (
            self.is_continuous and newiv.is_integer
        ):
            raise ValueError("The new iv is not compatible.")
        amp = self.amplitude.subs({self.iv: newiv})
        return self.clone(self.__class__, amp, iv=newiv)

    def eval(self, xvals, params={}):
        if not isinstance(params, dict):
            raise ValueError("Parameter values must be in a dictionary")
        xvals = np.atleast_1d(xvals)
        if xvals.ndim > 1:
            raise ValueError("Arrays of independent variable values must be 1D")
        result = []
        for xval in xvals:
            if isinstance(xval, float) and xval.is_integer():
                xval = int(xval)
            if not xval in self.domain:
                raise ValueError("Independent variable value not in domain")
            if hasattr(self, "_eval_extra"):
                # pylint: disable-msg=no-member
                self._eval_extra(xvals, params)
                # pylint: enable-msg=no-member
            replacements = {self.iv: xval}
            replacements.update(params)
            result.append(self.amplitude.subs(replacements))
        return result[0] if len(result) == 1 else result

    def generate(self, start=0, step=1, size=1, overlap=0):
        """
        Signal value generator. Evaluates signal value chunks of size `size`,
        every `step` units, starting at `start`. Each chunk overlaps `overlap`
        values with the previous chunk.

        Args:
            start: Starting index of chunk; defaults to 0.
            step: Distance between indexes; defaults to 1.
            size: Number of values in chunk; defaults to 1.
            overlap: Number of values overlapped between chunks; defaults to 0.
                idx or a slice.

        Yields:
            List: A chunk of signal values.

        Examples:
            1. `generate(start=0, step=1, size=3, overlap=2)` returns
            [s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]], ...

            2. `generate(start=-1, step=1, size=2, overlap=0)` returns
            [s[-1], s[0]], [s[1], s[2]], [s[3], s[4]], ...

            3. `generate(start=0, step=0.1, size=3, overlap=0.1)` returns
            [s[0], s[0.1], s[0.2]], [s[0.1], s[0.2], s[0.3]],
            [s[0.2], s[0.3], s[0.4]], ...

        """
        if self.is_discrete and any(
            [not sp.S(p).is_integer for p in (start, step, size, overlap)]
        ):
            raise ValueError("Arguments(s) not valid for discrete signal")

        s = start
        while True:
            sl = np.linspace(s, s + (size * step), size, endpoint=False)
            yield self[sl]
            s += size * step - overlap

    def __eq__(self, other):
        if other is None or not isinstance(other, Signal):
            return False
        equal = (
            (other.domain == self.domain)
            and (other.codomain == self.codomain)
        )
        if not equal:
            return False
        if other.iv == self.iv:
            return other.amplitude - self.amplitude == sp.S.Zero
        else:
            return (other.iv.assumptions0 == self.iv.assumptions0) and (
                other.amplitude.xreplace({other.iv: self.iv}) - self.amplitude
                == sp.S.Zero
            )

    def __call__(self, xvals, *args):
        p = dict()
        if len(args) != 0:
            for s, v in zip(self.free_symbols, args):
                p[s] = v
        return self.eval(xvals, p)

    def __getitem__(self, key):
        """
        Evaluates signal at `key`; i.e
        :math:`y[key]`,
        :math:`y(key)`.

        Args:
            key: Range of index values of the signal; could be either a single
            idx or a slice.

        Returns:
            Signal values at `key`.

        """
        if isinstance(key, tuple):
            if isinstance(key[0], slice):
                return self.__call__(
                    np.arange(key[0].start, key[0].stop, key[0].step), *key[1:]
                )
            else:
                return self.__call__(*key)
        if isinstance(key, slice):
            return self.eval(np.arange(key.start, key.stop, key.step))
        return self.eval(key)

    def __rshift__(self, k):
        return self.shift(k)

    __irshift__ = __rshift__

    def __lshift__(self, k):
        return self.shift(-k)

    __ilshift__ = __lshift__

    def shift(self, k):
        """
        Delays (or advances) the signal by k.
        """
        if k not in self.domain:
            raise ValueError("Delay not allowed.")
        return self.clone(self.__class__, self.amplitude.subs({self.iv: self.iv - k}))

    delay = shift

    def flip(self):
        """Returns the signal reversed in time; ie s[-n] if discrete,
        otherwise s(-t).
        """
        return self.clone(self.__class__, self.amplitude.subs({self.iv: -self.iv}))

    def __abs__(self):
        return self.abs

    def _join_codomain(self, other):
        if self.codomain == sp.S.Complexes or other.codomain == sp.S.Complexes:
            return sp.S.Complexes
        # None para que se calcule
        return None

    def _join_period(self, other):
        if self.period is not None and other.period is not None:
            return sp.lcm(self.period, other.period)
        return None

    def _convert_other(self, other, identity):
        other = sp.S(other)
        if isinstance(other, Signal):
            if (
                self.is_discrete
                and other.is_continuous
                or self.is_continuous
                and other.is_discrete
            ):
                return NotImplemented, None, None
            if self.iv != other.iv:
                return NotImplemented, None, None
            if other.amplitude - identity == 0:
                return None, self._period, self.codomain
            return (other, self._join_period(other), self._join_codomain(other))
        if isinstance(other, (Number, sp.Expr)):
            if other - identity == 0:
                return None, None, None
            # pylint: disable-msg=too-many-function-args
            cls = self._upclass()
            other = Signal.__new__(
                cls, other, self.iv, self.domain, self.codomain, period=None
            )
            return (other, self._join_period(other), self._join_codomain(other))
            # pylint: enable-msg=too-many-function-args
        return NotImplemented, None, None

    def __add__(self, other):
        other, period, codomain = self._convert_other(other, sp.S.Zero)
        if other is None:
            return self
        if other is NotImplemented:
            return other
        amp = self.amplitude + other.amplitude
        obj = self.clone(None, amp, period=period, codomain=codomain)
        return obj

    @call_highest_priority("__add__")
    def __radd__(self, other):
        return self + other

    __iadd__ = __add__

    def __sub__(self, other):
        other, period, codomain = self._convert_other(other, sp.S.Zero)
        if other is None:
            return self
        if other is NotImplemented:
            return other
        amp = self.amplitude - other.amplitude
        obj = self.clone(None, amp, period=period, codomain=codomain)
        return obj

    @call_highest_priority("__sub__")
    def __rsub__(self, other):
        return (-self) + other

    __isub__ = __sub__

    def __mul__(self, other):
        other, period, codomain = self._convert_other(other, sp.S.One)
        if other is None:
            return self
        if other is NotImplemented:
            return other
        amp = self.amplitude * other.amplitude
        if amp.has(sp.Pow):
            amp = sp.powsimp(amp)
        amp = stepsimp(amp)
        cls = (
            self.__class__
            if other.amplitude.is_constant(self.iv) and not other.amplitude.is_zero
            else None
        )
        obj = self.clone(cls, amp, period=period, codomain=codomain)
        return obj

    @call_highest_priority("__mul__")
    def __rmul__(self, other):
        return self * other

    __imul__ = __mul__

    def __truediv__(self, other):
        other, period, codomain = self._convert_other(other, sp.S.One)
        if other is None:
            return self
        if other is NotImplemented:
            return other
        if other.amplitude.is_zero:
            raise ZeroDivisionError
        if not other.amplitude.is_constant():
            raise TypeError("Invalid operation.")
        amp = self.amplitude / other.amplitude
        cls = self.__class__
        obj = self.clone(cls, amp, period=period, codomain=codomain)
        return obj

    __itruediv__ = __truediv__

    def __neg__(self):
        amp = -self.amplitude
        cls = self.__class__
        obj = self.clone(cls, amp)
        return obj

    def __pow__(self, b):
        b = sp.S(b)
        if not b.is_integer or not b.is_nonnegative:
            raise ValueError("Undefinded operation.")
        amp = sp.S.One if b.is_zero else self.amplitude ** b
        obj = self.clone(None, amp, period=None, codomain=sp.S.Reals)
        return obj

    def convolve(self, other):
        other, period, codomain = self._convert_other(other, sp.S.One)
        if other is None:
            return self
        if other is NotImplemented:
            return other
        if self.is_discrete:
            k = sp.Symbol('k', integer=True)
            amp = self.amplitude.subs({self.iv: k}) * other.amplitude.subs({other.iv: self.iv - k})
            if period is not None:
                # TODO periodic convolution
                raise NotImplementedError
            amp = stepsimp(amp)
            if isinstance(amp, sp.Piecewise):
                cond = amp.args[0].cond
                expr = amp.args[0].expr
                sy = self.clone(None, expr, iv=k, period=None, codomain=codomain)
                amp = sp.Piecewise((sy.sum(), cond), *amp.args[1:])
            else:
                sy = self.clone(None, amp, iv=k, period=None, codomain=codomain)
                amp = sy.sum()
        else:
            # TODO continuous
            raise NotImplementedError
        amp = stepsimp(amp)
        amp = deltasimp(amp, self.iv)
        obj = self.clone(None, amp, period=None, codomain=codomain)
        return obj

    def __matmul__(self, other):
        return self.convolve(other)

    @call_highest_priority("__matmul__")
    def __rmatmul__(self, other):
        return self @ other

    __imatmul__ = __matmul__

    def correlate(self, other, normalized=False):
        # if not self.is_energy or not other.is_energy:
        #     raise ValueError("Cannot correlate infinite energy signals.")
        rso = self.convolve(other.flip())
        if normalized:
            rso = rso / sp.sqrt(self.energy() * other.energy())
        return rso

    def auto_correlate(self, normalized=False):
        return self.correlate(self, normalized)

    def cross_correlate(self, other, normalized=False):
        return self.correlate(other, normalized)
