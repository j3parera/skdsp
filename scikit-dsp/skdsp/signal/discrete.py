import re
import sys

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.utilities.iterables import flatten, is_sequence

from skdsp.signal.functions import (UnitDelta, UnitDeltaTrain, UnitRamp,
                                    UnitStep, deltasimp)
from skdsp.signal.signal import Signal
from skdsp.util.util import as_coeff_polar, ipystem

__all__ = [s for s in dir() if not s.startswith("_")]

n, m, k = sp.symbols("n, m, k", integer=True)


class DiscreteSignal(Signal):

    _default_iv = n

    @classmethod
    def from_sampling(cls, expr, civ, div, fs, **kwargs):
        expr = sp.S(expr)
        if expr.has(sp.DiracDelta):
            raise ValueError("Dirac delta cannot be sampled.")
        expr = expr.subs({civ: div / fs})
        if expr.has(sp.Heaviside):
            hvsds = list(expr.atoms(sp.Heaviside))
            for h in hvsds:
                n0 = list(sp.solveset(h.args[0], n))[0]
                if n0.is_integer:
                    expr = expr.replace(
                        sp.Heaviside(h.args[0], h.args[1]), UnitStep(n - n0)
                    )
        obj = cls._try_subclass(expr, iv=div, **kwargs)
        if obj is None:
            iv = div
            codomain = kwargs.pop("codomain", None)
            obj = cls(expr, iv, codomain, **kwargs)
        return obj

    @classmethod
    def from_formula(cls, expr, **kwargs):
        expr = sp.S(expr)
        obj = cls._try_subclass(expr, **kwargs)
        if obj is None:
            iv = kwargs.pop("iv", cls._default_iv)
            codomain = kwargs.pop("codomain", None)
            # strip known data if they exist
            kwargs.pop("domain", None)
            obj = cls(expr, iv, codomain, **kwargs)
        return obj

    @staticmethod
    def _deltify(data, start, iv, periodic):
        if periodic:
            M = len(data)
            s = start % M
            data = data[M - s : M] + data[0 : M - s]
        expr = sp.S.Zero
        for k, d in enumerate(data):
            if periodic:
                expr += d * UnitDelta(sp.Mod((iv - k), M))
            else:
                expr += d * UnitDelta(iv - start - k)
        return expr

    def __new__(cls, amp, iv, codomain, **kwargs):
        if iv is not None and (not iv.is_symbol or not iv.is_integer):
            raise ValueError("Invalid independent variable.")
        return Signal.__new__(
            cls, amp, iv, sp.S.Integers, codomain, **kwargs
        )

    def display(self, span=None):
        if span is None:
            span = range(-3, 16)
        s = "{ \u22ef "
        for k in span:
            if k != span.start:
                s += ", "
            vs = sp.sstr(self[k], full_prec=False)
            if k == 0:
                s += "_{0}_".format(vs)
            else:
                s += "{0}".format(vs)
        s += " \u22ef }"
        return s

    def ipystem(
        self,
        span,
        xlabel=None,
        title=None,
        axis=None,
        color="C0",
        marker="o",
        markersize=8,
        **kwargs
    ):
        n = np.array(span)
        v = [float(x) for x in self[span]]
        pre = kwargs.get("pretitle", None)
        if pre is not None:
            title = pre + " " + r"${0}$".format(self.latex())
        if xlabel is None:
            xlabel = r"${0}$".format(sp.latex(self.iv))
        return ipystem(n, v, xlabel, title, axis, color, marker, markersize)

    def sum(self, low=None, high=None, var=None, term=None, doit=True):
        var = sp.S(var) if var is not None else self.iv
        low = sp.S(low) if low is not None else sp.S.NegativeInfinity
        high = sp.S(high) if high is not None else sp.S.Infinity
        amp = self.amplitude
        if term is not None:
            term = sp.S(term)
            amp *= term ** var
        c, u = amp.as_coeff_mul(UnitStep)
        if len(u) == 0:
            if amp.has(UnitRamp):
                S = sp.Sum(amp.rewrite(sp.Piecewise), (var, low, high))
            else:
                S = sp.Sum(amp, (var, low, high))
        elif len(u) == 1:
            # unit step changes limits of summation if same variable
            st = u[0]
            if var in st.free_symbols:
                k0 = sp.Wild("k0")
                a0 = sp.Wild("a0")
                n0 = sp.Wild("n0")
                pattern = a0 * UnitStep(n0 * var - k0)
                d = st.match(pattern)
                if d is not None:
                    amp = c
                    if d[n0] == 1:
                        # u[n-k0]
                        low = sp.Max(low, d[k0])
                    elif d[n0] == -1:
                        # u[-n-k0]
                        high = sp.Min(high, d[k0])
                else:
                    k1 = sp.Wild("k1")
                    a1 = sp.Wild("a1")
                    n1 = sp.Wild("n1")
                    pattern = (a0 * UnitStep(n0 * var - k0)) - (
                        a1 * UnitStep(n1 * var - k1)
                    )
                    d = st.match(pattern)
                    if d is not None:
                        if d[a0] == d[a1]:
                            if d[n0] == d[n1]:
                                if d[n0] == sp.S.One:
                                    amp = c * d[a0]
                                    try:
                                        if d[k1] >= d[k0]:
                                            # u[n-k0] - u[n-k1]
                                            low = sp.Max(low, d[k0])
                                            high = sp.Min(high, d[k1] - 1)
                                        else:
                                            # -(u[n-k0] - u[n-k1]) = u[n-k1] - u[n-k0]
                                            low = sp.Max(low, d[k1])
                                            high = sp.Min(high, d[k0] - 1)
                                            amp *= -sp.S.One
                                    except TypeError:
                                        # cannot compare, assume k1 > k0 ?
                                        low = sp.Max(low, d[k0])
                                        high = sp.Min(high, d[k1] - 1)
                                elif d[n0] == -sp.S.One:
                                    amp = c * d[a0]
                                    if d[k0] <= d[k1]:
                                        # u[-n-k1] - u[-n-k0]
                                        low = sp.Max(low, -d[k1] + 1)
                                        high = sp.Min(high, -d[k0])
                                    else:
                                        # -(u[-n-k1] - u[-n-k1]) = u[-n-k1] - u[-n-k0]
                                        low = sp.Max(low, -d[k0] + 1)
                                        high = sp.Min(high, -d[k1])
                                        amp *= - sp.S.One
                try:
                    if high < low:
                        return sp.S.Zero
                except:
                    pass
            if amp.has(UnitRamp):
                S = sp.Sum(amp.rewrite(sp.Piecewise), (var, low, high))
            else:
                S = sp.Sum(amp, (var, low, high))
        elif len(u) == 2:
            # a*u[s0v-k0]*u[s1v-k1]
            a = sp.Wild("a")
            k0 = sp.Wild("k0")
            s0 = sp.Wild("s0")
            k1 = sp.Wild("k1")
            a1 = sp.Wild("a1")
            s1 = sp.Wild("s1")
            pattern = a * UnitStep(s0 * var - k0) * UnitStep(s1 * var - k1)
            st = u[0] * u[1]
            d = st.match(pattern)
            if d is not None:
                # s0 = s1 = 1
                if d[s0] == sp.S.One and d[s1] == sp.S.One:
                    # u[v-k0]*u[v-k1]
                    low0 = sp.Max(low, d[k0])
                    S0 = sp.Sum(d[a] * c, (var, low0, high))
                    cond0 = sp.simplify(d[k0] > d[k1])
                    low1 = sp.Max(low, d[k1])
                    S1 = sp.Sum(d[a] * c, (var, low1, high))
                    cond1 = sp.simplify(d[k1] >= d[k0])
                    S = sp.Piecewise((S0, cond0), (S1, cond1))
                elif d[s0] == sp.S.NegativeOne and d[s1] == sp.S.NegativeOne:
                    # u[-v-k0]*u[-v-k1]
                    # u[v-k0]*u[v-k1]
                    high0 = sp.Min(high, d[k1] - 1)
                    S0 = sp.Sum(d[a] * c, (var, low, high0))
                    cond0 = sp.simplify(d[k0] >= d[k1])
                    high1 = sp.Min(high, d[k0] - 1)
                    S1 = sp.Sum(d[a] * c, (var, low, high1))
                    cond1 = sp.simplify(d[k1] > d[k0])
                    S = sp.Piecewise((S0, cond0), (S1, cond1))
        if doit:
            S = S.doit(deep=True, sum=True)
        return S

    @sp.cacheit
    def energy(self, Nmax=sp.S.Infinity):
        # TODO permitir intervalos [n1, n2]
        ie = self.square_abs
        if isinstance(ie, DataSignal):
            if Nmax.is_number:
                sup = ie.support
                m = max(-Nmax, sup.inf)
                M = min(Nmax, sup.sup)
                E = sum(ie[m : M + 1])
            return E
        if Nmax == sp.S.Infinity:
            try:
                N = sp.Dummy("N", integer=True, positive=True)
                S = ie.sum(-N, N)
                S = self.apply_constraints(S, tosym=True)
                E = sp.limit(S, N, Nmax)
                E = self.undo_constraints(E)
                E = sp.simplify(E)
            except:
                E = None
        else:
            E = ie.sum(-Nmax, Nmax)
        return E

    @sp.cacheit
    def mean_power(self, Nmax=sp.S.Infinity):
        # TODO permitir intervalos [n1, n2]
        if Nmax == sp.S.Infinity:
            try:
                N = sp.Dummy("N", integer=True, positive=True)
                expr = (sp.S.One / (2 * N + 1)) * self.energy(N)
                P = sp.limit(expr, N, Nmax)
            except:
                P = None
        else:
            P = (sp.S.One / (2 * Nmax + 1)) * self.energy(Nmax)
        return P

    @property
    def is_causal(self):
        if self.is_periodic:
            return False
        negs = sp.Range(sp.S.NegativeInfinity, 0)
        return self.support.intersect(negs) == sp.EmptySet

    @property
    def is_abs_summable(self):
        s = sp.Sum(
            sp.Abs(self.amplitude), (self.iv, sp.S.NegativeInfinity, sp.S.Infinity)
        ).doit()
        return s.is_finite


class Undefined(DiscreteSignal):
    @classmethod
    def _match_expression(cls, expr, **kwargs):
        expr = sp.S(expr)
        if not expr.has(AppliedUndef):
            return None
        iv = kwargs.pop("iv", DiscreteSignal._default_iv)
        A = sp.Wild("A", properties=[lambda a: a.is_constant(iv)])
        f = sp.Wild("f", properties=[lambda a: isinstance(a, AppliedUndef)])
        m = expr.match(A * f)
        if m and f in m.keys() and isinstance(m[f], AppliedUndef):
            name = m[f].name
            delay, flip, _ = Signal._match_iv_transform(m[f].args[0], iv)
            codomain = kwargs.pop("codomain", sp.S.Reals)
            sg = Undefined(name, iv, codomain, **kwargs)
            sg = Signal._apply_gain_and_iv_transform(sg, m[A], delay, flip)
            return sg
        return None

    def __new__(
        cls, name, iv=None, codomain=sp.S.Reals, **kwargs
    ):
        iv = sp.sympify(iv) if iv is not None else n
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        undef = sp.Function(name, nargs=1)(iv)
        return DiscreteSignal.__new__(cls, undef, iv, codomain, **kwargs)

    @property
    def is_causal(self):
        return self.duration is not None


class Constant(DiscreteSignal):
    """
    Discrete constant signal.
    """

    is_finite = True

    @classmethod
    def _match_expression(cls, expr, **kwargs):
        expr = sp.S(expr)
        iv = kwargs.pop("iv", DiscreteSignal._default_iv)
        if expr.is_constant(iv, simplify=False):
            sg = Constant(expr, iv, **kwargs)
            return sg
        return None

    def __new__(cls, const, iv=None, **kwargs):
        const = sp.sympify(const)
        iv = sp.sympify(iv) if iv is not None else n
        if not const.is_constant(iv):
            raise ValueError("const value is not constant")
        codomain = sp.S.Reals if const in sp.S.Reals else sp.S.Complexes
        # strip known data if they exist
        kwargs.pop("period", None)
        kwargs.pop("domain", None)
        kwargs.pop("codomain", None)
        return Signal.__new__(
            cls, const, iv, sp.S.Integers, codomain, period=sp.S.One, **kwargs
        )

    @property
    def support(self):
        return sp.S.Integers

    @property
    def is_periodic(self):
        return True

    @property
    def is_causal(self):
        return False

    def __repr__(self):
        return "Constant({0})".format(str(self.amplitude))


class Delta(DiscreteSignal):

    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def _match_expression(cls, expr, **kwargs):
        expr = sp.S(expr)
        iv = kwargs.pop("iv", DiscreteSignal._default_iv)
        expr = deltasimp(expr, iv)
        A = sp.Wild("A", properties=[lambda a: a.is_constant(iv)])
        tiv = sp.Wild("tiv")
        m = expr.match(A * UnitDelta(tiv))
        if m and tiv in m.keys():
            delay, flip, _ = Signal._match_iv_transform(m[tiv], iv)
            sg = Delta(iv, **kwargs)
            sg = Signal._apply_gain_and_iv_transform(sg, m[A], delay, flip)
            return sg
        return None

    def __new__(cls, iv=None, **kwargs):
        iv = sp.sympify(iv) if iv is not None else n
        # strip known data if exist
        kwargs.pop("domain", None)
        kwargs.pop("codomain", None)
        kwargs.pop("period", None)
        obj = DiscreteSignal.__new__(
            cls, UnitDelta(iv), iv, sp.S.Reals, period=sp.S.Zero, **kwargs
        )
        return obj

    @property
    def support(self):
        inf = self._solve_func_arg(self.amplitude, 0)
        return sp.Range(inf, inf + 1)

    @property
    def is_periodic(self):
        return False

    def __repr__(self):
        return "Delta({0})".format(str(self))


class Step(DiscreteSignal):

    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def _match_expression(cls, expr, **kwargs):
        expr = sp.S(expr)
        iv = kwargs.pop("iv", DiscreteSignal._default_iv)
        A = sp.Wild("A", properties=[lambda a: a.is_constant(iv)])
        tiv = sp.Wild("tiv")
        m = expr.match(A * UnitStep(tiv))
        if m and tiv in m.keys():
            delay, flip, _ = Signal._match_iv_transform(m[tiv], iv)
            sg = Step(iv, **kwargs)
            sg = Signal._apply_gain_and_iv_transform(sg, m[A], delay, flip)
            return sg
        return None

    def __new__(cls, iv=None, **kwargs):
        iv = sp.sympify(iv) if iv is not None else n
        # strip known data if they exist
        kwargs.pop("period", None)
        kwargs.pop("domain", None)
        kwargs.pop("codomain", None)
        obj = DiscreteSignal.__new__(
            cls, UnitStep(iv), iv, sp.S.Reals, period=sp.S.Zero, **kwargs
        )
        return obj

    @property
    def support(self):
        inf = self._solve_func_arg(self.amplitude, 0)
        return sp.Range(inf, sp.S.Infinity)

    @property
    def is_periodic(self):
        return False

    @property
    def __repr__(self):
        return "Step({0})".format(str(self))


class Ramp(DiscreteSignal):

    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def _match_expression(cls, expr, **kwargs):
        expr = sp.S(expr)
        iv = kwargs.pop("iv", DiscreteSignal._default_iv)
        A = sp.Wild("A", properties=[lambda a: a.is_constant(iv)])
        tiv = sp.Wild("tiv")
        m = expr.match(A * UnitRamp(tiv))
        if m and tiv in m.keys():
            delay, flip, _ = Signal._match_iv_transform(m[tiv], iv)
            sg = Ramp(iv, **kwargs)
            sg = Signal._apply_gain_and_iv_transform(sg, m[A], delay, flip)
            return sg
        return None

    def __new__(cls, iv=None, **kwargs):
        iv = sp.sympify(iv) if iv is not None else n
        # strip known data if they exist
        kwargs.pop("period", None)
        kwargs.pop("domain", None)
        kwargs.pop("codomain", None)
        obj = DiscreteSignal.__new__(
            cls, UnitRamp(iv), iv, sp.S.Reals, period=sp.S.Zero, **kwargs
        )
        return obj

    @property
    def support(self):
        inf = self._solve_func_arg(self.amplitude, 0)
        return sp.Range(inf + 1, sp.S.Infinity)

    @property
    def is_periodic(self):
        return False

    def __repr__(self):
        return "Ramp({0})".format(str(self))


class DeltaTrain(DiscreteSignal):
    """
    Discrete delta train signal.
    """

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, iv=None, period=0, **kwargs):
        period = sp.sympify(period)
        if not period.is_integer or not period.is_positive:
            raise ValueError("Period must be an integer greater than 0.")
        iv = sp.sympify(iv) if iv is not None else n
        obj = DiscreteSignal.__new__(
            cls, UnitDeltaTrain(iv, period), iv, sp.S.Reals, period=period, **kwargs
        )
        return obj

    @property
    def support(self):
        return sp.S.Integers

    @property
    def is_periodic(self):
        return True

    @property
    def is_causal(self):
        return False

    def __repr__(self):
        return "DeltaTrain({0}, {1})".format(self.iv, self.period)


class DataSignal(DiscreteSignal):

    is_finite = True

    @classmethod
    def _match_expression(cls, expr, **kwargs):
        atoms = expr.atoms(sp.Function)
        if len(atoms) > 1 and all(isinstance(a, UnitDelta) for a in list(atoms)):
            iv = kwargs.pop("iv", DiscreteSignal._default_iv)
            start = kwargs.pop("start", None)
            periodic = kwargs.pop("periodic", None)
            codomain = kwargs.pop("codomain", sp.S.Reals)
            sg = DataSignal(expr, start, periodic, iv, codomain, isexpr=True, **kwargs)
            return sg
        return None

    def __new__(cls, data, start=0, periodic=None, iv=None, codomain=None, **kwargs):
        isexpr = kwargs.pop("isexpr", False)
        if not isexpr:
            data = sp.sympify(data)
            if isinstance(data, sp.Dict):
                # si es un diccionario asegurar claves enteras y expandir como lista
                keys = data.keys()
                for key in data.keys():
                    if not key.is_integer:
                        raise TypeError("All keys in dict must be integer")
                start = min(keys)
                end = max(keys)
                y = [0] * (end - start + 1)
                for k, v in data.items():
                    y[k - start] = v
                data = sp.sympify(y)
            if is_sequence(data):
                # las sequencias deben ser unidimensionales
                data = list(flatten(data))
            # iv
            if iv is None:
                free = set()
                for v in data:
                    free.update(v.free_symbols)
                if len(free) > 1:
                    raise ValueError(
                        "The independent variable must be supplied explicitly"
                    )
                elif len(free) == 0:
                    iv = n
                else:
                    iv = free.pop()
            else:
                if not isinstance(iv, sp.Symbol):
                    raise ValueError("The independent variable is not a valid symbol")
            # ajustar duracion
            if not periodic:
                while start < 0 and data[0] == 0:
                    start += 1
                    data = data[1:]
                while len(data) > 0 and data[-1] == 0:
                    data = data[:-1]
                if len(data) == 0:
                    # this is a constant 0
                    return Constant(0, iv)
            # codomain
            if codomain is None:
                codomain = (
                    sp.S.Complexes
                    if any([sp.sympify(x).as_real_imag()[1] for x in data])
                    else sp.S.Reals
                )
        # deltificar
        if isexpr:
            expr = data
            period = sp.S.Zero
        else:
            if len(data) == 1 and not periodic:
                return Delta(iv - start)
            else:
                expr = DiscreteSignal._deltify(data, start, iv, periodic)
                period = len(data) if periodic else sp.S.Zero
        # strip known data if they exist
        kwargs.pop("period", None)
        kwargs.pop("domain", None)
        kwargs.pop("codomain", None)
        obj = DiscreteSignal.__new__(cls, expr, iv, codomain, period=period, **kwargs)
        return obj

    @property
    def support(self):
        if self.is_periodic:
            return sp.S.Integers
        atoms = self.amplitude.atoms(UnitDelta)
        v = []
        for a in list(atoms):
            k = sp.solve_linear(a.func_arg, 0, [self.iv])[1]
            v.append(k)
        return sp.Range(min(v), max(v) + 1)

    @property
    def is_causal(self):
        return self.support.inf >= 0

class _TrigonometricDiscreteSignal(DiscreteSignal):
    @staticmethod
    def _period(omega):
        if omega.is_zero:
            # it's a constant
            return sp.S.One
        if omega.has(sp.S.Pi):
            _, e = omega.as_coeff_exponent(sp.S.Pi)
            if e != 1:
                # there is pi**(e != 1)
                return sp.S.Zero
            om, _ = omega.as_independent(sp.S.Pi)
            try:
                r = sp.Rational(str(om))
                return sp.S(2 * r.q)
            except:
                # not rational
                return sp.S.Zero
        return sp.S.Zero

    @staticmethod
    def _reduce_phase(phi, nonnegative=False):
        phi0 = sp.Mod(phi, 2 * sp.S.Pi)
        if not nonnegative and phi0 >= sp.S.Pi:
            phi0 -= 2 * sp.S.Pi
        return phi0

    @property
    def gain(self):
        return self.A

    @property
    def frequency(self):
        return self.omega

    @property
    def phase(self):
        return self.phi

    @property
    def support(self):
        return sp.S.Integers

    @property
    def is_causal(self):
        return False

    def _hashable_content(self):
        return (self.A, self.omega, self.phi) + super()._hashable_content(self)

    def reduced_phase(self, nonnegative=False):
        if self.phase.is_number:
            return _TrigonometricDiscreteSignal._reduce_phase(self.phase, nonnegative)
        else:
            return self.phase

    def reduced_frequency(self, nonnegative=False):
        if self.omega.is_number:
            return _TrigonometricDiscreteSignal._reduce_phase(self.omega, nonnegative)
        else:
            return self.omega

    def _clone_extra(self, obj):
        obj.A = self.A
        obj.omega = self.omega
        obj.phi = self.phi

    def alias(self, interval=0):
        romega = self.reduced_frequency() + 2 * sp.S.Pi * interval
        rphi = self.reduced_phase()
        # pylint: disable-msg=no-value-for-parameter
        return self.__class__(self.A, romega, rphi, self.iv)
        # pylint: enable-msg=no-value-for-parameter


class Sinusoid(_TrigonometricDiscreteSignal):

    is_finite = True

    @classmethod
    def _match_expression(cls, expr, **kwargs):
        expr = sp.S(expr)
        expr = expr.rewrite(sp.cos)
        iv = kwargs.pop("iv", DiscreteSignal._default_iv)
        A = sp.Wild("A", properties=[lambda a: a.is_constant(iv)])
        psi = sp.Wild("psi")
        m = expr.match(A * sp.cos(psi))
        if m and psi in m.keys():
            delay, omega, phi, N = Signal._match_trig_args(m[psi], iv)
            om, pio = omega.as_independent(sp.S.Pi)
            ph, pip = phi.as_independent(sp.S.Pi)
            sg = Sinusoid(
                m[A],
                sp.Rational(sp.sstr(om)) * pio * N,
                sp.Rational(sp.sstr(ph)) * pip,
                iv,
            )
            sg = Signal._apply_gain_and_iv_transform(
                sg, sp.S.One, delay / N, False
            )
            return sg
        return None

    def __new__(cls, A=1, omega=None, phi=0, iv=None, **kwargs):
        A = sp.sympify(A)
        if omega is None:
            raise ValueError("Frequency must be provided.")
        omega = sp.sympify(omega)
        phi = sp.sympify(phi)
        if not A.is_real or not omega.is_real or not phi.is_real:
            raise ValueError("All parameters must be real.")
        iv = sp.sympify(iv) if iv is not None else n
        expr = A * sp.cos(omega * iv + phi)
        period = _TrigonometricDiscreteSignal._period(omega)
        obj = DiscreteSignal.__new__(cls, expr, iv, sp.S.Reals, period=period, **kwargs)
        obj.A = A
        obj.omega = omega
        obj.phi = phi
        return obj

    def __getnewargs__(self):
        return (self.A, self.omega, self.phi, self.iv)

    def _eval_extra(self, vals, params):
        if self.gain in params.keys():
            g = sp.S(params[self.gain])
            if not g.is_real:
                raise ValueError("Amplitude must be real.")

    @property
    def in_phase(self):
        g = self.gain * sp.cos(self.phase)
        return g * sp.cos(self.frequency * self.iv)

    @property
    def I(self):
        return self.in_phase

    @property
    def in_quadrature(self):
        g = -self.gain * sp.sin(self.phase)
        return g * sp.sin(self.frequency * self.iv)

    @property
    def Q(self):
        return self.in_quadrature

    @property
    def euler(self):
        return self.amplitude.rewrite(sp.exp)

    def __repr__(self):
        return "Sinusoid({0}, {1}, {2}, {3})".format(
            str(self.gain), str(self.frequency), str(self.phase), str(self.iv)
        )


class Exponential(_TrigonometricDiscreteSignal):
    @classmethod
    def _match_expression(cls, expr, **kwargs):
        expr = sp.S(expr)
        iv = kwargs.pop("iv", DiscreteSignal._default_iv)
        C = sp.Wild("C", properties=[lambda a: a.is_constant(iv)])
        alpha = sp.Wild(
            "alpha", properties=[lambda a: a.is_constant(iv)], exclude=(sp.E,)
        )
        exp = sp.Wild(
            "exp", properties=[lambda a: not (a.is_Add or a.is_Pow) and a.has(iv)]
        )
        psi = sp.Wild("psi")
        # 1) C * alpha ** (nT)
        m = expr.match(C * alpha ** exp)
        if m and exp in m.keys():
            T = m[exp].as_coefficient(iv)
            sg = Exponential(m[C], m[alpha] ** T, iv)
            return sg
        # 2) C * sp.exp(sp.I * (omg * (nT - k)))
        m = expr.match(C * sp.exp(sp.I * psi))
        if m and psi in m.keys():
            delay, omega, phi, T = Signal._match_trig_args(m[psi], iv)
            om, pio = omega.as_independent(sp.S.Pi)
            ph, pip = phi.as_independent(sp.S.Pi)
            sg = Exponential(
                m[C] * sp.exp(sp.I * sp.Rational(sp.sstr(ph)) * pip),
                sp.exp(sp.I * sp.Rational(sp.sstr(om)) * pio * T),
                iv,
            )
            sg = Signal._apply_gain_and_iv_transform(
                sg, sp.S.One, delay / T, False
            )
            return sg
        # 3) C * alpha ** nT1 * sp.exp(sp.I * (omg * (nT2 - k) + phi))
        m = expr.match(C * alpha ** exp * sp.exp(sp.I * psi))
        if m and exp in m.keys() and psi in m.keys():
            T1 = m[exp].as_coefficient(iv)
            delay, omega, phi, T2 = Signal._match_trig_args(m[psi], iv)
            om, pio = omega.as_independent(sp.S.Pi)
            ph, pip = phi.as_independent(sp.S.Pi)
            sg = Exponential(
                m[C] * sp.exp(sp.I * sp.Rational(sp.sstr(ph)) * pip),
                m[alpha] ** T1 * sp.exp(sp.I * sp.Rational(sp.sstr(om)) * pio * T2),
                iv,
            )
            sg = Signal._apply_gain_and_iv_transform(
                sg, sp.S.One, delay / T2, False
            )
            return sg
        return None

    def __new__(cls, C=1, alpha=None, iv=None, **kwargs):
        alpha = sp.S(alpha)
        if alpha is None:
            raise ValueError("The exponential base must be supplied.")
        if alpha == sp.S.Zero:
            raise ValueError("The exponential base cannot be 0.")
        C = sp.S(C)
        iv = sp.sympify(iv) if iv is not None else n
        if not C.is_constant(iv) or not alpha.is_constant(iv):
            raise ValueError(
                "Gain and base must me constant with respect to independent variable."
            )
        r, omega = as_coeff_polar(alpha)
        if r == sp.S.One:
            if omega == sp.S.Zero:
                amp = C
                period = 1
            elif omega == sp.S.Pi:
                amp = sp.Mul(C, sp.Pow(-1, iv), evaluate=False)
                period = 2
            else:
                amp = sp.Mul(C, sp.exp(sp.I * omega * iv), evaluate=False)
                period = _TrigonometricDiscreteSignal._period(omega)
        else:
            if (omega == sp.S.Zero or omega == sp.S.Pi) or (
                alpha.is_real is not None and alpha.is_real
            ):
                amp = sp.Mul(C, sp.Pow(alpha, iv), evaluate=False)
            else:
                amp = sp.Mul(
                    C, sp.Pow(r, iv) * sp.exp(sp.I * omega * iv), evaluate=False
                )
            period = 0
        amp = sp.powsimp(amp)
        obj = DiscreteSignal.__new__(cls, amp, iv, None, period=period, **kwargs)
        c, phi = as_coeff_polar(C)
        A = c * sp.Pow(r, iv)
        obj.A = A
        obj.omega = omega
        obj.phi = phi
        obj.C = C
        obj.alpha = alpha
        return obj

    def __getnewargs__(self):
        return (self.C, self.alpha, self.iv)

    def _hashable_content(self):
        return (self.C, self.alpha) + super()._hashable_content(self)

    def _clone_extra(self, obj):
        super()._clone_extra(obj)
        obj.C = self.C
        obj.alpha = self.alpha

    def subs(self, *args, **kwargs):
        obj = super().subs(*args, **kwargs)
        obj.C = obj.C.subs(*args, **kwargs)
        obj.alpha = obj.alpha.subs(*args, **kwargs)
        return obj

    @property
    def base(self):
        return self.alpha

    @property
    def phasor(self):
        return sp.Abs(self.C) * sp.exp(sp.I * self.phase)

    @property
    def carrier(self):
        return sp.exp(sp.I * self.frequency * self.iv)

    @property
    def as_phasor_carrier(self):
        return (self.phasor, self.carrier)

    def __repr__(self):
        return "Exponential({0}, {1}, {2})".format(self.C, self.alpha, self.iv)
