import re
import sys

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.utilities.iterables import flatten, is_sequence
from skdsp.signal.functions import UnitDelta, UnitDeltaTrain, UnitRamp, UnitStep, deltasimp
from skdsp.signal.signal import Signal
from skdsp.util.util import as_coeff_polar, ipystem

__all__ = [s for s in dir() if not s.startswith("_")]

n, m, k = sp.symbols("n, m, k", integer=True)


class DiscreteSignal(Signal):
    @classmethod
    def _all_subclasses(cls):
        subclasses = set()
        for s in cls.__subclasses__():
            sub = s.__subclasses__()
            if len(sub) == 0:
                subclasses.add(s)
            else:
                subclasses.update(s._all_subclasses())
        return subclasses

    @classmethod
    def _try_subclass(cls, expr, **kwargs):
        subclasses = cls._all_subclasses()
        for s in subclasses:
            obj = s._match_expression(expr, **kwargs)
            if obj == NotImplementedError:
                # TODO
                obj = None
            if obj is not None:
                return obj
        return None

    @staticmethod
    def _match_iv_transform(expr, iv):
        k = sp.Wild("k", integer=True)
        s = sp.Wild("s", properties=[lambda s: abs(s) == 1])
        n = sp.Wild("n", integer=True) if iv is None else iv
        m = expr.match(s * n - k)
        if m:
            return m[n] if iv is None else iv, m[k], m[s] == -1
        return iv, 0, False

    @staticmethod
    def _match_trig_args(expr, iv):
        k = sp.Wild("k", integer=True)
        omega = sp.Wild("omega")
        phi = sp.Wild("phi")
        n = sp.Wild("n", integer=True) if iv is None else iv
        m = expr.match(omega * (n - k) + phi)
        if m:
            return m[n] if iv is None else iv, m[k], m[omega], m[phi]
        return iv, 0, 1, 0

    @staticmethod
    def _apply_iv_transform(sg, delay, flip):
        if delay:
            sg = sg.delay(delay)
        if flip:
            sg = sg.flip()
        return sg

    @classmethod
    def _transmute_renew(cls, obj):
        tcls, d, A, k, omg, phi, N, s = cls._transmute(obj, False)
        delay = 0
        done = False
        if tcls is not None:
            delay = d.get(k, 0)
            if tcls == Sinusoid:
                om, pio = d[omg].as_independent(sp.S.Pi)
                ph, pip = d[phi].as_independent(sp.S.Pi)
                if d.get(N) is not None:
                    # it's a sine
                    delay = d.get(N)
                    ph += sp.Rational(1, 2)
                obj = tcls(
                    d[A],
                    sp.Rational(sp.sstr(om)) * pio,
                    sp.Rational(sp.sstr(ph)) * pip,
                    obj.iv,
                )
                done = True
            elif tcls == Exponential:
                if d.get(N) is None:
                    om, pio = d[omg].as_independent(sp.S.Pi)
                    obj = tcls(
                        d[A],
                        sp.exp(sp.I * sp.Rational(sp.sstr(om))) * pio,
                        d[s] * obj.iv,
                    )
                elif d.get(omg) is None:
                    obj = tcls(d[A], d[N], obj.iv)
                else:
                    om, pio = d[omg].as_independent(sp.S.Pi)
                    ph, pip = d[phi].as_independent(sp.S.Pi)
                    obj = tcls(
                        d[A] * sp.exp(sp.I * sp.Rational(sp.sstr(ph)) * pip),
                        sp.exp(sp.I * sp.Rational(sp.sstr(om)) * pio),
                        obj.iv,
                    )
                done = True
            elif tcls == Constant:
                obj = tcls(d[A], obj.iv)
                done = True
            elif tcls == Delta:
                obj = d[A] * tcls(obj.iv)
                done = True
            # TODO otros casos
            if delay != 0:
                obj = obj.shift(delay)
        return done, obj

    @classmethod
    def _periodicity(cls, amp, iv, domain):
        if amp.has(UnitDeltaTrain):
            a, b = amp.as_independent(UnitDeltaTrain)
            return b.args[1]
        return super()._periodicity(amp, iv, domain)

    @classmethod
    def from_period(cls, expr, iv, period, **kwargs):
        expr = sp.S(expr)
        expr = expr.subs({iv: sp.Mod(iv, period)})
        obj = cls._try_subclass(expr, iv=iv, period=period, **kwargs)
        if obj is None:
            codomain = kwargs.pop("codomain", None)
            obj = cls(expr, iv=iv, period=period, codomain=codomain, **kwargs)
        return obj

    @classmethod
    def from_sampling(cls, expr, civ, div, fs, **kwargs):
        expr = sp.S(expr)
        if expr.has(sp.DiracDelta):
            raise ValueError("Dirac delta cannot be sampled.")
        expr = expr.subs({civ: div / fs})
        obj = cls._try_subclass(expr, iv=div, **kwargs)
        if obj is None:
            iv = div
            codomain = kwargs.pop("codomain", None)
            period = kwargs.pop("period", None)
            obj = cls(expr, iv=iv, period=period, codomain=codomain, **kwargs)
        return obj

    @classmethod
    def from_formula(cls, expr, **kwargs):
        expr = sp.S(expr)
        obj = cls._try_subclass(expr, **kwargs)
        if obj is None:
            iv = kwargs.pop("iv", n)
            codomain = kwargs.pop("codomain", None)
            period = kwargs.pop("period", None)
            obj = cls(expr, iv=iv, period=period, codomain=codomain, **kwargs)
        return obj

    @staticmethod
    def _transmute(obj, apply=True):
        # is DataSignal?
        atoms = obj.amplitude.atoms(sp.Function)
        if len(atoms) > 1:
            if all(isinstance(a, UnitDelta) for a in list(atoms)):
                if apply:
                    obj.__class__ = DataSignal
                    return None, None, None, None, None, None, None, None
                else:
                    return DataSignal, None, None, None, None, None, None, None
        A = sp.Wild("A")
        k = sp.Wild("k")
        s = sp.Wild("s")
        omg = sp.Wild("omega")
        phi = sp.Wild("phi")
        N = sp.Wild("N", exclude=(sp.Piecewise,))
        if obj.amplitude.has(UnitDelta, UnitDeltaTrain, sp.KroneckerDelta):
            patterns = [
                (A * UnitDelta(obj.iv - k), Delta),
                (A * UnitDeltaTrain((obj.iv - k), N), DeltaTrain),
            ]
        elif obj.amplitude.has(UnitStep):
            patterns = [(A * UnitStep(s * obj.iv - k), Step)]
        elif obj.amplitude.has(UnitRamp):
            patterns = [(A * UnitRamp(s * obj.iv - k), Ramp)]
        elif obj.amplitude.has(sp.cos, sp.sin):
            patterns = [
                (A * sp.cos(omg * (obj.iv - k) + phi), Sinusoid),
                (A * sp.sin(omg * (obj.iv - N) + phi), Sinusoid),
            ]
        elif obj.amplitude.has(sp.exp, sp.Pow):
            patterns = [
                (A * N ** (obj.iv - k), Exponential),
                (A * sp.exp(sp.I * (omg * (obj.iv - k))), Exponential),
                (
                    A * N ** (obj.iv - k) * sp.exp(sp.I * (omg * (obj.iv - k) + phi)),
                    Exponential,
                ),
            ]
        else:
            patterns = [(A, Constant)]
        for pattern in patterns:
            d = obj.amplitude.match(pattern[0])
            if d is not None:
                try:
                    if d[A].is_constant(obj.iv):
                        if apply:
                            # TODO cuando pattern[1] es Exponential o Sinusoid
                            # hay que crear los campos que no son args.
                            obj.__class__ = pattern[1]
                            break
                        else:
                            return (pattern[1], d, A, k, omg, phi, N, s)
                except:
                    pass
        return None, None, None, None, None, None, None, None

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

    def __new__(cls, amp, iv, period, codomain):
        if iv is not None and (not iv.is_symbol or not iv.is_integer):
            raise ValueError("Invalid independent variable.")
        # pylint: disable-msg=too-many-function-args
        return Signal.__new__(cls, amp, iv, sp.S.Integers, codomain, period)
        # pylint: enable-msg=too-many-function-args

    def __str__(self, *_args, **_kwargs):
        return sp.sstr(self.amplitude)

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
        c, u = amp.as_coeff_mul(UnitStep)
        if len(u) != 0:
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
                                        amp *= -sp.S.One
                try:
                    if high < low:
                        return sp.S.Zero
                except:
                    pass
        if term is not None:
            term = sp.S(term)
            amp *= term ** var
        # due to a sympy bug, but cannot be done in all cases
        if amp.has(UnitRamp):
            S = sp.Sum(amp.rewrite(sp.Piecewise), (var, low, high))
        else:
            S = sp.Sum(amp, (var, low, high))
        if doit:
            S = S.doit(deep=True)
        return S

    def energy(self, Nmax=sp.S.Infinity):
        # Los límites dan muchos problemas
        # NO fiarse
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
                E = sp.limit(S, N, Nmax)
            except:
                E = None
        else:
            E = ie.sum(-Nmax, Nmax)
        return E

    def mean_power(self, Nmax=sp.S.Infinity):
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
        if self[-10:0] != [sp.S.Zero] * 10:
            return False
        s = self.sum(high=-1)
        return s == sp.S.Zero

    @property
    def is_abs_summable(self):
        s = sp.Sum(
            sp.Abs(self.amplitude), (self.iv, sp.S.NegativeInfinity, sp.S.Infinity)
        ).doit()
        return s.is_finite


class Undefined(DiscreteSignal):
    @staticmethod
    def _match_expression(expr, **kwargs):
        expr = sp.S(expr)
        A = sp.Wild("A")
        f = sp.WildFunction("f", nargs=1)
        m = expr.match(A * f)
        if m and isinstance(m[f], AppliedUndef):
            name = m[f].name
            iv = kwargs.pop("iv", n)
            iv, delay, flip = DiscreteSignal._match_iv_transform(m[f].args[0], iv)
            period = kwargs.pop("period", 0)
            duration = kwargs.pop("duration", None)
            codomain = kwargs.pop("codomain", sp.S.Reals)
            sg = Undefined(name, iv, period, duration, codomain, **kwargs)
            if m[A] != sp.S.One:
                sg = m[A] * sg
            sg = DiscreteSignal._apply_iv_transform(sg, delay, flip)
            return sg
        return None

    def __new__(
        cls, name, iv=None, period=None, duration=None, codomain=sp.S.Reals, **_kwargs
    ):
        iv = sp.sympify(iv) if iv is not None else n
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        undef = sp.Function(name, nargs=1)(iv)
        if period is not None:
            undef.period = period
        elif hasattr(undef, "period"):
            del undef.period
        if duration is not None:
            undef.duration = duration
        elif hasattr(undef, "duration"):
            del undef.duration
        # pylint: disable-msg=too-many-function-args
        return Signal.__new__(cls, undef, iv, sp.S.Integers, codomain, None, **_kwargs)
        # pylint: enable-msg=too-many-function-args

    @property
    def is_causal(self):
        return self.duration is not None


class Constant(DiscreteSignal):
    """
    Discrete constant signal.
    """

    is_finite = True

    @staticmethod
    def _match_expression(expr, **kwargs):
        expr = sp.S(expr)
        iv = kwargs.pop("iv", n)
        if expr.is_constant(iv, simplify=False):
            sg = Constant(expr, iv, **kwargs)
            return sg
        return None

    def __new__(cls, const, iv=None):
        const = sp.sympify(const)
        iv = sp.sympify(iv) if iv is not None else n
        if not const.is_constant(iv):
            raise ValueError("const value is not constant")
        codomain = sp.S.Reals if const in sp.S.Reals else sp.S.Complexes
        # pylint: disable-msg=too-many-function-args
        return Signal.__new__(cls, const, iv, sp.S.Integers, codomain, sp.S.One)
        # pylint: enable-msg=too-many-function-args

    @property
    def support(self):
        return sp.S.Integers

    @property
    def is_periodic(self):
        return True

    def __repr__(self):
        return "Constant({0})".format(str(self.amplitude))


class Delta(DiscreteSignal):

    is_finite = True
    is_integer = True
    is_nonnegative = True

    @staticmethod
    def _match_expression(expr, **kwargs):
        expr = sp.S(expr)
        A = sp.Wild("A")
        f = sp.WildFunction("f", nargs=(1, 2))
        m = expr.match(A * f)
        if m and m[f].func == UnitDelta:
            iv = kwargs.pop("iv", n)
            expr = deltasimp(expr, iv)
            arg = m[f].args[0]
            iv, delay, flip = DiscreteSignal._match_iv_transform(arg, iv)
            period = kwargs.pop("period", 0)
            duration = kwargs.pop("duration", None)
            codomain = kwargs.pop("codomain", sp.S.Reals)
            sg = Delta(iv, **kwargs)
            if m[A] != sp.S.One:
                sg = m[A] * sg
            sg = DiscreteSignal._apply_iv_transform(sg, delay, flip)
            return sg
        return None

    def __new__(cls, iv=None, **kwargs):
        iv = sp.sympify(iv) if iv is not None else n
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(cls, UnitDelta(iv), iv, sp.S.Zero, sp.S.Reals, **kwargs)
        # pylint: enable-msg=too-many-function-args
        # obj.amplitude.__class__ = UnitDelta
        return obj

    def _clone_extra(self, obj):
        # if isinstance(obj.amplitude, sp.KroneckerDelta):
        #     obj.amplitude.__class__ = UnitDelta
        # for arg in obj.amplitude.args:
        #     if isinstance(arg, sp.KroneckerDelta):
        #         arg.__class__ = UnitDelta
        return obj

    @property
    def is_periodic(self):
        return False

    @property
    def support(self):
        inf = self._solve_func_arg(UnitDelta, 0)
        return sp.Range(inf, inf + 1)

    def __repr__(self):
        return "Delta({0})".format(str(self))


class Step(DiscreteSignal):
    @staticmethod
    def _match_expression(expr, **_kwargs):
        # TODO
        return NotImplementedError

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, iv=None):
        iv = sp.sympify(iv) if iv is not None else n
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(cls, UnitStep(iv), iv, sp.S.Zero, sp.S.Reals)
        # pylint: enable-msg=too-many-function-args
        return obj

    @property
    def is_periodic(self):
        return False

    @property
    def support(self):
        inf = self._solve_func_arg(UnitStep, 0)
        return sp.Range(inf, sp.S.Infinity)

    def __repr__(self):
        return "Step({0})".format(str(self))


class Ramp(DiscreteSignal):
    @staticmethod
    def _match_expression(expr, **_kwargs):
        # TODO
        return NotImplementedError

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, iv=None):
        iv = sp.sympify(iv) if iv is not None else n
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(cls, UnitRamp(iv), iv, sp.S.Zero, sp.S.Reals)
        # pylint: enable-msg=too-many-function-args
        return obj

    @property
    def is_periodic(self):
        return False

    @property
    def support(self):
        inf = self._solve_func_arg(UnitRamp, 0)
        return sp.Range(inf + 1, sp.S.Infinity)

    def __repr__(self):
        return "Ramp({0})".format(str(self))


class DeltaTrain(DiscreteSignal):
    """
    Discrete delta train signal.
    """

    @staticmethod
    def _match_expression(expr, **_kwargs):
        # TODO
        return NotImplementedError

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, iv=None, N=0):
        N = sp.sympify(N)
        if not N.is_integer or not N.is_positive:
            raise ValueError("N must be an integer greater than 0.")
        iv = sp.sympify(iv) if iv is not None else n
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(cls, UnitDeltaTrain(iv, N), iv, N, sp.S.Reals)
        # pylint: enable-msg=too-many-function-args
        return obj

    @property
    def is_periodic(self):
        return True

    @property
    def support(self):
        return sp.S.Integers

    def __repr__(self):
        return "DeltaTrain({0}, {1})".format(self.iv, self.period)


class DataSignal(DiscreteSignal):

    is_finite = True

    @staticmethod
    def _match_expression(expr, **kwargs):
        atoms = expr.atoms(sp.Function)
        if len(atoms) > 1 and all(isinstance(a, UnitDelta) for a in list(atoms)):
            iv = kwargs.pop("iv", n)
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
                    raise ValueError("The independent variable must be supplied explicitly")
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
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(cls, expr, iv, period, codomain, **kwargs)
        # pylint: enable-msg=too-many-function-args
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

    @staticmethod
    def _match_expression(expr, **kwargs):
        expr = sp.S(expr)
        A = sp.Wild("A")
        f = sp.WildFunction("f", nargs=1)
        m = expr.match(A * f)
        if m and m[f].func in [sp.cos, sp.sin]:
            iv = kwargs.pop("iv", n)
            iv, delay, omega, phi = DiscreteSignal._match_trig_args(m[f].args[0], iv)
            om, pio = omega.as_independent(sp.S.Pi)
            ph, pip = phi.as_independent(sp.S.Pi)
            if m[f].func == sp.sin:
                ph += sp.S.Half
            sg = Sinusoid(
                m[A], sp.Rational(sp.sstr(om)) * pio, sp.Rational(sp.sstr(ph)) * pip, iv
            )
            sg = DiscreteSignal._apply_iv_transform(sg, delay, False)
            return sg
        return None

    def __new__(cls, A=1, omega=None, phi=0, iv=None):
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
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(cls, expr, iv, period, sp.S.Reals)
        # pylint: enable-msg=too-many-function-args
        obj.A = A
        obj.omega = omega
        obj.phi = phi
        return obj

    def _eval_extra(self, vals, params):
        if self.gain in params.keys():
            g = sp.S(params[self.gain])
            if not g.is_real:
                raise ValueError("Aplitude must be real.")

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
    @staticmethod
    def _match_expression(expr, **_kwargs):
        # TODO
        return NotImplementedError

    # TODO Trigonometric? Quizás derivar ComplexExponential que sí

    def __new__(cls, C=1, alpha=None, iv=None):
        if alpha is None:
            raise ValueError("The exponential base must be supplied")
        C = sp.S(C)
        alpha = sp.S(alpha)
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
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(cls, amp, iv, period, None)
        # pylint: enable-msg=too-many-function-args
        c, phi = as_coeff_polar(C)
        A = c * sp.Pow(r, iv)
        obj.A = A
        obj.omega = omega
        obj.phi = phi
        obj.C = C
        obj.alpha = alpha
        return obj

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
