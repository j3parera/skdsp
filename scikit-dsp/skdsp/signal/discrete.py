import re

import numpy as np
import sympy as sp

from skdsp.signal.functions import UnitDelta, UnitDeltaTrain, UnitRamp, UnitStep
from skdsp.signal.signal import Signal
from skdsp.signal.util import as_coeff_polar, ipystem

__all__ = [s for s in dir() if not s.startswith("_")]

n, m, k = sp.symbols("n, m, k", integer=True)


def filter(B, A, x, ci=None):
    # muy preliminar
    # transposed DFII
    mm = max(len(B), len(A))
    B = B + [0] * (mm - len(B))
    B = sp.S(B)
    A = A + [0] * (mm - len(A))
    A = sp.S([a / A[0] for a in A])
    M = sp.S(ci) if ci is not None else sp.S([0] * (mm - 1))
    Y = sp.S([0] * len(x))
    x = sp.S(x)
    for k, v in enumerate(x):
        y = B[0] * v + M[0]
        Y[k] = y
        for m in range(0, len(M) - 1):
            M[m] = B[m + 1] * v - A[m + 1] * y + M[m + 1]
        M[-1] = B[-1] * v - A[-1] * y
    return Y


class DiscreteSignal(Signal):
    @classmethod
    def _transmute_renew(cls, obj):
        tcls, d, A, k, omg, phi, N = cls._transmute(obj, False)
        delay = 0
        done = False
        if tcls is not None:
            delay = d.get(k, 0)
            if tcls == Sinusoid.__mro__[0]:
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
            elif tcls == Exponential.__mro__[0]:
                if d.get(N) is None:
                    om, pio = d[omg].as_independent(sp.S.Pi)
                    obj = tcls(
                        d[A], sp.exp(sp.I * sp.Rational(sp.sstr(om))) * pio, obj.iv
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
            # TODO otros casos
            if delay != 0:
                obj = obj.shift(delay)
        return done, obj

    @classmethod
    def from_period(cls, amp, iv, period, codomain=sp.S.Reals):
        amp = sp.S(amp)
        amp = amp.subs({iv: sp.Mod(iv, period)})
        obj = cls(amp, iv, period, sp.S.Integers, codomain)
        done, obj = cls._transmute_renew(obj)
        if not done:
            Signal.__init__(obj)
        return obj

    @classmethod
    def from_sampling(cls, camp, civ, div, fs, codomain=sp.S.Reals):
        camp = sp.S(camp)
        amp = camp.subs({civ: div / fs})
        obj = cls(amp, div, None, sp.S.Integers, codomain)
        done, obj = cls._transmute_renew(obj)
        if not done:
            Signal.__init__(obj)
        return obj

    @classmethod
    def from_formula(cls, amp, iv=None, codomain=sp.S.Reals):
        amp = sp.S(amp)
        obj = cls(amp, iv, None, sp.S.Integers, codomain)
        done, obj = cls._transmute_renew(obj)
        if not done:
            Signal.__init__(obj)
        return obj

    @staticmethod
    def _transmute(obj, apply=True):
        try:
            if obj.amplitude.is_constant(obj.iv) and apply:
                obj.__class__ = Constant.__mro__[0]
                return
        except:
            pass
        A = sp.Wild("A")
        k = sp.Wild("k")
        omg = sp.Wild("omega")
        phi = sp.Wild("phi")
        N = sp.Wild("N")
        if obj.amplitude.has(UnitDelta, UnitDeltaTrain, sp.KroneckerDelta):
            patterns = [
                (A * UnitDelta(obj.iv - k), Delta.__mro__[0]),
                (A * UnitDeltaTrain((obj.iv - k), N), Ramp.__mro__[0]),
            ]
        elif obj.amplitude.has(UnitStep):
            patterns = [(A * UnitStep(obj.iv - k), Step.__mro__[0])]
        elif obj.amplitude.has(UnitRamp):
            patterns = [(A * UnitRamp(obj.iv - k), Ramp.__mro__[0])]
        elif obj.amplitude.has(sp.cos, sp.sin):
            patterns = [
                (A * sp.cos(omg * (obj.iv - k) + phi), Sinusoid.__mro__[0]),
                (A * sp.sin(omg * (obj.iv - N) + phi), Sinusoid.__mro__[0]),
            ]
        elif obj.amplitude.has(sp.exp, sp.Pow):
            patterns = [
                (A * N ** (obj.iv - k), Exponential.__mro__[0]),
                (A * sp.exp(sp.I * (omg * (obj.iv - k))), Exponential.__mro__[0]),
                (
                    A * N ** (obj.iv - k) * sp.exp(sp.I * (omg * (obj.iv - k) + phi)),
                    Exponential.__mro__[0],
                ),
            ]
        for pattern in patterns:
            d = obj.amplitude.match(pattern[0])
            if d is not None:
                try:
                    # if d[A] in obj.free_symbols or d[A].is_constant():
                    if d[A].is_constant():
                        if apply:
                            obj.__class__ = pattern[1]
                            break
                        else:
                            return (pattern[1], d, A, k, omg, phi, N)
                except:
                    pass
        return None, None, None, None, None, None, None

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

    def __new__(cls, amp, iv, period, domain, codomain):
        if iv is not None and (not iv.is_symbol or not iv.is_integer):
            raise ValueError("Invalid independent variable.")
        # pylint: disable-msg=too-many-function-args
        return Signal.__new__(cls, amp, iv, period, domain, codomain)
        # pylint: enable-msg=too-many-function-args

    def __str__(self, *_args, **_kwargs):
        return sp.sstr(self.amplitude)

    def _latex(self, printer=None):
        return printer.doprint(self.amplitude, imaginary_unit="rj")

    def latex(self):
        ltx = sp.latex(self.amplitude, imaginary_unit="rj")
        fr = re.compile(r"(.+)(\\frac{.+)(.+)(})({.+})(.+)")
        s = fr.split(ltx)
        ivs = "{0}".format(self.iv.name)
        if ivs in s:
            sr = ""
            k0 = -100
            for k, x in enumerate(s):
                if k == k0:
                    continue
                elif k == k0 + 3:
                    sr += ivs
                if x.startswith(r"\frac{"):
                    k0 = k + 1
                sr += x
            return sr
        return ltx

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
        color="k",
        marker="o",
        markersize=8,
        **kwargs
    ):
        n = np.array(span)
        v = [float(x.evalf(2)) for x in self[span]]
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
                                    if d[k1] >= d[k0]:
                                        # u[n-k0] - u[n-k1]
                                        low = sp.Max(low, d[k0])
                                        high = sp.Min(high, d[k1] - 1)
                                    else:
                                        # -(u[n-k0] - u[n-k1]) = u[n-k1] - u[n-k0]
                                        low = sp.Max(low, d[k1])
                                        high = sp.Min(high, d[k0] - 1)
                                        amp *= -sp.S.One
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
                if high < low:
                    return sp.S.Zero
        if term is not None:
            term = sp.S(term)
            amp *= term ** var
        S = sp.Sum(amp, (var, low, high))
        if doit:
            S = S.doit(deep=True)
        return S

    def energy(self, Nmax=sp.S.Infinity):
        # TODO tests
        ie = self.square_abs
        if Nmax == sp.S.Infinity:
            try:
                N = sp.Dummy("N", integer=True, positive=True)
                E = sp.limit(ie.sum(-N, N), N, Nmax)
            except:
                E = None
        else:
            E = ie.sum(-Nmax, Nmax)
        return E

    def mean_power(self, Nmax=sp.S.Infinity):
        # TODO tests
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


class Constant(DiscreteSignal):
    """
    Discrete constant signal. Not a degenerate case for constant functions
    such as `A*cos(0)`, `A*sin(pi/2)`, `A*exp(0*n)`, althought it could be.
    """

    is_finite = True

    def __new__(cls, const, iv=None):
        const = sp.sympify(const)
        iv = sp.sympify(iv) if iv is not None else n
        if not const.is_constant():
            raise ValueError("const value is not constant")
        codomain = sp.S.Reals if const in sp.S.Reals else sp.S.Complexes
        # pylint: disable-msg=too-many-function-args
        return Signal.__new__(cls, const, iv, sp.S.One, sp.S.Integers, codomain)
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

    def __new__(cls, iv=None):
        iv = sp.sympify(iv) if iv is not None else n
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(
            cls, UnitDelta(iv), iv, sp.S.Zero, sp.S.Integers, sp.S.Reals
        )
        # pylint: enable-msg=too-many-function-args
        obj.amplitude.__class__ = UnitDelta
        return obj

    def _clone_extra(self, obj):
        for arg in obj.amplitude.args:
            if isinstance(arg, sp.KroneckerDelta):
                arg.__class__ = UnitDelta
        return obj

    @property
    def is_periodic(self):
        return False

    @property
    def support(self):
        return sp.Range(0, 0)

    def __repr__(self):
        return "Delta({0})".format(str(self))


class Step(DiscreteSignal):

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, iv=None):
        iv = sp.sympify(iv) if iv is not None else n
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(
            cls, UnitStep(iv), iv, sp.S.Zero, sp.S.Integers, sp.S.Reals
        )
        # pylint: enable-msg=too-many-function-args
        return obj

    @property
    def is_periodic(self):
        return False

    @property
    def support(self):
        return sp.Range(0, sp.S.Infinity)

    def __repr__(self):
        return "Step({0})".format(str(self))


class Ramp(DiscreteSignal):

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, iv=None):
        iv = sp.sympify(iv) if iv is not None else n
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(
            cls, UnitRamp(iv), iv, sp.S.Zero, sp.S.Integers, sp.S.Reals
        )
        # pylint: enable-msg=too-many-function-args
        return obj

    @property
    def is_periodic(self):
        return False

    @property
    def support(self):
        return sp.Range(1, sp.S.Infinity)

    def __repr__(self):
        return "Ramp({0})".format(str(self))


class DeltaTrain(DiscreteSignal):
    """
    Discrete delta train signal.
    """

    is_finite = True
    is_integer = True
    is_nonnegative = True

    def __new__(cls, iv=None, N=0):
        N = sp.sympify(N)
        if not N.is_integer or not N.is_positive:
            raise ValueError("N must be an integer greater than 0.")
        iv = sp.sympify(iv) if iv is not None else n
        # pylint: disable-msg=too-many-function-args
        obj = DiscreteSignal.__new__(
            cls, UnitDeltaTrain(iv, N), iv, N, sp.S.Integers, sp.S.Reals
        )
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


class Data(DiscreteSignal):

    is_finite = True

    def __new__(cls, data, start=0, periodic=None, iv=None, codomain=None):
        from sympy.utilities.iterables import flatten, is_sequence

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
            while data[-1] == 0:
                data = data[:-1]
        # codomain
        if codomain is None:
            codomain = (
                sp.S.Complexes
                if any([sp.sympify(x).as_real_imag()[1] for x in data])
                else sp.S.Reals
            )
        # deltificar
        if len(data) == 1 and not periodic:
            obj = Delta(iv - start)
        else:
            expr = DiscreteSignal._deltify(data, start, iv, periodic)
            period = len(data) if periodic else sp.S.Zero
            # pylint: disable-msg=too-many-function-args
            obj = DiscreteSignal.__new__(cls, expr, iv, period, sp.S.Integers, codomain)
            # pylint: enable-msg=too-many-function-args
        return obj


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
        return self.__class__.__mro__[0](self.A, romega, rphi, self.iv)
        # pylint: enable-msg=no-value-for-parameter


class Sinusoid(_TrigonometricDiscreteSignal):

    is_finite = True

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
        obj = DiscreteSignal.__new__(cls, expr, iv, period, sp.S.Integers, sp.S.Reals)
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

    is_real = True

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
        obj = DiscreteSignal.__new__(cls, amp, iv, period, sp.S.Integers, None)
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

    def __repr__(self):
        return "Exponential({0}, {1}, {2})".format(self.C, self.alpha, self.iv)
