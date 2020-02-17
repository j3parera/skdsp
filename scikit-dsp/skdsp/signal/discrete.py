import numpy as np
import sympy as sp

from skdsp.signal.functions import UnitDelta, UnitRamp, UnitStep, UnitDeltaTrain
from skdsp.signal.signal import Signal
from skdsp.signal.util import stem, ipystem

__all__ = [s for s in dir() if not s.startswith("_")]

n, m, k = sp.symbols("n, m, k", integer=True)


class DiscreteSignal(Signal):

    @classmethod
    def from_period(cls, amp, iv, period, codomain=sp.S.Reals):
        amp = sp.S(amp)
        amp = amp.subs({iv: sp.Mod(iv, period)})
        return cls(amp, iv, period, sp.S.Integers, codomain)

    @classmethod
    def from_sampling(cls, camp, civ, div, fs, codomain=sp.S.Reals):
        camp = sp.S(camp)
        amp = camp.subs({civ: div/fs})
        return cls(amp, div, None, sp.S.Integers, codomain)

    @staticmethod
    def deltify(data, start, iv, periodic):
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
        if not iv.is_symbol or not iv.is_integer:
            raise ValueError("Invalid independent variable.")
        # pylint: disable-msg=too-many-function-args
        return Signal.__new__(cls, amp, iv, period, domain, codomain)
        # pylint: enable-msg=too-many-function-args

    def __str__(self, *_args, **_kwargs):
        return sp.sstr(self.amplitude)

    def _latex(self, printer=None):
        return printer.doprint(self.amplitude)

    def latex(self):
        return sp.latex(self.amplitude, imaginary_unit="rj")

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
    ):
        n = np.array(span)
        v = [float(x.evalf(2)) for x in self[span]]
        if title is None:
            title = r"${0}$".format(self.latex())
        if xlabel is None:
            xlabel = r"${0}$".format(sp.latex(self.iv))
        return ipystem(n, v, xlabel, title, axis, color, marker, markersize)


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
            cls, UnitDelta(iv), iv, None, sp.S.Integers, sp.S.Reals
        )
        # pylint: enable-msg=too-many-function-args
        obj.amplitude.__class__ = UnitDelta
        return obj

    def clone(self, cls, amplitude, **kwargs):
        obj = super().clone(cls, amplitude, **kwargs)
        obj.amplitude.__class__ = UnitDelta
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
            cls, UnitStep(iv), iv, None, sp.S.Integers, sp.S.Reals
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
            cls, UnitRamp(iv), iv, None, sp.S.Integers, sp.S.Reals
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
            expr = DiscreteSignal.deltify(data, start, iv, periodic)
            period = len(data) if periodic else None
            # pylint: disable-msg=too-many-function-args
            obj = DiscreteSignal.__new__(cls, expr, iv, period, sp.S.Integers, codomain)
            # pylint: enable-msg=too-many-function-args
        return obj


class _TrigonometricDiscreteSignal(DiscreteSignal):
    def __init__(self, A, omega, phi):
        DiscreteSignal.__init__(self)
        self.A = sp.sympify(A)
        self.omega = sp.sympify(omega)
        self.phi = sp.sympify(phi)

    @staticmethod
    def _period(omega):
        if omega.is_zero:
            # it's a constant
            return sp.S.One
        if omega.has(sp.S.Pi):
            _, e = omega.as_coeff_exponent(sp.S.Pi)
            if e != 1:
                # there is pi**(e != 1)
                return None
            om, _ = omega.as_independent(sp.S.Pi)
            try:
                r = sp.Rational(str(om))
                return sp.S(2 * r.q)
            except:
                # not rational
                return None
        return None

    @staticmethod
    def _reduce_phase(phi, nonnegative=False):
        phi0 = sp.Mod(phi, 2 * sp.S.Pi)
        if not nonnegative and phi0 >= sp.S.Pi:
            phi0 -= 2 * sp.S.Pi
        return phi0

    @property
    def frequency(self):
        return self.omega

    @property
    def phase(self):
        return self.phi

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
        return obj

    def __init__(self, A=1, omega=None, phi=0, iv=None):
        _TrigonometricDiscreteSignal.__init__(self, A, omega, phi)

    def _eval_extra(self, vals, params):
        if self.gain in params.keys():
            g = sp.S(params[self.gain])
            if not g.is_real:
                raise ValueError("Aplitude must be real.")

    @property
    def gain(self):
        return self.A

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

    def alias(self, interval=0):
        romega = self.reduced_frequency() + 2 * sp.S.Pi * interval
        rphi = self.reduced_phase()
        return Sinusoid(self.gain, romega, rphi, self.iv)

    def __repr__(self):
        return "Sinusoid({0}, {1}, {2}, {3})".format(
            str(self.gain), str(self.frequency), str(self.phase), str(self.iv)
        )
