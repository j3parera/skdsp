import sympy as sp

from skdsp.signal.discrete import Constant, Delta, DiscreteSignal, Ramp, Step, n
from skdsp.signal.functions import UnitDelta, UnitStep, stepsimp
from skdsp.signal.signal import Signal
from skdsp.system.system import System
from skdsp.util.lccde import LCCDE

__all__ = [s for s in dir() if not s.startswith("_")]

_x = sp.Function("x")
_y = sp.Function("y")


class DiscreteSystem(System):

    is_discrete = True
    is_continuous = False
    is_hybrid = False
    is_input_discrete = True
    is_output_discrete = True
    is_input_continuous = False
    is_output_continuous = False

    @classmethod
    def from_coefficients(cls, B, A, x=None, y=None):
        lccde = LCCDE(B, A, x, y)
        obj = DiscreteSystem.__new__(cls, lccde)
        return obj

    def __new__(cls, T, x=_x(n), y=_y(n)):
        T = sp.sympify(T)
        return System.__new__(cls, T, x, y)

    @property
    def as_lccde(self):
        try:
            eq = sp.Eq(self.output_, self.mapping)
            return LCCDE.from_expression(eq, self.input_, self.output_)
        except:
            return None

    @property
    def is_stable(self):
        lccde = self.as_lccde
        if lccde is None:
            try:
                h = self.impulse_response()
                if h is not None and h.is_abs_summable:
                    return True
            except:
                pass
            return super().is_stable
        if self.is_fir:
            return True
        y_roots = lccde.y_roots
        stable = all(sp.Abs(r) < sp.S.One for r in y_roots.keys())
        return stable

    def _response(self, sin, force_lti=False, select="causal"):
        if self.is_lti:
            return self.apply(sin)
        lccde = self.as_lccde
        if lccde is not None and force_lti:
            he = lccde.solve(sin.amplitude, ac=select)
            return DiscreteSignal.from_formula(he, lccde.iv)
        return None

    def impulse_response(self, force_lti=False, select="causal"):
        return self._response(Delta(n), force_lti, select)

    def step_response(self, force_lti=False, select="causal"):
        return self._response(Step(n), force_lti, select)

    def ramp_response(self, force_lti=False, select="causal"):
        return self._response(Ramp(n), force_lti, select)

    def convolve(self, other):
        # TODO ¿es todo? ¿separar aquí la convolución periódica?
        if not self.is_lti:
            return None
        if isinstance(other, Signal):
            return self.impulse_response.convolve(other)
        if self.is_compatible(other) and other.is_lti:
            return self.impulse_response.convolve(other.impulse_response)
        raise TypeError("Cannot convolve.")

    def as_rational(self, sym=None, cancel=False):
        lccde = self.as_lccde
        if lccde is None:
            return None
        if sym is None:
            sym = sp.Symbol('z1')
        num = sp.Poly(reversed(lccde.B), sym)
        den = sp.Poly(reversed(lccde.A), sym)
        rat = num / den
        if cancel:
            rat = sp.cancel(rat)
        return rat
    
    @property
    def is_fir(self):
        sym = sp.Symbol('z1')
        rat = self.as_rational(sym=sym, cancel=True)
        if rat is None:
            return not self._depends_on_outputs
        num, den = rat.as_numer_denom()
        return den.is_constant(sym)

    @property
    def is_iir(self):
        return not self.is_fir

    def filter(self, x, ci=None, span=None):
        lccde = self.as_lccde
        if lccde is None:
            raise TypeError("The system cannot implement a filter.")
        if isinstance(x, DiscreteSignal):
            if span is None:
                raise TypeError("Cannot filter an everlasting signal.")
            else:
                x = x(span)
        # muy preliminar
        # transposed DFII
        mm = max(len(lccde.B), len(lccde.A))
        B = lccde.B + [0] * (mm - len(lccde.B))
        A = lccde.A + [0] * (mm - len(lccde.A))
        M = sp.S(ci) if ci is not None else sp.S([0] * max(1, (mm - 1)))
        Y = sp.S([0] * len(x))
        x = sp.S(x)
        for k, v in enumerate(x):
            y = B[0] * v + M[0]
            Y[k] = y
            for m in range(0, len(M) - 1):
                M[m] = B[m + 1] * v - A[m + 1] * y + M[m + 1]
            M[-1] = B[-1] * v - A[-1] * y
        return Y


class Zero(DiscreteSystem):

    _depends_on_inputs = False
    _depends_on_outputs = False
    is_memoryless = True
    is_time_invariant = True
    is_linear = True
    is_causal = True
    is_anticausal = False
    is_recursive = False
    is_stable = True
    is_lti = True

    def __new__(cls):
        T = sp.Eq(_y(n), sp.S.Zero)
        return DiscreteSystem.__new__(cls, T)

    def apply(self, _ins):
        return Constant(0)


Null = Zero


class Identity(DiscreteSystem):

    _depends_on_inputs = True
    _depends_on_outputs = False
    is_memoryless = True
    is_time_invariant = True
    is_linear = True
    is_causal = True
    is_anticausal = False
    is_recursive = False
    is_stable = True
    is_lti = True

    def __new__(cls):
        T = sp.Eq(_y(n), _x(n))
        return DiscreteSystem.__new__(cls, T)

    def apply(self, ins):
        return ins


One = Identity


class Delay(DiscreteSystem):

    _depends_on_inputs = True
    _depends_on_outputs = False
    is_memoryless = False
    is_time_invariant = True
    is_linear = True
    is_recursive = False
    is_stable = True
    is_lti = True

    def __new__(cls, k):
        k = sp.S(k)
        if not k.is_integer or not k.is_constant(n):
            raise ValueError("Delay must be integer constant.")
        T = sp.Eq(_y(n), _x(n - k))
        return DiscreteSystem.__new__(cls, T)

