import sympy as sp
from skdsp.system.system import System
from skdsp.signal.signal import Signal
from skdsp.signal.discrete import n, Delta, Constant
from skdsp.util.lccde import LCCDE

__all__ = [s for s in dir() if not s.startswith("_")]

_x = sp.Function("x", real=True)
_y = sp.Function("y", real=True)


def filter(B, A, x, ci=None):
    # muy preliminar
    # transposed DFII
    mm = max(len(B), len(A))
    B = B + [0] * (mm - len(B))
    B = [sp.S(b) / sp.S(A[0]) for b in B]
    A = A + [0] * (mm - len(A))
    A = [sp.S(a) / sp.S(A[0]) for a in A]
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


class DiscreteSystem(System):
    
    is_discrete = True
    is_continuous = False
    is_hybrid = False
    is_input_discrete = True
    is_output_discrete = True
    is_input_continuous = False
    is_output_continuous = False

    def __new__(cls, T):
        T = sp.sympify(T)
        return System.__new__(cls, T, _x(n), _y(n))

    def impulse_response(self, force_lti=False, select='causal'):
        if self.is_lti:
            return self.apply(Delta(n))
        if self.is_recursive and force_lti:
            # TODO
            return None
        return None

    def convolve(self, other):
        # TODO ¿es todo? ¿separar aquí la convolución periódica?
        if not self.is_lti:
            return None
        if isinstance(other, Signal):
            return self.impulse_response.convolve(other)
        if self.is_compatible(other) and other.is_lti:
            return self.impulse_response.convolve(other.impulse_response)
        raise TypeError("Cannot convolve.")

    @property
    def as_lccde(self):
        try:
            eq = sp.Eq(self.output_, self.mapping)
            return LCCDE.from_expression(eq, self.input_, self.output_)
        except:
            return None     

    def is_fir(self):
        # TODO
        raise NotImplementedError

    def is_iir(self):
        # TODO
        raise NotImplementedError

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

