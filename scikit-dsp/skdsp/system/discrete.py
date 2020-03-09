import sympy as sp
from skdsp.system.system import System
from skdsp.signal.signal import Signal
from skdsp.signal.discrete import n, Delta

__all__ = [s for s in dir() if not s.startswith("_")]

_x = sp.Function("x", real=True)
_y = sp.Function("y", real=True)


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

    @property
    def impulse_response(self):
        if not self.is_lti:
            return None
        return self.apply(Delta(n))

    def convolve(self, other):
        # TODO ¿es todo? ¿separar aquí la convolución periódica?
        if not self.is_lti:
            return None
        if isinstance(other, Signal):
            return self.impulse_response.convolve(other)
        if self.is_compatible(other) and other.is_lti:
            return self.impulse_response.convolve(other.impulse_response)
        raise TypeError("Cannot convolve.")

    def is_fir(self):
        # TODO
        raise NotImplementedError

    def is_iir(self):
        # TODO
        raise NotImplementedError

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

