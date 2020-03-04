import pytest
import sympy as sp

from skdsp.signal.discrete import n, Delta, Exponential, Step
from skdsp.system.discrete import Delay, DiscreteSystem, Identity


class Test_Identity(object):
    def test_Identity_misc(self):
        S = Identity()
        assert S.is_discrete
        assert S.is_input_discrete
        assert S.is_output_discrete
        assert not S.is_continuous
        assert not S.is_input_continuous
        assert not S.is_input_continuous
        assert not S.is_hybrid
        assert S.is_memoryless
        assert S.is_time_invariant
        assert S.is_linear
        assert S.is_causal
        assert not S.is_anticausal
        assert not S.is_recursive
        assert S.is_stable

    def test_Identity_impulse_response(self):
        S = Identity()
        h = S.impulse_response
        assert h == Delta()


class Test_Delay(object):
    def test_Delay_misc(self):
        S = Delay(sp.Symbol('k', integer=True, positive=True))
        assert S.is_discrete
        assert S.is_input_discrete
        assert S.is_output_discrete
        assert not S.is_continuous
        assert not S.is_input_continuous
        assert not S.is_input_continuous
        assert not S.is_hybrid
        assert not S.is_memoryless
        assert S.is_time_invariant
        assert S.is_linear
        assert S.is_causal
        assert not S.is_anticausal
        assert not S.is_recursive
        assert S.is_stable

        S = Delay(sp.Symbol('k', integer=True))
        assert not S.is_causal

        with pytest.raises(ValueError):
            Delay(sp.Symbol('N'))

        with pytest.raises(ValueError):
            Delay(2*n)

    def test_Delay_impulse_response(self):
        S = Delay(3)
        h = S.impulse_response
        assert h == Delta().shift(3)
