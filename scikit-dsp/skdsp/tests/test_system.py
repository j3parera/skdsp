import sympy as sp
import pytest

from skdsp.system.system import System
from skdsp.signal.signal import Signal

class Test_System(object):

    def test_System_constructor(self):
        n = sp.Symbol("n", integer=True)
        k = sp.Symbol("k", integer=True)
        t = sp.Symbol("t", real=True)
        tau = sp.Symbol("tau", real=True)

        x = sp.Function('x', real=True)
        y = sp.Function('y', real=True)

        T = sp.Eq(y(n), x(n-1))
        S = System(T, x(n), y(n))
        assert S is not None

        with pytest.raises(ValueError):
            System(T, x=y(n), y=x(n))

        S = System(y(n) - x(n-k) + x(n-2) - y(n-1)**2, x(n), y(n))
        assert S is not None

        T = sp.Eq(y(t), x(t-1))
        S = System(T, x(t), y(t))
        assert S is not None

        with pytest.raises(ValueError):
            System(T, x=y(t), y=x(t))

        S = System(y(t) - x(t-tau) + x(t-2) - y(t-1)**2, x(t), y(t))
        assert S is not None

        T = sp.Eq(y(t), x(n-1))
        S = System(T, x(n), y(t))
        assert S is not None

        with pytest.raises(ValueError):
            System(T, x=y(t), y=x(n))

        S = System(y(t) - x(n-k) + x(n-2) - y(t-1)**2, x(n), y(t))
        assert S is not None

        T = sp.Eq(y(n), x(t-1))
        S = System(T, x(t), y(n))
        assert S is not None

        with pytest.raises(ValueError):
            System(T, x=y(n), y=x(t))

        S = System(y(n) - x(t-tau) + x(t-2) - y(n-1)**2, x(t), y(n))
        assert S is not None

    def test_System_classify_discrete_or_continuous(self):
        n = sp.Symbol("n", integer=True)
        k = sp.Symbol("k", integer=True)
        t = sp.Symbol("t", real=True)
        tau = sp.Symbol("tau", real=True)

        x = sp.Function('x', real=True)
        y = sp.Function('y', real=True)

        T = sp.Eq(y(n), x(n-1))

        S = System(T, x(n), y(n))
        assert S.is_discrete
        assert not S.is_continuous
        assert not S.is_hybrid
        assert S.is_input_discrete
        assert not S.is_input_continuous
        assert S.is_output_discrete
        assert not S.is_output_continuous

        S = System(y(n) - x(n-k) + x(n-2) - y(n-1)**2, x(n), y(n))
        assert S.is_discrete
        assert not S.is_continuous
        assert not S.is_hybrid
        assert S.is_input_discrete
        assert not S.is_input_continuous
        assert S.is_output_discrete
        assert not S.is_output_continuous

        T = sp.Eq(y(t), x(t-1))
        S = System(T, x(t), y(t))
        assert not S.is_discrete
        assert S.is_continuous
        assert not S.is_hybrid
        assert not S.is_input_discrete
        assert S.is_input_continuous
        assert not S.is_output_discrete
        assert S.is_output_continuous

        S = System(y(t) - x(t-tau) + x(t-2) - y(t-1)**2, x(t), y(t))
        assert not S.is_discrete
        assert S.is_continuous
        assert not S.is_hybrid
        assert not S.is_input_discrete
        assert S.is_input_continuous
        assert not S.is_output_discrete
        assert S.is_output_continuous

        T = sp.Eq(y(t), x(n-1))
        S = System(T, x(n), y(t))
        assert not S.is_discrete
        assert not S.is_continuous
        assert S.is_hybrid
        assert S.is_input_discrete
        assert not S.is_input_continuous
        assert not S.is_output_discrete
        assert S.is_output_continuous

        S = System(y(t) - x(n-k) + x(n-2) - y(t-1)**2, x(n), y(t))
        assert not S.is_discrete
        assert not S.is_continuous
        assert S.is_hybrid
        assert S.is_input_discrete
        assert not S.is_input_continuous
        assert not S.is_output_discrete
        assert S.is_output_continuous

        T = sp.Eq(y(n), x(t-1))
        S = System(T, x(t), y(n))
        assert not S.is_discrete
        assert not S.is_continuous
        assert S.is_hybrid
        assert not S.is_input_discrete
        assert S.is_input_continuous
        assert S.is_output_discrete
        assert not S.is_output_continuous

        S = System(y(n) - x(t-tau) + x(t-2) - y(n-1)**2, x(t), y(n))
        assert not S.is_discrete
        assert not S.is_continuous
        assert S.is_hybrid
        assert not S.is_input_discrete
        assert S.is_input_continuous
        assert S.is_output_discrete
        assert not S.is_output_continuous

    def test_System_classify_memory(self):
        n = sp.Symbol("n", integer=True)
        k = sp.Symbol("k", integer=True)
        t = sp.Symbol("t", real=True)
        tau = sp.Symbol("tau", real=True)

        x = sp.Function('x', real=True)
        y = sp.Function('y', real=True)

        T = sp.Eq(y(n), 3*x(n))
        S = System(T, x(n), y(n))
        assert S.is_memoryless
        assert S.is_static
        assert not S.is_dynamic

        S = System(y(n) - x(n-k) + x(n-2) - y(n-1)**2, x(n), y(n))
        assert not S.is_memoryless
        assert not S.is_static
        assert S.is_dynamic

        T = sp.Eq(y(t), x(t))
        S = System(T, x(t), y(t))
        assert S.is_memoryless
        assert S.is_static
        assert not S.is_dynamic

        S = System(y(t) - x(t-tau) + x(t-2) - y(t-1)**2, x(t), y(t))
        assert not S.is_memoryless
        assert not S.is_static
        assert S.is_dynamic

        T = sp.Eq(y(t), x(n))
        S = System(T, x(n), y(t))
        assert S.is_memoryless
        assert S.is_static
        assert not S.is_dynamic

        S = System(y(t) - x(n-k) + x(n-2) - y(t-1)**2, x(n), y(t))
        assert not S.is_memoryless
        assert not S.is_static
        assert S.is_dynamic

        T = sp.Eq(y(n), x(t))
        S = System(T, x(t), y(n))
        assert S.is_memoryless
        assert S.is_static
        assert not S.is_dynamic

        S = System(y(n) - x(t-tau) + x(t-2) - y(n-1)**2, x(t), y(n))
        assert not S.is_memoryless
        assert not S.is_static
        assert S.is_dynamic
