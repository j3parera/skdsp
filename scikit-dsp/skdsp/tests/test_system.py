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

        x = sp.Function("x", real=True)
        y = sp.Function("y", real=True)

        T = sp.Eq(y(n), x(n - 1))
        S = System(T, x(n), y(n))
        assert S is not None

        with pytest.raises(ValueError):
            System(T, x=y(n), y=x(n))

        S = System(y(n) - x(n - k) + x(n - 2) - y(n - 1) ** 2, x(n), y(n))
        assert S is not None

        T = sp.Eq(y(t), x(t - 1))
        S = System(T, x(t), y(t))
        assert S is not None

        with pytest.raises(ValueError):
            System(T, x=y(t), y=x(t))

        S = System(y(t) - x(t - tau) + x(t - 2) - y(t - 1) ** 2, x(t), y(t))
        assert S is not None

        T = sp.Eq(y(t), x(n - 1))
        S = System(T, x(n), y(t))
        assert S is not None

        with pytest.raises(ValueError):
            System(T, x=y(t), y=x(n))

        S = System(y(t) - x(n - k) + x(n - 2) - y(t - 1) ** 2, x(n), y(t))
        assert S is not None

        T = sp.Eq(y(n), x(t - 1))
        S = System(T, x(t), y(n))
        assert S is not None

        with pytest.raises(ValueError):
            System(T, x=y(n), y=x(t))

        S = System(y(n) - x(t - tau) + x(t - 2) - y(n - 1) ** 2, x(t), y(n))
        assert S is not None

    def test_System_classify_discrete_or_continuous(self):
        n = sp.Symbol("n", integer=True)
        k = sp.Symbol("k", integer=True)
        t = sp.Symbol("t", real=True)
        tau = sp.Symbol("tau", real=True)

        x = sp.Function("x", real=True)
        y = sp.Function("y", real=True)

        T = sp.Eq(y(n), x(n - 1))

        S = System(T, x(n), y(n))
        assert S.is_discrete
        assert not S.is_continuous
        assert not S.is_hybrid
        assert S.is_input_discrete
        assert not S.is_input_continuous
        assert S.is_output_discrete
        assert not S.is_output_continuous

        S = System(y(n) - x(n - k) + x(n - 2) - y(n - 1) ** 2, x(n), y(n))
        assert S.is_discrete
        assert not S.is_continuous
        assert not S.is_hybrid
        assert S.is_input_discrete
        assert not S.is_input_continuous
        assert S.is_output_discrete
        assert not S.is_output_continuous

        T = sp.Eq(y(t), x(t - 1))
        S = System(T, x(t), y(t))
        assert not S.is_discrete
        assert S.is_continuous
        assert not S.is_hybrid
        assert not S.is_input_discrete
        assert S.is_input_continuous
        assert not S.is_output_discrete
        assert S.is_output_continuous

        S = System(y(t) - x(t - tau) + x(t - 2) - y(t - 1) ** 2, x(t), y(t))
        assert not S.is_discrete
        assert S.is_continuous
        assert not S.is_hybrid
        assert not S.is_input_discrete
        assert S.is_input_continuous
        assert not S.is_output_discrete
        assert S.is_output_continuous

        T = sp.Eq(y(t), x(n - 1))
        S = System(T, x(n), y(t))
        assert not S.is_discrete
        assert not S.is_continuous
        assert S.is_hybrid
        assert S.is_input_discrete
        assert not S.is_input_continuous
        assert not S.is_output_discrete
        assert S.is_output_continuous

        S = System(y(t) - x(n - k) + x(n - 2) - y(t - 1) ** 2, x(n), y(t))
        assert not S.is_discrete
        assert not S.is_continuous
        assert S.is_hybrid
        assert S.is_input_discrete
        assert not S.is_input_continuous
        assert not S.is_output_discrete
        assert S.is_output_continuous

        T = sp.Eq(y(n), x(t - 1))
        S = System(T, x(t), y(n))
        assert not S.is_discrete
        assert not S.is_continuous
        assert S.is_hybrid
        assert not S.is_input_discrete
        assert S.is_input_continuous
        assert S.is_output_discrete
        assert not S.is_output_continuous

        S = System(y(n) - x(t - tau) + x(t - 2) - y(n - 1) ** 2, x(t), y(n))
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

        x = sp.Function("x", real=True)
        y = sp.Function("y", real=True)

        T = sp.Eq(y(n), 3 * x(n))
        S = System(T, x(n), y(n))
        assert S.is_memoryless
        assert S.is_static
        assert not S.is_dynamic

        S = System(y(n) - x(n - k) + x(n - 2) - y(n - 1) ** 2, x(n), y(n))
        assert not S.is_memoryless
        assert not S.is_static
        assert S.is_dynamic

        T = sp.Eq(y(t), x(t))
        S = System(T, x(t), y(t))
        assert S.is_memoryless
        assert S.is_static
        assert not S.is_dynamic

        S = System(y(t) - x(t - tau) + x(t - 2) - y(t - 1) ** 2, x(t), y(t))
        assert not S.is_memoryless
        assert not S.is_static
        assert S.is_dynamic

        T = sp.Eq(y(t), x(n))
        S = System(T, x(n), y(t))
        assert S.is_memoryless
        assert S.is_static
        assert not S.is_dynamic

        S = System(y(t) - x(n - k) + x(n - 2) - y(t - 1) ** 2, x(n), y(t))
        assert not S.is_memoryless
        assert not S.is_static
        assert S.is_dynamic

        T = sp.Eq(y(n), x(t))
        S = System(T, x(t), y(n))
        assert S.is_memoryless
        assert S.is_static
        assert not S.is_dynamic

        S = System(y(n) - x(t - tau) + x(t - 2) - y(n - 1) ** 2, x(t), y(n))
        assert not S.is_memoryless
        assert not S.is_static
        assert S.is_dynamic

    def test_System_apply(self):
        n = sp.Symbol("n", integer=True)
        k = sp.Symbol("k", integer=True)
        t = sp.Symbol("t", real=True)
        tau = sp.Symbol("tau", real=True)
        omega = sp.Symbol("omega", real=True)
        phi = sp.Symbol("phi", real=True)

        x = sp.Function("x", real=True)
        y = sp.Function("y", real=True)
        sx = Signal(sp.cos(omega * n + phi), n, codomain=sp.S.Reals)

        T = sp.Eq(y(n), 3 * x(n))
        S = System(T, x(n), y(n))
        sy = S.apply(sx)
        assert isinstance(sy, Signal)
        assert sy.amplitude == 3 * sp.cos(omega * n + phi)
        sy2 = S.eval(sx)
        assert sy == sy2
        sy2 = S(sx)
        assert sy == sy2

        T = sp.Eq(y(n), x(n - 1))
        S = System(T, x(n), y(n))
        sy = S.apply(sx)
        assert isinstance(sy, Signal)
        assert sy.amplitude == sp.cos(omega * (n - 1) + phi)
        sy2 = S.eval(sx)
        assert sy == sy2
        sy2 = S(sx)
        assert sy == sy2

        T = sp.Eq(y(k), 2 * x(k - 1)**3 + 3 * x(k - 2))
        S = System(T, x(k), y(k))
        sy = S.apply(sx)
        assert isinstance(sy, Signal)
        assert sy.amplitude == 2 * sp.cos(omega * (n - 1) + phi)**3 + 3 * sp.cos(omega * (n - 2) + phi)
        sy2 = S.eval(sx)
        assert sy == sy2
        sy2 = S(sx)
        assert sy == sy2

        T = sp.Eq(y(t), x(t - tau) + x(t - 2) - y(t - 1) ** 2)
        S = System(T, x(t), y(t))
        sx = Signal(2**t)
        sy = S.apply(sx)
        assert isinstance(sy, Signal)
        assert sp.simplify(sy.amplitude - (2**(t - tau) + 2**(t - 2) - y(t - 1)**2)) == sp.S.Zero
        sy2 = S.eval(sx)
        assert sy == sy2
        sy2 = S(sx)
        assert sy == sy2

    def test_System_invariance(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        k = sp.Symbol("k", integer=True)
        tau = sp.Symbol("tau", real=True)
        x = sp.Function("x", real=True)
        y = sp.Function("y", real=True)
        h = sp.Function("h", real=True)
        g = sp.Function("g", real=True)

        T = sp.Eq(y(n), x(n-1))
        S = System(T, x(n), y(n))
        assert S.is_time_invariant
        assert not S.is_time_variant
        assert S.is_shift_invariant
        assert not S.is_shift_variant

        T = y(n) - g(n)*x(n)
        S = System(T, x(n), y(n))
        assert not S.is_time_invariant
        assert S.is_time_variant
        assert not S.is_shift_invariant
        assert S.is_shift_variant

        T = y(n) - x(n)**2
        S = System(T, x(n), y(n))
        assert S.is_time_invariant
        assert not S.is_time_variant
        assert S.is_shift_invariant
        assert not S.is_shift_variant

        T = y(n) - n*x(n)
        S = System(T, x(n), y(n))
        assert not S.is_time_invariant
        assert S.is_time_variant
        assert not S.is_shift_invariant
        assert S.is_shift_variant

        T = y(n) - x(-n)
        S = System(T, x(n), y(n))
        assert not S.is_time_invariant
        assert S.is_time_variant
        assert not S.is_shift_invariant
        assert S.is_shift_variant

        T = sp.Eq(y(t), x(t-1))
        S = System(T, x(t), y(t))
        assert S.is_time_invariant
        assert not S.is_time_variant
        assert S.is_shift_invariant
        assert not S.is_shift_variant

        T = y(t) - g(t)*x(t)
        S = System(T, x(t), y(t))
        assert not S.is_time_invariant
        assert S.is_time_variant
        assert not S.is_shift_invariant
        assert S.is_shift_variant

        T = y(t) - x(t)**2
        S = System(T, x(t), y(t))
        assert S.is_time_invariant
        assert not S.is_time_variant
        assert S.is_shift_invariant
        assert not S.is_shift_variant

        T = y(t) - t*x(t)
        S = System(T, x(t), y(t))
        assert not S.is_time_invariant
        assert S.is_time_variant
        assert not S.is_shift_invariant
        assert S.is_shift_variant

        T = y(t) - x(-t)
        S = System(T, x(t), y(t))
        assert not S.is_time_invariant
        assert S.is_time_variant
        assert not S.is_shift_invariant
        assert S.is_shift_variant

    def test_System_linearity(self):
        pass