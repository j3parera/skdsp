import numpy as np
import pytest
import sympy as sp

import skdsp.signal.discrete as ds
from skdsp.signal.functions import UnitDelta, UnitStep, UnitRamp, UnitDeltaTrain
from skdsp.util.util import stem


class Test_Discrete_Arithmetic(object):
    def test_Discrete_neg(self):
        s = -ds.Delta()
        assert s[-1:3] == [0, -1, 0, 0]
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = ds.Constant(0)
        for shift in range(0, 3):
            s += -(ds.Delta() >> shift)
        assert s[-1:5] == [0, -1, -1, -1, 0, 0]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = -ds.Delta() + 3
        assert s[-1:4] == [3, 2, 3, 3, 3]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

    def test_Discrete_add(self):
        s = ds.Delta() + ds.Step()
        assert s[-2:3] == [0, 0, 2, 1, 1]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Constant(0)
        for shift in range(0, 3):
            s += ds.Delta() >> shift
        assert s[-1:5] == [0, 1, 1, 1, 0, 0]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Delta() + 3
        assert s[-1:4] == [3, 4, 3, 3, 3]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Delta() + 0
        assert s == ds.Delta()
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = 0 + ds.Delta()
        assert s == ds.Delta()
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = ds.Delta() + ds.Constant(0)
        assert s == ds.Delta()
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Constant(0) + ds.Delta()
        assert s == ds.Delta()
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Delta()
        assert s == s + 0
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        assert s == 0 + s
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        assert s == s + ds.Constant(0)
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True
        assert s == ds.Constant(0) + s
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

    def test_Discrete_sub(self):
        s = ds.Delta() - ds.Step()
        assert s[-2:3] == [0, 0, 0, -1, -1]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Constant(0)
        for shift in range(0, 3):
            s -= ds.Delta() >> shift
        assert s[-1:5] == [0, -1, -1, -1, 0, 0]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Delta() - 3
        assert s[-1:4] == [-3, -2, -3, -3, -3]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Delta() - 0
        assert s == ds.Delta()
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = 0 - ds.Delta()
        assert s == -ds.Delta()
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = ds.Delta() - ds.Constant(0)
        assert s == ds.Delta()
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = ds.Constant(0) - ds.Delta()
        assert s == -ds.Delta()
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = ds.Delta()
        assert s == s - 0
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        assert -s == 0 - s
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        assert s == s - ds.Constant(0)
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        assert -s == ds.Constant(0) - s
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        x1 = ds.Delta() + ds.Delta().shift(1)
        x2 = ds.Delta().shift(1)
        x3 = x1 - x2
        assert isinstance(x3, ds.Delta)

    def test_Discrete_mul(self):
        s = ds.Delta() * ds.Step()
        assert s[-1:2] == [0, 1, 0]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Delta()
        assert s * 0 == ds.Constant(0)
        assert isinstance(s * 0, ds.Constant)
        assert s.is_discrete == True
        assert 0 * s == ds.Constant(0)
        assert isinstance(0 * s, ds.Constant)
        assert s.is_discrete == True
        assert ds.Constant(0) == s * ds.Constant(0)
        assert isinstance(s * ds.Constant(0), ds.Constant)
        assert s.is_discrete == True
        assert ds.Constant(0) == ds.Constant(0) * s
        assert isinstance(ds.Constant(0) * s, ds.Constant)
        assert s.is_discrete == True
        assert s == s * 1
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        assert s == 1 * s
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        assert s == s * ds.Constant(1)
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        assert s == ds.Constant(1) * s
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = ds.Constant(1)
        for shift in range(0, 3):
            s *= ds.Delta() >> shift
        assert s[-1:5] == [0, 0, 0, 0, 0, 0]
        assert isinstance(s, ds.DiscreteSignal)
        assert s.is_discrete == True

        s = ds.Ramp() * 3
        assert s[-1:4] == [0, 0, 3, 6, 9]
        assert isinstance(s, ds.Ramp)
        assert s.is_discrete == True

        s = 3 * ds.Ramp()
        assert s[-1:4] == [0, 0, 3, 6, 9]
        assert isinstance(s, ds.Ramp)
        assert s.is_discrete == True

        s = ds.Delta() * complex(2, 2)
        assert s[-1:2] == [0, 2.0 + 2.0 * sp.I, 0]
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = complex(2, 2) * ds.Delta()
        assert s[-1:2] == [0, 2.0 + 2.0 * sp.I, 0]
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = sp.I * ds.Delta()
        assert s[-1:2] == [0, sp.I, 0]
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        s = 2 * ds.Step().shift(3) * ds.Step().shift(5)
        assert s == 2 * ds.Step().shift(5)

        s = ds.Step().shift(3).flip() * ds.Step().shift(5).flip()
        assert s == ds.Step().shift(5).flip()

        s = ds.Step().shift(3) * ds.Step().shift(5).flip()
        assert s == ds.Constant(0)

        s = ds.Step().shift(3) * ds.Step().shift(-5).flip()
        assert s == ds.Step().shift(3) - ds.Step().shift(6)

    def test_Discrete_div(self):
        with pytest.raises(TypeError):
            s = ds.Delta() / ds.Step()

        s = ds.Delta()
        with pytest.raises(ZeroDivisionError):
            s / 0

        with pytest.raises(TypeError):
            0 / s

        with pytest.raises(ZeroDivisionError):
            s / ds.Constant(0)

        with pytest.raises(TypeError):
            ds.Constant(0) / s

        assert s == s / 1
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        with pytest.raises(TypeError):
            1 / s
        assert s == s / ds.Constant(1)
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True
        with pytest.raises(TypeError):
            ds.Constant(1) / s

        s = ds.Constant(1)
        with pytest.raises(TypeError):
            s /= ds.Delta()

        s = ds.Ramp() / 3
        assert s[-1:4] == [0, 0, sp.Rational(1, 3), sp.Rational(2, 3), 1]
        assert isinstance(s, ds.Ramp)
        assert s.is_discrete == True
        s /= 2
        assert s[-1:4] == [0, 0, sp.Rational(1, 6), sp.Rational(2, 6), sp.S.Half]
        assert isinstance(s, ds.Ramp)
        assert s.is_discrete == True

        with pytest.raises(TypeError):
            3 / ds.Ramp()

        s = ds.Delta() / complex(2, 2)
        assert s[-1:2] == [0, 0.25 - 0.25 * sp.I, 0]
        assert isinstance(s, ds.Delta)
        assert s.is_discrete == True

        with pytest.raises(TypeError):
            s = complex(2, 2) / ds.Delta()

    def test_Discrete_pow(self):
        N = sp.Symbol("N", integer=True, nonnegative=True)
        M = sp.Symbol("M", integer=True)
        r = sp.Symbol("r")

        s = ds.Delta() ** 2
        assert s == ds.Delta()

        s = ds.Delta() ** 3
        assert s == ds.Delta()

        s = ds.Step() ** 3
        assert s == ds.Step()

        s = ds.Delta() ** 0
        assert s == ds.Constant(1)

        s = ds.Ramp() ** 2
        assert s.amplitude == ds.n ** 2 * UnitStep(ds.n)

        s = ds.Ramp() ** N
        assert s.amplitude == ds.n ** N * UnitStep(ds.n)

        with pytest.raises(ValueError):
            s = ds.Ramp() ** M

        with pytest.raises(ValueError):
            s = ds.Ramp() ** r

        with pytest.raises(ValueError):
            s = ds.Delta() ** (-2)

        with pytest.raises(ValueError):
            ds.Delta() ** (1 / 2)

        with pytest.raises(ValueError):
            ds.Delta() ** sp.sqrt(2)

class Test_Discrete_Other(object):
    def test_Discrete_abs(self):
        s = ds.Sinusoid(2, sp.S.Pi / 4)
        assert s.odd_part == s
        assert s.even_part == 0
        assert s.abs == ds.DiscreteSignal.from_formula(
            sp.Abs(2 * sp.cos(sp.S.Pi * ds.n / 4))
        )

        s = ds.Sinusoid(2, sp.S.Pi / 4, -sp.S.Pi / 2)
        assert s.odd_part == 0
        assert s.even_part == s
        assert s.abs == ds.DiscreteSignal.from_formula(
            sp.Abs(2 * sp.sin(sp.S.Pi * ds.n / 4))
        )

    def test_Discrete_convolution(self):
        a = sp.Symbol('a', real=True)
        s1 = ds.Exponential(1, a) * ds.Step()
        s2 = ds.Step()
        s3 = s1 @ s2
        s3 = s3.subs({sp.Eq(a, 1): False})
        expected = (1 - a ** (ds.n + 1)) / (1 - a)*ds.Step()
        assert sp.simplify((s3 - expected).amplitude) == sp.S.Zero
        s3 = s1.convolve(s2).subs({sp.Eq(a, 1): False})
        assert sp.simplify((s3 - expected).amplitude) == sp.S.Zero
        s3 = s2.convolve(s1).subs({sp.Eq(1/a, 1): False})
        assert sp.simplify((s3 - expected).amplitude) == sp.S.Zero
        s1 @= s2
        s1 = s1.subs({sp.Eq(a, 1): False})
        assert sp.simplify((s1 - expected).amplitude) == sp.S.Zero

    def test_Discrete_correlate(self):
        # x = ds.DataSignal([2, -1, 3, 7, 1, 2, -3], start=-4)
        # y = ds.DataSignal([1, -1, 2, -2, 4, 1, -2, 5], start=-4)
        # z = x.correlate(y)
        # # TODO
        # assert z
        # z = x.cross_correlate(y)
        # # TODO
        # assert z
        # z = y.correlate(x)
        # # TODO
        # assert z
        # z = x.auto_correlate()
        # # TODO
        # assert z
        # z = x.auto_correlate(normalized=True)
        # # TODO
        # assert z

        # TODO Los límites fallan y más cuando hay símbolos a los que no se pueden
        # poner restricciones (assumptions) como 0 < a < 1
        a = sp.Symbol('a', real=True, positive=True)
        # x1 = ds.Exponential(alpha=a, iv=ds.n)
        # x2 = ds.Step(ds.n)
        x = ds.DiscreteSignal.from_formula(a**ds.n*UnitStep(ds.n), iv=ds.n)
        rxx = x.auto_correlate()
        assert rxx

class Test_Print(object):
    def test_Latex(self):
        x = 0.9 * ds.Delta().shift(5)
        s = x.latex()
        assert s == "0.9 \\delta\\left[n - 5\\right]"

        x = ds.Exponential(1, 0.9 * sp.exp(sp.I * sp.S.Pi / 11))
        s = x.latex()
        assert s is not None

    def test_Display(self):
        x = 0.9 * ds.Delta().shift(5)
        s = x.display()
        assert (
            s
            == "{ \u22ef 0, 0, 0, _0_, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 \u22ef }"
        )

        s = x.display(range(-1, 7))
        assert s == "{ \u22ef 0, _0_, 0, 0, 0, 0, 0.9, 0 \u22ef }"

    def test_Plot(self):
        x = ds.DeltaTrain(period=4)
        ax = x.ipystem(range(-3, 16))
        assert ax is not None


class Test_Summations(object):
    def test_Sum_Deltas(self):
        s = ds.Delta()
        S = s.sum(-2, -1)
        assert S == 0
        S = s.sum(-2, 0)
        assert S == 1
        S = s.sum(-2, 2)
        assert S == 1
        S = s.sum(0, 0)
        assert S == 1
        S = s.sum(0, 2)
        assert S == 1
        S = s.sum(1, 2)
        assert S == 0

        L = sp.symbols("L", integer=True, positive=True)
        s = ds.Delta().shift(L)  # L > 0
        S = s.sum(-2, -1)
        assert S == 0
        S = s.sum(-2, 0)
        assert S == 0
        S = s.sum(-2, 2)
        assert S == sp.Piecewise((1, sp.Le(L, 2)), (0, True))
        S = s.sum(0, 0)
        assert S == 0
        S = s.sum(0, 2)
        assert S == sp.Piecewise((1, sp.Le(L, 2)), (0, True))
        S = s.sum(1, 2)
        assert S == sp.Piecewise((1, sp.Le(L, 2)), (0, True))

        L = sp.symbols("L", integer=True, negative=True)
        s = ds.Delta().shift(L)  # L < 0
        S = s.sum(-2, -1)
        assert S == sp.Piecewise((1, sp.Ge(L, -2)), (0, True))
        S = s.sum(-2, 0)
        assert S == sp.Piecewise((1, sp.Ge(L, -2)), (0, True))
        S = s.sum(-2, 2)
        assert S == sp.Piecewise((1, sp.Ge(L, -2)), (0, True))
        S = s.sum(0, 0)
        assert S == 0
        S = s.sum(0, 2)
        assert S == 0
        S = s.sum(1, 2)
        assert S == 0

    def test_Sum_Steps(self):
        L = sp.symbols("L", integer=True, positive=True)
        s = ds.Step()
        S = s.sum(0, L - 1)
        assert S == L
        S = s.sum(-L, L)
        assert S == L + 1
        S = s.sum()
        assert S == sp.S.Infinity

        s = ds.Step() - ds.Step().shift(1)
        S = s.sum()
        assert S == 1
        S = s.sum(-5, 5)
        assert S == 1
        S = s.sum(-5, -3)
        assert S == 0
        S = s.sum(3, 5)
        assert S == 0

        s = ds.Step() - ds.Step().shift(4)
        S = s.sum()
        assert S == 4
        S = s.sum(-5, 5)
        assert S == 4
        S = s.sum(-5, -3)
        assert S == 0
        S = s.sum(3, 5)
        assert S == 1

        s = ds.Step().shift(-4) - ds.Step().shift(4)
        S = s.sum()
        assert S == 8
        S = s.sum(-5, 5)
        assert S == 8
        S = s.sum(-5, -3)
        assert S == 2
        S = s.sum(3, 5)
        assert S == 1

        s = ds.Step().shift(4) - ds.Step()
        S = s.sum()
        assert S == -4
        S = s.sum(-5, 5)
        assert S == -4
        S = s.sum(-5, -3)
        assert S == 0
        S = s.sum(3, 5)
        assert S == -1

        s = ds.Step().shift(4).flip() - ds.Step()
        S = s.sum(-5, 5)
        assert S == -4
        S = s.sum(-5, -3)
        assert S == 2
        S = s.sum(3, 5)
        assert S == -3

        s = ds.Step().flip() - ds.Step().shift(8).flip()
        S = s.sum(-5, 5)
        assert S == 6
        S = s.sum(-5, -3)
        assert S == 3
        S = s.sum(3, 5)
        assert S == 0

        s = ds.Step().shift(4).flip() - ds.Step().flip()
        S = s.sum(-5, 5)
        assert S == -4
        S = s.sum(-5, -3)
        assert S == -1
        S = s.sum(3, 5)
        assert S == 0

    
class Test_KK(object):
    def test_transmute_exponential(self):
        x = ds.Exponential(alpha=0.9)
        y = 3 * x
        assert isinstance(y, ds.Exponential)

        a = sp.Symbol("a", real=True)
        x = ds.Exponential(alpha=a)
        y = a * x
        assert isinstance(y, ds.Exponential)
        assert y.amplitude == a ** (ds.n + 1)

        y = a * x.shift(1)
        assert isinstance(y, ds.Exponential)
        assert y.amplitude == a ** ds.n

    def test_subs_exponential(self):
        a = sp.Symbol("a", real=True)
        y0 = ds.Exponential(alpha=a)
        ya = y0.subs({a: 0.9})
        assert isinstance(ya, ds.Exponential)

        y1 = a * y0.shift(1)
        ya = y1.subs({a: 0.9})
        assert isinstance(ya, ds.Exponential)

        with pytest.raises(ValueError):
            y1.subs({ds.n: sp.Symbol('m')})
