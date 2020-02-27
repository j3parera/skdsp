import numpy as np
import pytest
import sympy as sp

import skdsp.signal.discrete as ds
from skdsp.signal.functions import UnitDelta, UnitStep, UnitRamp, UnitDeltaTrain
from skdsp.signal.util import stem


class Test_Discrete_Arithmetic(object):
    def test_Arithmetic_neg(self):
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

    def test_Arithmetic_add(self):
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

    def test_Arithmetic_sub(self):
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

    def test_Arithmetic_mul(self):
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

        # problemas con s = sp.I * ds.Delta() porque sympy lo intenta interpretar como Expr * Expr

    def test_Arithmetic_div(self):
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
        N = sp.Symbol('N', integer=True, nonnegative=True)
        M = sp.Symbol('M', integer=True)
        r = sp.Symbol('r')

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


class Test_Discrete_factories(object):
    def test_Discrete_from_sampling(self):
        # TODO Otros muestreos
        t = sp.Symbol("t", real=True)
        n = sp.Symbol("n", integer=True)
        fs = 8e3
        cs = 50 * sp.cos(2 * sp.S.Pi * 1200 * t + sp.S.Pi / 4)
        s = ds.DiscreteSignal.from_sampling(cs, t, n, fs)
        d = ds.Sinusoid(50, 3 * sp.S.Pi / 10, sp.S.Pi / 4, n)
        assert s == d
        cs = 50 * sp.cos(1200 * t + sp.S.Pi / 4)
        s = ds.DiscreteSignal.from_sampling(cs, t, n, fs)
        d = ds.Sinusoid(50, sp.Rational(3, 20), sp.S.Pi / 4, n)
        assert s == d
        cs = 50 * sp.sin(1200 * t + sp.S.Pi / 4)
        s = ds.DiscreteSignal.from_sampling(cs, t, n, fs)
        d = ds.Sinusoid(50, sp.Rational(3, 20), 3 * sp.S.Pi / 4, n)
        assert s == d

    # TODO
    # def test_Discrete_from_formula(self):


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

    # def test_Stem(self):
    #     x = 0.9 * ds.Delta().shift(5)
    #     x.stem(range(-1, 7))

    def test_Plot(self):
        x = ds.DeltaTrain(N=4)
        ax = x.ipystem(range(-3, 16))
        assert ax is not None
        # stem(x.amplitude, (ds.n, -3, 15))


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
    def test_KK_1(self):
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

    def test_KK_2(self):
        a = sp.Symbol("a", real=True)
        y0 = ds.Exponential(alpha=a)
        ya = y0.subs({a: 0.9})
        assert isinstance(ya, ds.Exponential)

        y1 = a * y0.shift(1)
        ya = y1.subs({a: 0.9})
        assert isinstance(ya, ds.Exponential)

    def test_KK_3(self):
        import re
        x = ds.Sinusoid(omega=sp.S.Pi * sp.Rational(1, 17), phi=-sp.S.Pi * sp.S.Half)
        ltx = x.latex()
        assert ltx != ''
