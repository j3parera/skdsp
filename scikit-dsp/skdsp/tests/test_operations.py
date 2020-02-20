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

    def test_Discrete_abs(self):
        s = ds.Sinusoid(2, sp.S.Pi / 4)
        assert s.odd == s
        assert s.even == 0
        assert s.abs == ds.DiscreteSignal.from_formula(
            sp.Abs(2 * sp.cos(sp.S.Pi * ds.n / 4))
        )

        s = ds.Sinusoid(2, sp.S.Pi / 4, -sp.S.Pi / 2)
        assert s.odd == 0
        assert s.even == s
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

    # TODO
    # def test_Discrete_from_formula(self):


class Test_Print(object):
    def test_Latex(self):
        x = 0.9 * ds.Delta().shift(5)
        s = x.latex()
        assert s == "0.9 \\delta\\left[n - 5\\right]"

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

