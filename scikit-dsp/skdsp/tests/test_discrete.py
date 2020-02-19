import numpy as np
import pytest
import sympy as sp

import skdsp.signal.discrete as ds
from skdsp.signal.functions import UnitDelta, UnitStep, UnitRamp, UnitDeltaTrain
from skdsp.signal.util import stem


class Test_Constant(object):
    def test_Constant_constructor(self):
        d = ds.Constant(5)
        assert d is not None

        d = ds.Constant(5 * sp.I, ds.n)
        assert d is not None

        d = ds.Constant(0, sp.Symbol("r", integer=True))
        assert d is not None

        d = ds.Constant(3 + 4 * sp.I, ds.m)
        assert d is not None

        with pytest.raises(ValueError):
            ds.Constant(sp.Symbol("z", real=True))

    def test_Constant_eval(self):
        d = ds.Constant(3)

        assert d[0] == 3
        assert d[1] == 3
        assert d[-1] == 3
        with pytest.raises(ValueError):
            d[0.5]
        with pytest.raises(ValueError):
            d.eval(0.5)

        assert d[0:2] == [3, 3]
        assert d[-1:2] == [3, 3, 3]
        assert d[-4:1:2] == [3, 3, 3]
        assert d[3:-2:-2] == [3, 3, 3]
        with pytest.raises(ValueError):
            d[0:2:0.5]

        assert d.eval(range(0, 2)) == [3, 3]
        assert d.eval(sp.Range(0, 2)) == [3, 3]
        assert d.eval(range(-1, 2)) == [3, 3, 3]
        assert d.eval(sp.Range(-1, 2)) == [3, 3, 3]
        assert d.eval(range(-4, 1, 2)) == [3, 3, 3]
        assert d.eval(sp.Range(-4, 1, 2)) == [3, 3, 3]
        assert d.eval(range(3, -2, -2)) == [3, 3, 3]
        assert d.eval(sp.Range(3, -2, -2)) == [3, 3, 3]

        with pytest.raises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_Constant_iv(self):
        d = ds.Constant(4)
        assert d.is_discrete == True
        assert d.is_continuous == False
        assert d.iv == ds.n
        # shift
        shift = 5
        d = ds.Constant(4).shift(shift)
        assert d.iv == ds.n
        assert d[0] == 4
        assert d[shift] == 4
        d = ds.Constant(4, ds.n).shift(ds.k)
        assert d.iv == ds.n
        assert d([0, 1, 2], 1) == [4, 4, 4]
        # flip
        d = ds.Constant(4).flip()
        assert d.iv == ds.n
        assert d[0] == 4
        # shift and flip
        shift = 5
        d = ds.Constant(4).shift(shift).flip()
        assert d[-shift] == 4
        assert d[shift] == 4
        # flip and shift
        shift = 5
        d = ds.Constant(4).flip().shift(shift)
        assert d[-shift] == 4
        assert d[shift] == 4

    def test_Constant_generator(self):
        d = ds.Constant(123.456)
        with pytest.raises(ValueError):
            dg = d.generate(0, step=0.1)
            next(dg)

        dg = d.generate(start=-3, size=5, overlap=3)
        assert next(dg) == [123.456] * 5
        assert next(dg) == [123.456] * 5
        assert next(dg) == [123.456] * 5

    def test_Constant_misc(self):

        d = ds.Constant(3 + 5 * sp.I)
        assert d.amplitude == 3 + 5 * sp.I
        assert d.real == 3
        assert d.imag == 5
        assert d.is_periodic == True
        assert d.period == 1
        assert d.support == sp.S.Integers
        assert d.duration == None

        f = sp.lambdify(d.iv, d.amplitude)
        assert f(0) == complex(3, 5)
        assert f(1) == complex(3, 5)


class Test_Delta(object):
    def test_Delta_function(self):

        f = UnitDelta
        assert f(0) == 1
        assert f(-1) == 0
        assert f(1) == 0

        # it should but KnoneckerDelta doesn't checks for integer
        # with pytest.raises(ValueError):
        #    f(0.5)

        m = sp.Symbol("m", integer=True, negative=True)
        assert f(m) == 0
        m = sp.Symbol("m", integer=True, positive=True)
        assert f(m) == 0

        g = f(ds.n).rewrite(UnitStep)
        assert g == UnitStep(ds.n) - UnitStep(ds.n - 1)

        g = f(ds.n).rewrite(sp.Piecewise)
        assert g == sp.Piecewise((0, sp.Ne(ds.n, 0)), (1, True))

        f = UnitDelta(ds.k)
        assert str(f) == "UnitDelta(k)"
        assert repr(f) == "\u03b4[k]"
        assert sp.sstr(f) == "UnitDelta(k)"

    def test_Delta_constructor(self):
        """ Delta: constructors.
        """
        # delta discreta
        d = ds.Delta()
        assert d is not None
        # delta discreta
        d = ds.Delta(ds.n)
        assert d is not None
        # delta discreta
        # otras variables simbólicas
        d = ds.Delta(sp.Symbol("r", integer=True))
        assert d is not None
        d = ds.Delta(ds.m)
        assert d is not None
        # variables no enteras
        with pytest.raises(ValueError):
            ds.Delta(sp.Symbol("z", real=True))

    def test_Delta_eval(self):
        d = ds.Delta()

        assert d[0] == 1
        assert d[1] == 0
        assert d[-1] == 0
        with pytest.raises(ValueError):
            d[0.5]
        with pytest.raises(ValueError):
            d.eval(0.5)

        assert d[0:2] == [1, 0]
        assert d[-1:2] == [0, 1, 0]
        assert d[-4:1:2] == [0, 0, 1]
        assert d[3:-2:-2] == [0, 0, 0]
        with pytest.raises(ValueError):
            d[0:2:0.5]

        assert d.eval(range(0, 2)) == [1, 0]
        assert d.eval(sp.Range(0, 2)) == [1, 0]
        assert d.eval(range(-1, 2)) == [0, 1, 0]
        assert d.eval(sp.Range(-1, 2)) == [0, 1, 0]
        assert d.eval(range(-4, 1, 2)) == [0, 0, 1]
        assert d.eval(sp.Range(-4, 1, 2)) == [0, 0, 1]
        assert d.eval(range(3, -2, -2)) == [0, 0, 0]
        assert d.eval(sp.Range(3, -2, -2)) == [0, 0, 0]

        with pytest.raises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_Delta_iv(self):
        """ Delta: independent variable.
        """
        # delta discreta
        d = ds.Delta()
        assert d.is_discrete == True
        assert d.is_continuous == False
        assert d.iv == ds.n
        # shift
        shift = 5
        d = ds.Delta().shift(shift)
        assert d.iv == ds.n
        assert d[0] == 0
        assert d[shift] == 1
        d = ds.Delta(ds.n).shift(ds.k)
        assert d.iv == ds.n
        assert d([0, 1, 2], 1) == [0, 1, 0]
        # flip
        d = ds.Delta().flip()
        assert d.iv == ds.n
        assert d[0] == 1
        # shift and flip
        shift = 5
        d = ds.Delta().shift(shift).flip()
        assert d[-shift] == 1
        assert d[shift] == 0
        # flip and shift
        shift = 5
        d = ds.Delta().flip().shift(shift)
        assert d[-shift] == 0
        assert d[shift] == 1

    def test_Delta_generator(self):
        d = ds.Delta()
        with pytest.raises(ValueError):
            dg = d.generate(0, step=0.1)
            next(dg)

        dg = d.generate(start=-3, size=5, overlap=4)
        assert next(dg) == [0, 0, 0, 1, 0]
        assert next(dg) == [0, 0, 1, 0, 0]
        assert next(dg) == [0, 1, 0, 0, 0]

    def test_Delta_misc(self):
        d = ds.Delta()
        assert d.amplitude == UnitDelta(ds.n)
        assert d.real == UnitDelta(ds.n)
        assert d.imag == 0
        assert d.is_periodic == False
        assert d.period == None
        assert d.support == sp.Range(0, 0)
        assert d.duration == 1

        f = sp.lambdify(d.iv, d.amplitude)
        assert f(0) == 1
        assert f(1) == 0
        x = f(np.array([-1, 0, 1]))
        np.testing.assert_array_equal(x, [0, 1, 0])


class Test_Step(object):
    def test_Step_function(self):

        f = UnitStep
        assert f(0) == 1
        assert f(-1) == 0
        assert f(1) == 1

        m = sp.Symbol("m", integer=True, negative=True)
        assert f(m) == 0
        m = sp.Symbol("m", integer=True, positive=True)
        assert f(m) == 1

        # it should but not because of some functions not using integer symbols: periodicity, plot ....
        # with pytest.raises(ValueError):
        #    x = sp.Symbol("x")
        #    f(x)
        # with pytest.raises(ValueError):
        #    f(0.5)

        s = f(ds.n).rewrite(UnitDelta)
        g = s.doit()
        assert g == UnitStep(ds.n).rewrite(sp.Piecewise)

        s = f(ds.n).rewrite(UnitDelta, form="accum")
        g = s.doit()
        assert g == UnitStep(ds.n).rewrite(sp.Piecewise)

        g = f(ds.n).rewrite(sp.Piecewise)
        assert g == sp.Piecewise((1, ds.n >= 0), (0, True))

        g = f(ds.n).rewrite(UnitRamp)
        assert g == UnitRamp(ds.n + 1) - UnitRamp(ds.n)

        f = UnitStep(ds.k)
        assert str(f) == "UnitStep(k)"
        assert repr(f) == "u[k]"
        assert sp.sstr(f) == "UnitStep(k)"
        assert sp.latex(f) == r"u\left[k\right]"

    def test_Step_constructor(self):
        d = ds.Step()
        assert d is not None

        d = ds.Step(ds.n)
        assert d is not None

        d = ds.Step(sp.Symbol("r", integer=True))
        assert d is not None

        d = ds.Step(ds.m)
        assert d is not None

        with pytest.raises(ValueError):
            ds.Delta(sp.Symbol("z", real=True))

    def test_Step_eval(self):
        d = ds.Step()

        assert d[0] == 1
        assert d[1] == 1
        assert d[-1] == 0
        with pytest.raises(ValueError):
            d[0.5]
        with pytest.raises(ValueError):
            d.eval(0.5)

        assert d[0:2] == [1, 1]
        assert d[-1:2] == [0, 1, 1]
        assert d[-4:1:2] == [0, 0, 1]
        assert d[3:-2:-2] == [1, 1, 0]
        with pytest.raises(ValueError):
            d[0:2:0.5]

        assert d.eval(range(0, 2)) == [1, 1]
        assert d.eval(sp.Range(0, 2)) == [1, 1]
        assert d.eval(range(-1, 2)) == [0, 1, 1]
        assert d.eval(sp.Range(-1, 2)) == [0, 1, 1]
        assert d.eval(range(-4, 1, 2)) == [0, 0, 1]
        assert d.eval(sp.Range(-4, 1, 2)) == [0, 0, 1]
        assert d.eval(range(3, -2, -2)) == [1, 1, 0]
        assert d.eval(sp.Range(3, -2, -2)) == [1, 1, 0]

        with pytest.raises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_Step_iv(self):

        d = ds.Step()
        assert d.is_discrete == True
        assert d.is_continuous == False
        assert d.iv == ds.n
        # shift
        shift = 5
        d = ds.Step().shift(shift)
        assert d.iv == ds.n
        assert d[shift - 1] == 0
        assert d[shift] == 1
        d = ds.Step(ds.n).shift(ds.k)
        assert d.iv == ds.n
        assert d([0, 1, 2], 1) == [0, 1, 1]
        # flip
        d = ds.Step().flip()
        assert d.iv == ds.n
        assert d[0] == 1
        assert d[1] == 0
        assert d[-1] == 1
        # shift and flip
        shift = 5
        d = ds.Step().shift(shift).flip()
        assert d[-shift] == 1
        assert d[shift] == 0
        # flip and shift
        shift = 5
        d = ds.Step().flip().shift(shift)
        assert d[-shift] == 1
        assert d[shift] == 1

    def test_Step_generator(self):
        d = ds.Step()
        with pytest.raises(ValueError):
            dg = d.generate(0, step=0.1)
            next(dg)

        dg = d.generate(start=-3, size=5, overlap=4)
        assert next(dg) == [0, 0, 0, 1, 1]
        assert next(dg) == [0, 0, 1, 1, 1]
        assert next(dg) == [0, 1, 1, 1, 1]

    def test_Step_misc(self):
        d = ds.Step()
        assert d.amplitude == UnitStep(ds.n)
        assert d.real == UnitStep(ds.n)
        assert d.imag == 0
        assert d.is_periodic == False
        assert d.period == None
        assert d.support == sp.Range(0, sp.S.Infinity)
        assert d.duration == sp.S.Infinity

        f = sp.lambdify(d.iv, d.amplitude)
        assert f(0) == 1
        assert f(1) == 1
        x = f([-1, 0, 1])
        np.testing.assert_array_equal(x, [0, 1, 1])


class Test_Ramp(object):
    def test_Ramp_function(self):

        f = UnitRamp
        assert f(0) == 0
        assert f(-1) == 0
        assert f(1) == 1
        assert f(10) == 10
        m = sp.Symbol("m", integer=True, negative=True)
        assert f(m) == 0
        m = sp.Symbol("m", integer=True, positive=True)
        assert f(m) == m

        # it should but not because of some functions not using integer symbols: periodicity, plot ....
        # with pytest.raises(ValueError):
        #    x = sp.Symbol("x")
        #    f(x)
        # with pytest.raises(ValueError):
        #    f(0.5)

        g = f(ds.n).rewrite(UnitStep)
        assert g == ds.n * UnitStep(ds.n)

        g = f(ds.n).rewrite(UnitStep, form='accum')
        k = sp.Dummy(integer=True)
        assert g.dummy_eq(sp.Sum(UnitStep(k - 1), (k, sp.S.NegativeInfinity, ds.n)))

        g = f(ds.n).rewrite(sp.Piecewise)
        assert g == sp.Piecewise((ds.n, ds.n >= 0), (0, True))

        g = f(ds.n).rewrite(sp.Max)
        assert g == sp.Max(0, ds.n)

        g = f(ds.m).rewrite(sp.Abs)
        assert sp.S.Half * (ds.n + sp.Abs(ds.n))

        f = UnitRamp(ds.k)
        assert str(f) == "UnitRamp(k)"
        assert repr(f) == "r[k]"
        assert sp.sstr(f) == "UnitRamp(k)"
        assert sp.latex(f) == r"r\left[k\right]"

    def test_Ramp_constructor(self):

        d = ds.Ramp()
        assert d is not None

        d = ds.Ramp(ds.n)
        assert d is not None

        d = ds.Ramp(sp.Symbol("r", integer=True))
        assert d is not None

        d = ds.Ramp(ds.m)
        assert d is not None

        with pytest.raises(ValueError):
            ds.Ramp(sp.Symbol("z", real=True))

    def test_Ramp_eval(self):
        d = ds.Ramp()

        assert d[0] == 0
        assert d[1] == 1
        assert d[-1] == 0
        with pytest.raises(ValueError):
            d[0.5]
        with pytest.raises(ValueError):
            d.eval(0.5)

        assert d[0:3] == [0, 1, 2]
        assert d[-1:3] == [0, 0, 1, 2]
        assert d[-4:1:2] == [0, 0, 0]
        assert d[3:-2:-2] == [3, 1, 0]
        with pytest.raises(ValueError):
            d[0:2:0.5]

        assert d.eval(range(0, 3)) == [0, 1, 2]
        assert d.eval(sp.Range(0, 3)) == [0, 1, 2]
        assert d.eval(range(-1, 3)) == [0, 0, 1, 2]
        assert d.eval(sp.Range(-1, 3)) == [0, 0, 1, 2]
        assert d.eval(range(-4, 1, 2)) == [0, 0, 0]
        assert d.eval(sp.Range(-4, 1, 2)) == [0, 0, 0]
        assert d.eval(range(3, -2, -2)) == [3, 1, 0]
        assert d.eval(sp.Range(3, -2, -2)) == [3, 1, 0]

        with pytest.raises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_Ramp_iv(self):

        d = ds.Ramp()
        assert d.is_discrete == True
        assert d.is_continuous == False
        assert d.iv == ds.n
        # shift
        shift = 5
        d = ds.Ramp().shift(shift)
        assert d.iv == ds.n
        assert d[0] == 0
        assert d[shift] == 0
        d = ds.Ramp(ds.n).shift(ds.k)
        assert d.iv == ds.n
        assert d([0, 1, 2], 1) == [0, 0, 1]
        # flip
        d = ds.Ramp().flip()
        assert d.iv == ds.n
        assert d[0] == 0
        assert d[-5] == 5
        # shift and flip
        shift = 5
        d = ds.Ramp().shift(shift).flip()
        assert d[-shift] == 0
        assert d[shift] == 0
        # flip and shift
        shift = 5
        d = ds.Ramp().flip().shift(shift)
        assert d[-shift] == 10
        assert d[shift] == 0

    def test_Ramp_generator(self):
        d = ds.Ramp()
        with pytest.raises(ValueError):
            dg = d.generate(0, step=0.1)
            next(dg)

        dg = d.generate(start=-3, size=5, overlap=4)
        assert next(dg) == [0, 0, 0, 0, 1]
        assert next(dg) == [0, 0, 0, 1, 2]
        assert next(dg) == [0, 0, 1, 2, 3]

    def test_Ramp_misc(self):
        d = ds.Ramp()
        assert d.amplitude == UnitRamp(ds.n)
        assert d.real == UnitRamp(ds.n)
        assert d.imag == 0
        assert d.is_periodic == False
        assert d.period == None
        assert d.support == sp.Range(1, sp.S.Infinity)
        assert d.duration == sp.S.Infinity

        f = sp.lambdify(d.iv, d.amplitude)
        assert f(0) == 0
        assert f(1) == 1
        assert f(2) == 2
        x = f(np.array([-1, 0, 1, 2]))
        np.testing.assert_array_equal(x, [0, 0, 1, 2])


class Test_DeltaTrain(object):
    def test_DeltaTrain_function(self):

        f = UnitDeltaTrain
        assert f(0, 3) == 1
        assert f(-1, 3) == 0
        assert f(1, 3) == 0

        # it should but not because of some functions not using integer symbols: periodicity, plot ....
        # with pytest.raises(ValueError):
        #    x = sp.Symbol("x")
        #    f(x)
        # with pytest.raises(ValueError):
        #    f(0.5)

        m = sp.Symbol("m", integer=True)
        assert f(m, 3) == UnitDeltaTrain(m, 3)

        g = f(ds.n, 3).rewrite(sp.Piecewise)
        assert g == sp.Piecewise((1, sp.Eq(sp.Mod(ds.n, 3), 0)), (0, True))

        f = UnitDeltaTrain(ds.k, 3)
        assert str(f) == "UnitDeltaTrain(k, 3)"
        assert repr(f) == "\u0428[((k))3]"
        assert sp.sstr(f) == "UnitDeltaTrain(k, 3)"
        assert (
            sp.latex(f)
            == r"{\rotatebox[origin=c]{180}{$\Pi\kern-0.361em\Pi$}\left[((k))_{3}\right]"
        )

    def test_DeltaTrain_constructor(self):

        with pytest.raises(ValueError):
            d = ds.DeltaTrain()

        with pytest.raises(ValueError):
            d = ds.DeltaTrain(ds.n)

        d = ds.DeltaTrain(N=16)
        assert d is not None

        d = ds.DeltaTrain(sp.Symbol("r", integer=True), N=12)
        assert d is not None

        d = ds.DeltaTrain(ds.m, N=sp.Symbol("N", integer=True, positive=True))
        assert d is not None

        with pytest.raises(ValueError):
            d = ds.DeltaTrain(ds.m, N=sp.Symbol("N"))

    def test_DeltaTrain_iv(self):

        d = ds.DeltaTrain(N=3)
        assert d.is_discrete == True
        assert d.is_continuous == False
        assert d.iv == ds.n

        # shift
        shift = 5
        d = ds.DeltaTrain(N=3).shift(shift)
        assert d.iv == ds.n
        assert d[-1] == 1
        assert d[0] == 0
        assert d[1] == 0

        d = ds.DeltaTrain(N=3).shift(ds.k)
        assert d.iv == ds.n
        assert d([0, 1, 2], 1) == [0, 1, 0]
        assert d([0, 1, 2], 2) == [0, 0, 1]

        # flip
        d = ds.DeltaTrain(N=3).flip()
        assert d.iv == ds.n
        assert d([0, 1, 2]) == [1, 0, 0]

        # shift and flip
        shift = 5
        d = ds.DeltaTrain(N=3).shift(shift).flip()
        assert d([0, 1, 2]) == [0, 1, 0]

        # flip and shift
        shift = 5
        d = ds.DeltaTrain(N=3).flip().shift(shift)
        assert d([0, 1, 2]) == [0, 0, 1]

    def test_DeltaTrain_eval(self):

        d = ds.DeltaTrain(N=16)
        assert d.eval(0) == 1
        assert d.eval(1) == 0
        assert d.eval(-1) == 0
        with pytest.raises(ValueError):
            d.eval(0.5)

        assert d.eval(range(0, 2)) == [1, 0]
        assert d.eval(range(-1, 2)) == [0, 1, 0]
        assert d.eval(range(-4, 1, 2)) == [0, 0, 1]
        assert d.eval(range(3, -2, -2)) == [0, 0, 0]

        assert d[0] == 1
        assert d[1] == 0
        assert d[-1] == 0
        with pytest.raises(ValueError):
            d[0.5]

        assert d[0:2] == [1, 0]
        assert d[-1:2] == [0, 1, 0]
        assert d[-4:1:2] == [0, 0, 1]
        assert d[3:-2:-2] == [0, 0, 0]
        with pytest.raises(ValueError):
            d[0:2:0.5]

        d = ds.DeltaTrain(N=16).shift(1)
        assert d.eval(0) == 0
        assert d.eval(1) == 1
        assert d.eval(-1) == 0
        with pytest.raises(ValueError):
            d.eval(0.5)

        assert d.eval(range(0, 2)) == [0, 1]
        assert d.eval(range(-1, 2)) == [0, 0, 1]
        assert d.eval(range(-4, 1, 2)) == [0, 0, 0]
        assert d.eval(range(3, -2, -2)) == [0, 1, 0]
        with pytest.raises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

        assert d[0] == 0
        assert d[1] == 1
        assert d[-1] == 0
        with pytest.raises(ValueError):
            d[0.5]

        assert d[0:2] == [0, 1]
        assert d[-1:2] == [0, 0, 1]
        assert d[-4:1:2] == [0, 0, 0]
        assert d[3:-2:-2] == [0, 1, 0]
        with pytest.raises(ValueError):
            d[0:2:0.5]

    def test_DeltaTrain_generator(self):

        d = ds.DeltaTrain(N=5)
        with pytest.raises(ValueError):
            dg = d.generate(0, step=0.1)
            next(dg)

        dg = d.generate(start=-3, size=10, overlap=4)
        assert next(dg) == [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        assert next(dg) == [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        assert next(dg) == [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]

    def test_DeltaTrain_misc(self):

        N = 5
        d = ds.DeltaTrain(N=N)
        assert d.amplitude == UnitDeltaTrain(ds.n, N)
        assert d.real == UnitDeltaTrain(ds.n, N)
        assert d.imag == 0
        assert d.is_periodic == True
        assert d.period == N
        assert d.support == sp.S.Integers
        assert d.duration == None

        f = sp.lambdify(d.iv, d.amplitude)
        assert f(0) == 1
        assert f(1) == 0
        x = f(np.array([-1, 0, 1]))
        np.testing.assert_array_equal(x, [0, 1, 0])


class Test_Data(object):
    def test_Data_constructor(self):
        from numpy import array

        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        a = sp.Symbol("a", integer=True)
        b = sp.Symbol("b", real=True)

        s = ds.Data([1, 2, 3], start=3, iv=n)
        assert s.amplitude == (
            UnitDelta(n - 3) + 2 * UnitDelta(n - 4) + 3 * UnitDelta(n - 5)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data([1, -1], iv=n)
        assert s.amplitude == (UnitDelta(n) - UnitDelta(n - 1))
        assert s.codomain == sp.S.Reals

        s = ds.Data([1, 2, 3], start=3, iv=n)
        assert s.amplitude == (
            UnitDelta(n - 3) + 2 * UnitDelta(n - 4) + 3 * UnitDelta(n - 5)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data((1, 2, 3 * sp.I), iv=n)
        assert s.amplitude == (
            UnitDelta(n) + 2 * UnitDelta(n - 1) + 3 * sp.I * UnitDelta(n - 2)
        )
        assert s.codomain == sp.S.Complexes

        s = ds.Data(array([1, 2, 3]), iv=n)
        assert s.amplitude == (
            UnitDelta(n) + 2 * UnitDelta(n - 1) + 3 * UnitDelta(n - 2)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data({1: 4, 0: 3 * sp.I, -2: "cos(w*k)"}, iv=n, codomain=sp.S.Reals)
        assert s.amplitude == (
            sp.cos(sp.Symbol("w") * sp.Symbol("k")) * UnitDelta(n + 2)
            + 3 * sp.I * UnitDelta(n)
            + 4 * UnitDelta(n - 1)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data({7: a + b, 8: a * b, 9: 3}, iv=n)
        assert s.amplitude == (
            (a + b) * UnitDelta(n - 7)
            + (a * b) * UnitDelta(n - 8)
            + 3 * UnitDelta(n - 9)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data([[1, 2], [3, 4]], iv=n)
        assert s.amplitude == (
            UnitDelta(n)
            + 2 * UnitDelta(n - 1)
            + 3 * UnitDelta(n - 2)
            + 4 * UnitDelta(n - 3)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data([1, 2, "3"], iv=n)
        assert s.amplitude == (
            UnitDelta(n) + 2 * UnitDelta(n - 1) + 3 * UnitDelta(n - 2)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data(("1",), iv=n)
        assert s == ds.Delta(n)
        assert s.codomain == sp.S.Reals

        s = ds.Data([0] * 3 + [1, 2, 3] + [0] * 5, iv=n, start=-3)
        assert s.amplitude == (
            UnitDelta(n) + 2 * UnitDelta(n - 1) + 3 * UnitDelta(n - 2)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data([0] * 5 + [1, 2, 3] + [0] * 5, -3, iv=n)
        assert s.amplitude == (
            UnitDelta(n - 2) + 2 * UnitDelta(n - 3) + 3 * UnitDelta(n - 4)
        )
        assert s.codomain == sp.S.Reals

        s = ds.Data([0] * 5 + [sp.I, 2, 3] + [0] * 5, 3, iv=n)
        assert s.amplitude == (
            sp.I * UnitDelta(n - 8) + 2 * UnitDelta(n - 9) + 3 * UnitDelta(n - 10)
        )
        assert s.codomain == sp.S.Complexes

        with pytest.raises(ValueError):
            s = ds.Data([1, 2, 3], iv=t)

        with pytest.raises(TypeError):
            s = ds.Data({"1": 4, "c": 3, -2: 27}, iv=n)

    def test_Data_periodic(self):
        s = ds.Data([1, 2, 3], periodic=True, iv=ds.n)
        assert s.is_periodic == True
        assert s.period == 3
        assert s[0:3] == [1, 2, 3]
        assert s[3:6] == [1, 2, 3]
        assert s[-3:0] == [1, 2, 3]

        s = ds.Data([1, 2, 3, 4], start=10, periodic=True, iv=ds.n)
        assert s.is_periodic == True
        assert s.period == 4
        assert s[0:3] == [3, 4, 1]
        assert s.eval(0) == 3
        assert s.eval(-1) == 2
        assert s.eval(-2) == 1
        assert s.eval(range(-4, 6)) == [3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

        s = ds.Data([1, 2, 3], start=10, periodic=False, iv=ds.n)
        assert s.is_periodic == False
        assert s.period == None

        s = ds.Data([0, 1, 1, 0, 0, 0], periodic=True)
        assert s.is_periodic == True
        assert s.period == 6
        assert s[0:6] == [0, 1, 1, 0, 0, 0]
        assert s[3:9] == [0, 0, 0, 0, 1, 1]
        assert s[-3:3] == [0, 0, 0, 0, 1, 1]

        s = ds.DiscreteSignal(
            UnitDelta(sp.Mod(ds.n, 6) - 1) + UnitDelta(sp.Mod(ds.n, 6) - 2),
            ds.n,
            6,
            sp.S.Integers,
            sp.S.Reals,
        )
        assert s.is_periodic == True
        assert s.period == 6
        assert s[0:6] == [0, 1, 1, 0, 0, 0]
        assert s[3:9] == [0, 0, 0, 0, 1, 1]
        assert s[-3:3] == [0, 0, 0, 0, 1, 1]

        s0 = ds.Delta().shift(1) + ds.Delta().shift(2)
        s = ds.DiscreteSignal.from_period(s0.amplitude, ds.n, 6)
        assert s.is_periodic == True
        assert s.period == 6
        assert s[0:6] == [0, 1, 1, 0, 0, 0]
        assert s[3:9] == [0, 0, 0, 0, 1, 1]
        assert s[-3:3] == [0, 0, 0, 0, 1, 1]

    def test_Data_eval(self):
        n = sp.Symbol("n", integer=True)
        a = sp.Symbol("a", integer=True)
        b = sp.Symbol("b", real=True)

        s = ds.Data([1, 2, 3 * sp.I], 3, iv=n)
        assert s.eval(2) == 0
        assert s(3) == 1
        assert s(5) == 3 * sp.I
        assert s.eval(-1) == 0
        assert s[-1] == 0
        assert s.eval(range(-1, 2)) == [0] * 3
        assert s((3, 5, 4)) == [1, 3 * sp.I, 2]
        assert s[3:5] == [1, 2]

        s = ds.Data((1, 2, 3), iv=n)
        assert s.eval(2) == 3
        assert s(3) == 0
        assert s(1) == 2
        assert s.eval(-1) == 0
        assert s[-1] == 0
        assert s.eval(range(-1, 2)) == [0, 1, 2]
        assert s((2, 3)) == [3, 0]
        assert s[1:4] == [2, 3, 0]

        s = ds.Data(np.array([1, 2, 3 * sp.I]), iv=n)
        assert s.eval(2) == 3 * sp.I
        assert s(3) == 0
        assert s(1) == 2
        assert s.eval(-1) == 0
        assert s[-1] == 0
        assert s.eval(range(-1, 2)) == [0, 1, 2]
        assert s((2, 3)) == [3 * sp.I, 0]
        assert s[1:4] == [2, 3 * sp.I, 0]

        s = ds.Data({1: 4, 0: 3, -2: "cos(w*k)"}, iv=n)
        assert s.eval(-2) == sp.sympify("cos(w*k)")
        assert s(0) == 3
        assert s(1) == 4
        assert s.eval(-2, {list(s.free_symbols)[0]: 2, list(s.free_symbols)[1]: 3})
        assert s[-1] == 0
        assert s[-2, 2, 3] == sp.cos(6)
        assert s.eval(range(-1, 2)) == [0, 3, 4]
        assert s(range(-1, 2)) == [0, 3, 4]
        assert s[-2:2, 2, 3] == [sp.cos(6), 0, 3, 4]

        s = ds.Data({7: a + b, 8: a * b, 9: 3}, iv=n)
        assert s.eval(7) == a + b
        assert s.eval(8) == a * b
        assert s(9) == 3
        assert s(1001) == 0
        assert s[1001] == 0
        assert s[7, 1, 2] == 3
        assert s[8, 3, 2] == 6
        assert s.eval(range(7, 9)) == [a + b, a * b]
        assert s[9:11] == [3, 0]
        assert s[7:9, 1, 2] == [3, 2]

        s = ds.Data([[1, 2 * sp.I], [3, 4]], iv=n)
        assert s.eval(1) == 2 * sp.I
        assert s(3) == 4
        assert s(0) == 1
        assert s.eval(-1) == 0
        assert s[-1] == 0
        assert s.eval(range(-1, 2)) == [0, 1, 2 * sp.I]
        assert s[-1:2] == [0, 1, 2 * sp.I]

        s = ds.Data([1, 2, "3"], iv=n)
        assert s.eval(1) == 2
        assert s(3) == 0
        assert s(2) == 3
        assert s.eval(-1) == 0
        assert s[-1] == 0
        assert s.eval(range(-1, 2)) == [0, 1, 2]
        assert s[-1:2] == [0, 1, 2]

        s = ds.Data(("1",), iv=n)
        assert s.eval(1) == 0
        assert s(0) == 1
        assert s.eval(-1) == 0
        assert s[-1] == 0
        assert s.eval((1, -1)) == [0, 0]
        assert s[-1:2] == [0, 1, 0]

    def test_Delta_generator(self):
        n = sp.Symbol("n", integer=True)
        a = sp.Symbol("a", integer=True)
        b = sp.Symbol("b", real=True)

        d = ds.Data([1, 2, 3], periodic=True, iv=n)
        with pytest.raises(ValueError):
            dg = d.generate(0, step=0.1)
            next(dg)

        dg = d.generate(start=-3, size=5, overlap=3)
        assert next(dg) == [1, 2, 3, 1, 2]
        assert next(dg) == [3, 1, 2, 3, 1]
        assert next(dg) == [2, 3, 1, 2, 3]

        d = ds.Data([a, b, a * b, a + b], iv=n)
        dg = d.generate(start=-3, size=5, overlap=3)
        assert next(dg) == [0, 0, 0, a, b]
        assert next(dg) == [0, a, b, a * b, a + b]
        assert next(dg) == [b, a * b, a + b, 0, 0]

    def test_Data_misc(self):

        s = ds.Data([1, 2, 3], periodic=True, iv=ds.n)
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = ds.Data([1, 2, 3, 4], start=10, periodic=True, iv=ds.n)
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = ds.Data([1, 2, 3], start=10, periodic=False, iv=ds.n)
        assert s.support == sp.Range(10, 12)
        assert s.duration == 13

        s = ds.Data([1, 2, 3], periodic=True, iv=ds.n)
        f = sp.lambdify(s.iv, s.amplitude)
        assert f(0) == 1
        assert f(1) == 2
        assert f(-1) == 3
        x = f(np.array([-1, 0, 1]))
        np.testing.assert_array_equal(x, [3, 1, 2])


class Test_Sinusoid(object):
    def test_Sinusoid_constructor(self):

        with pytest.raises(ValueError):
            ds.Sinusoid()

        s = ds.Sinusoid(1, sp.S.Pi * sp.Rational(1, 4), sp.S.Pi * sp.Rational(3, 8))
        assert s is not None

        s = ds.Sinusoid(
            A=1, omega=sp.S.Pi * sp.Rational(1, 4), phi=sp.S.Pi * sp.Rational(3, 8)
        )
        assert s is not None

        s = ds.Sinusoid(omega=1, iv=sp.Symbol("r", integer=True))
        assert s is not None

        s = ds.Sinusoid(omega=1, iv=ds.m)
        assert s is not None

        with pytest.raises(ValueError):
            ds.Sinusoid(sp.Symbol("x", complex=True), 1)

        with pytest.raises(ValueError):
            ds.Sinusoid(omega=1, iv=sp.Symbol("z", real=True))

    def test_Sinusoid_eval(self):

        s = ds.Sinusoid(1, sp.S.Pi / 4, sp.S.Pi / 6)
        assert s.eval(0) == sp.sqrt(3) / 2
        assert s.eval(1) == (-sp.sqrt(2) + sp.sqrt(6)) / 4
        assert s.eval(2) == -1 / 2
        assert s.eval(-1) == (sp.sqrt(2) + sp.sqrt(6)) / 4
        assert s(0) == sp.sqrt(3) / 2
        assert s(1) == (-sp.sqrt(2) + sp.sqrt(6)) / 4
        assert s(2) == -1 / 2
        assert s(-1) == (sp.sqrt(2) + sp.sqrt(6)) / 4
        assert s[0] == sp.sqrt(3) / 2
        assert s[1] == (-sp.sqrt(2) + sp.sqrt(6)) / 4
        assert s[2] == -1 / 2
        assert s[-1] == (sp.sqrt(2) + sp.sqrt(6)) / 4

        s = ds.Sinusoid(1, 0, sp.S.Pi / 2)
        assert s.eval(0) == 0
        assert s(0) == 0
        assert s[0] == 0

        x = sp.Symbol("x", real=True)
        s = ds.Sinusoid(x, sp.S.Pi / 4)
        assert s.eval(0, {x: 3}) == 3
        assert s(0, 3) == 3
        assert s[0] == x

        with pytest.raises(ValueError):
            s(0, sp.I)

        with pytest.raises(ValueError):
            s.eval(0.5)

        with pytest.raises(ValueError):
            s[2.25]

        s = ds.Sinusoid(1, sp.S.Pi / 4, sp.S.Pi / 6)
        expected = [sp.sqrt(3) / 2, (-sp.sqrt(2) + sp.sqrt(6)) / 4, -1 / 2]
        assert expected == s.eval(np.arange(0, 3))
        assert expected == s[0:3]
        expected = [
            (sp.sqrt(2) + sp.sqrt(6)) / 4,
            sp.sqrt(3) / 2,
            (-sp.sqrt(2) + sp.sqrt(6)) / 4,
        ]
        assert expected == s.eval(np.arange(-1, 2))
        assert expected == s[-1:2]

        expected = [-sp.sqrt(3) / 2, 1 / 2, sp.sqrt(3) / 2]
        assert expected == s.eval(np.arange(-4, 1, 2))
        assert expected == s[-4:1:2]

        expected = [
            (-sp.sqrt(6) - sp.sqrt(2)) / 4,
            (-sp.sqrt(2) + sp.sqrt(6)) / 4,
            (sp.sqrt(2) + sp.sqrt(6)) / 4,
        ]
        assert expected == s.eval(np.arange(3, -2, -2))
        assert expected == s[3:-2:-2]

        A, omega, phi = sp.symbols("A omega phi", real=True)
        s = ds.Sinusoid(A, omega, phi)

    def test_Sinusoid_iv(self):

        s = ds.Sinusoid(omega=1)
        assert s.is_discrete == True
        assert s.is_continuous == False
        assert s.iv == ds.n

        s = ds.Sinusoid(omega=sp.S.Pi / 4)

        with pytest.raises(ValueError):
            s.shift(0.5)

        with pytest.raises(ValueError):
            s.delay(0.5)

        d = s.shift(1)
        assert d([-1, 0, 1]) == [0, sp.sqrt(2) / 2, 1]

        d = s.flip()
        assert d([-1, 0, 1]) == [sp.sqrt(2) / 2, 1, sp.sqrt(2) / 2]

        d = s.shift(1).flip()
        assert d([-1, 0, 1]) == [1, sp.sqrt(2) / 2, 0]

        d = s.flip().shift(1)
        assert d([-1, 0, 1]) == [0, sp.sqrt(2) / 2, 1]

    def test_Sinusoid_generator(self):

        s = ds.Sinusoid(omega=sp.S.Pi / 4)

        with pytest.raises(ValueError):
            dg = s.generate(0, step=0.1)
            next(dg)

        dg = s.generate(start=-3, size=3, overlap=1)
        assert next(dg) == [-sp.sqrt(2) / 2, 0, sp.sqrt(2) / 2]
        assert next(dg) == [sp.sqrt(2) / 2, 1, sp.sqrt(2) / 2]
        assert next(dg) == [sp.sqrt(2) / 2, 0, -sp.sqrt(2) / 2]

    def test_Sinusoid_frequency(self):

        d = ds.Sinusoid(2, sp.S.Pi / 6, 3 * sp.S.Pi / 5)
        assert d.is_periodic == True
        assert d.period == 12
        assert d.frequency == sp.S.Pi / 6
        assert d.reduced_frequency(True) == sp.S.Pi / 6
        assert d.reduced_frequency(False) == sp.S.Pi / 6
        assert d.phase == 3 * sp.S.Pi / 5
        assert d.reduced_phase(True) == 3 * sp.S.Pi / 5
        assert d.reduced_phase(False) == 3 * sp.S.Pi / 5
        assert d.alias(0) == d
        assert d.alias(1) == ds.Sinusoid(2, 13 * sp.S.Pi / 6, 3 * sp.S.Pi / 5)
        assert d.alias(-1) == ds.Sinusoid(2, -11 * sp.S.Pi / 6, 3 * sp.S.Pi / 5)

        x = sp.Symbol("x", real=True)
        d = ds.Sinusoid(x, 9 * sp.S.Pi / 4, 100 * sp.S.Pi / 12)
        assert d.is_periodic == True
        assert d.period == 8
        assert d.frequency == 9 * sp.S.Pi / 4
        assert d.reduced_frequency(True) == sp.S.Pi / 4
        assert d.reduced_frequency(False) == sp.S.Pi / 4
        assert d.phase == 100 * sp.S.Pi / 12
        assert d.reduced_phase(True) == sp.S.Pi / 3
        assert d.reduced_phase(False) == sp.S.Pi / 3
        assert d.alias(0) == ds.Sinusoid(x, sp.S.Pi / 4, sp.S.Pi / 3)
        assert d.alias(1) == ds.Sinusoid(x, 9 * sp.S.Pi / 4, sp.S.Pi / 3)
        assert d.alias(-1) == ds.Sinusoid(x, -7 * sp.S.Pi / 4, sp.S.Pi / 3)

        d = ds.Sinusoid(omega=3 * sp.S.Pi, phi=-5 * sp.S.Pi / 2)
        assert d.is_periodic == True
        assert d.period == 2
        assert d.frequency == 3 * sp.S.Pi
        assert d.reduced_frequency(True) == sp.S.Pi
        assert d.reduced_frequency(False) == -sp.S.Pi
        assert d.phase == -5 * sp.S.Pi / 2
        assert d.reduced_phase(True) == 3 * sp.S.Pi / 2
        assert d.reduced_phase(False) == -sp.S.Pi / 2
        assert d.alias(0) == d
        assert d.alias(1) == ds.Sinusoid(1, sp.S.Pi, -sp.S.Pi / 2)
        assert d.alias(-1) == ds.Sinusoid(1, -sp.S.Pi, -sp.S.Pi / 2)

        s = ds.Sinusoid(omega=1)
        assert s.is_periodic == False
        assert s.frequency == 1
        assert s.phase == 0

        s = ds.Sinusoid(1, sp.S.Pi / 4)
        assert s.period == 8
        assert s.frequency == sp.S.Pi / 4
        assert s.phase == 0

        s = ds.Sinusoid(1, 3 * sp.S.Pi / 8)
        assert s.period == 16
        assert s.frequency == 3 * sp.S.Pi / 8
        assert s.phase == 0

        s = ds.Sinusoid(1, -3 * sp.S.Pi / 8)
        assert s.period == 16
        assert s.frequency == -3 * sp.S.Pi / 8
        assert s.phase == 0

        s = ds.Sinusoid(1, 0.83 * sp.S.Pi)
        assert s.period == 200
        assert s.frequency == 0.83 * sp.S.Pi
        assert s.phase == 0

        s = ds.Sinusoid(1, 3 / 8)
        assert s.is_periodic == False
        assert s.frequency == 3 / 8
        assert s.phase == 0

        s = ds.Sinusoid(1, 1 / 4)
        assert s.is_periodic == False
        assert s.frequency == 1 / 4
        assert s.phase == 0

        s = ds.Sinusoid(1, -7 / 2)
        assert s.is_periodic == False
        assert s.frequency == -7 / 2
        assert s.reduced_frequency(False) == -7 / 2 + 2 * sp.S.Pi
        assert s.reduced_frequency(True) == -7 / 2 + 2 * sp.S.Pi
        assert s.phase == 0

        s = ds.Sinusoid(1, sp.sqrt(2) * sp.S.Pi)
        assert s.is_periodic == False
        assert s.frequency == sp.sqrt(2) * sp.S.Pi
        assert s.phase == 0
        assert s.alias(0) == ds.Sinusoid(1, -2 * sp.S.Pi + sp.sqrt(2) * sp.S.Pi)

        s = ds.Sinusoid(1, sp.S.Pi ** 2)
        assert s.is_periodic == False
        assert s.frequency == sp.S.Pi ** 2
        assert s.phase == 0
        assert s.alias(0) == ds.Sinusoid(1, -4 * sp.S.Pi + sp.S.Pi ** 2)

    def test_Sinusoid_misc(self):

        d = ds.Sinusoid(2, sp.S.Pi / 6, 3 * sp.S.Pi / 5)
        assert d.amplitude == 2 * sp.cos(sp.S.Pi * ds.n / 6 + 3 * sp.S.Pi / 5)
        assert d.real == 2 * sp.cos(sp.S.Pi * ds.n / 6 + 3 * sp.S.Pi / 5)
        assert d.imag == 0
        assert d.support == sp.S.Integers
        assert d.duration == None
        assert d.gain == 2
        assert (
            sp.simplify(
                d.in_phase - (1 / 2 - sp.sqrt(5) / 2) * sp.cos(sp.S.Pi * ds.n / 6)
            )
            == sp.S.Zero
        )
        assert (
            sp.simplify(
                d.in_quadrature
                - (-sp.sqrt(2 * sp.sqrt(5) + 10) * sp.sin(sp.S.Pi * ds.n / 6) / 2)
            )
            == sp.S.Zero
        )
        assert (
            sp.simplify(d.I - (1 / 2 - sp.sqrt(5) / 2) * sp.cos(sp.S.Pi * ds.n / 6))
            == sp.S.Zero
        )
        assert (
            sp.simplify(
                d.Q - (-sp.sqrt(2 * sp.sqrt(5) + 10) * sp.sin(sp.S.Pi * ds.n / 6) / 2)
            )
            == sp.S.Zero
        )

        x = sp.Symbol("x", real=True)
        d = ds.Sinusoid(x, sp.S.Pi / 4, sp.S.Pi / 12)
        assert d.amplitude == x * sp.cos(sp.S.Pi * ds.n / 4 + sp.S.Pi / 12)
        assert d.real == x * sp.cos(sp.S.Pi * ds.n / 4 + sp.S.Pi / 12)
        assert d.imag == 0
        assert d.support == sp.S.Integers
        assert d.duration == None
        assert d.gain == x
        assert d.I == (
            x * (sp.sqrt(2) / 4 + sp.sqrt(6) / 4) * sp.cos(sp.S.Pi * ds.n / 4)
        )
        assert d.Q == (
            -x * (-sp.sqrt(2) / 4 + sp.sqrt(6) / 4) * sp.sin(sp.S.Pi * ds.n / 4)
        )

        d = ds.Sinusoid(omega=3 * sp.S.Pi)
        assert d.amplitude == (-1) ** ds.n
        assert d.real == (-1) ** ds.n
        assert d.imag == 0
        assert d.support == sp.S.Integers
        assert d.duration == None
        assert d.gain == 1
        assert d.I == (-1) ** ds.n
        assert d.Q == 0

        d = ds.Sinusoid(2, sp.S.Pi / 4)
        f = sp.lambdify(d.iv, d.amplitude)
        np.testing.assert_almost_equal(f(-1), 2.0 * np.cos(np.pi / 4))
        np.testing.assert_almost_equal(f(0), 2.0)
        np.testing.assert_almost_equal(f(1), 2.0 * np.cos(np.pi / 4))
        x = f(np.array([-1, 0, 1]))
        np.testing.assert_array_almost_equal(
            x, 2.0 * np.cos(np.pi * np.arange(-1, 2) / 4)
        )

        s = ds.Sinusoid(omega=sp.S.Pi / 4, phi=sp.S.Pi / 3)
        assert (
            sp.simplify(
                s.euler
                - sp.S(
                    1 / 2 * sp.exp(-sp.I * (sp.S.Pi * ds.n / 4 + sp.S.Pi / 3))
                    + 1 / 2 * sp.exp(sp.I * (sp.S.Pi * ds.n / 4 + sp.S.Pi / 3))
                )
            )
            == 0
        )


class Test_Exponential(object):
    def test_Exponential_constructor(self):
        s = ds.Exponential(alpha=0.5)
        assert s is not None

        s = ds.Exponential(alpha=-1)
        assert s is not None

        s = ds.Exponential(alpha=1)
        assert s is not None

        s = ds.Exponential(3, sp.Rational(1, 2))
        assert s is not None

        s = ds.Exponential(3, 1 + sp.I)
        assert s is not None

        s = ds.Exponential(2 * sp.exp(sp.I * sp.S.Pi / 4), sp.Rational(1, 2))
        assert s is not None

        s = ds.Exponential(1 - sp.I, 1 + sp.I)
        assert s is not None

        s = ds.Exponential(-2 * sp.exp(sp.I * sp.S.Pi / 10), 3 * sp.exp(sp.I * sp.S.Pi / 4))
        assert s is not None

        with pytest.raises(ValueError):
            ds.Exponential()

        with pytest.raises(ValueError):
            ds.Exponential(C=ds.n)

        with pytest.raises(ValueError):
            ds.Exponential(alpha=ds.n)

    def test_Exponential_misc(self):
        s = ds.Exponential(alpha=0.5)
        assert s.amplitude == sp.Pow(0.5, ds.n)

        s = ds.Exponential(alpha=-1)
        assert s.amplitude == sp.Pow(-1, ds.n)

        s = ds.Exponential(2, sp.Rational(1, 2))
        assert sp.simplify(s.amplitude - 2*sp.Pow(sp.Rational(1, 2), ds.n)) == 0

        s = ds.Exponential(-2 * sp.exp(sp.I * sp.S.Pi / 10), 3 * sp.exp(sp.I * sp.S.Pi / 4))
        assert sp.simplify(s.amplitude - (sp.Pow(3, ds.n) * sp.exp(sp.I * sp.S.Pi * ds.n / 4)) * (-2 * sp.exp(sp.I * sp.S.Pi / 10))) == 0


class Test_Arithmetic(object):
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

    def test_Discrete_misc(self):
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

