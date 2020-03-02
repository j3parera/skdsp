import numpy as np
import pytest
import sympy as sp

import skdsp.signal.discrete as ds
from skdsp.signal.functions import UnitDelta, UnitStep, UnitRamp, UnitDeltaTrain
from skdsp.util.util import stem


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

        d = ds.Constant(sp.Symbol("z", real=True))
        assert d is not None

        with pytest.raises(ValueError):
            ds.Constant(ds.n ** 2)

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

        z = sp.Symbol("z", real=True)
        d = ds.Constant(z)
        assert d[0] == z
        assert d[1] == z
        assert d[-1] == z
        with pytest.raises(ValueError):
            d[0.5]
        with pytest.raises(ValueError):
            d.eval(0.5)

        assert d[0:2] == [z, z]
        assert d[-1:2] == [z, z, z]
        assert d[-4:1:2] == [z, z, z]
        assert d[3:-2:-2] == [z, z, z]
        with pytest.raises(ValueError):
            d[0:2:0.5]

        assert d.eval(range(0, 2)) == [z, z]
        assert d.eval(sp.Range(0, 2)) == [z, z]
        assert d.eval(range(-1, 2)) == [z, z, z]
        assert d.eval(sp.Range(-1, 2)) == [z, z, z]
        assert d.eval(range(-4, 1, 2)) == [z, z, z]
        assert d.eval(sp.Range(-4, 1, 2)) == [z, z, z]
        assert d.eval(range(3, -2, -2)) == [z, z, z]
        assert d.eval(sp.Range(3, -2, -2)) == [z, z, z]

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
        assert d.real_part == 3
        assert d.real == ds.Constant(3)
        assert d.imag_part == 5
        assert d.imag == ds.Constant(5)
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
        assert d.real_part == UnitDelta(ds.n)
        assert d.real == ds.Delta(ds.n)
        assert d.imag_part == 0
        assert d.imag == ds.Constant(0)
        assert d.is_periodic == False
        assert d.period == None
        assert d.support == sp.Range(0, 1)
        assert d.duration == 1

        N = sp.Symbol("N", integer=True, positive=True)
        assert d.energy(2) == 1
        assert d.energy(N) == 1
        assert d.energy() == 1
        assert d.is_energy == True
        assert d.mean_power(2) == sp.Rational(1, 5)
        assert d.mean_power(N) == sp.S.One / (2 * N + 1)
        assert d.mean_power() == 0
        assert d.is_power == False

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
        assert d.real_part == UnitStep(ds.n)
        assert d.imag_part == 0
        assert d.real == ds.Step()
        assert d.imag == ds.Constant(0)
        assert d.is_periodic == False
        assert d.period == None
        assert d.support == sp.Range(0, sp.S.Infinity)
        assert d.duration == sp.S.Infinity

        N = sp.Symbol("N", integer=True, positive=True)
        assert d.energy(2) == 3
        assert d.energy(N) == N + 1
        assert d.energy() == sp.S.Infinity
        assert d.is_energy == False
        assert d.mean_power(2) == sp.Rational(3, 5)
        assert d.mean_power(N) == (N + 1) / (2 * N + 1)
        assert d.mean_power() == sp.S.Half
        assert d.is_power == True

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

        g = f(ds.n).rewrite(UnitStep, form="accum")
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
        assert d.real_part == UnitRamp(ds.n)
        assert d.imag_part == 0
        assert d.real == ds.Ramp(ds.n)
        assert d.imag == ds.Constant(0)
        assert d.is_periodic == False
        assert d.period == None
        assert d.support == sp.Range(1, sp.S.Infinity)
        assert d.duration == sp.S.Infinity

        N = sp.Symbol("N", integer=True, positive=True)
        assert d.energy(2) == 5
        assert d.energy(N) == N ** 3 / 3 + N ** 2 / 2 + N / 6
        assert d.energy() == sp.S.Infinity
        assert d.is_energy == False
        assert d.mean_power(2) == sp.S.One
        assert d.mean_power(N) == (N ** 3 / 3 + N ** 2 / 2 + N / 6) / (2 * N + 1)
        assert d.mean_power() == sp.S.Infinity
        assert d.is_power == False

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
        assert d.real_part == UnitDeltaTrain(ds.n, N)
        assert d.imag_part == 0
        assert d.real == ds.DeltaTrain(N=N)
        assert d.imag == ds.Constant(0)
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

        s = ds.Data([1, 1])
        assert s.amplitude == UnitDelta(ds.n) + UnitDelta(ds.n - 1)

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
        assert s.support == sp.Range(10, 13)
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
        assert d.real_part == 2 * sp.cos(sp.S.Pi * ds.n / 6 + 3 * sp.S.Pi / 5)
        assert d.imag_part == 0
        assert d.real == ds.Sinusoid(2, sp.S.Pi / 6, 3 * sp.S.Pi / 5)
        assert d.imag == ds.Constant(0)
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
        assert d.real_part == x * sp.cos(sp.S.Pi * ds.n / 4 + sp.S.Pi / 12)
        assert d.imag_part == 0
        assert d.real == ds.DiscreteSignal.from_formula(
            x * sp.cos(sp.S.Pi * ds.n / 4 + sp.S.Pi / 12), ds.n
        )
        assert d.imag == ds.Constant(0)
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
        assert d.real_part == (-1) ** ds.n
        assert d.imag_part == 0
        assert d.real == ds.Exponential(alpha=-1)
        assert d.imag == ds.Constant(0)
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

        A = sp.Symbol("A", real=True, positive=True)
        N = sp.Symbol("N", integer=True, positive=True)
        s = ds.Sinusoid(A, sp.S.Pi / 4)
        assert s.energy(2) == 2 * A ** 2
        assert s.energy(N) == sp.Sum(
            A ** 2 * sp.cos(ds.n * sp.S.Pi / 4) ** 2, (ds.n, -N, N)
        )
        assert s.energy() == None
        assert s.energy(s.period / 2) == 5 * A ** 2
        assert s.is_energy == False
        assert s.mean_power(2) == 2 * A ** 2 / 5
        assert s.mean_power(N) == sp.Sum(
            A ** 2 * sp.cos(ds.n * sp.S.Pi / 4) ** 2, (ds.n, -N, N)
        ) / (2 * N + 1)
        assert s.mean_power() == None
        assert s.mean_power(s.period / 2) == 5 * A ** 2 / 9
        assert s.is_power == False


class Test_Exponential(object):
    def test_Exponential_constructor(self):
        a = sp.Symbol("a", real=True, constant=True)
        s = ds.Exponential(1, alpha=a)
        assert s is not None

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

        s = ds.Exponential(
            -2 * sp.exp(sp.I * sp.S.Pi / 10), 3 * sp.exp(sp.I * sp.S.Pi / 4)
        )
        assert s is not None

        with pytest.raises(ValueError):
            ds.Exponential()

        with pytest.raises(ValueError):
            ds.Exponential(C=ds.n)

        with pytest.raises(ValueError):
            ds.Exponential(alpha=ds.n)

    def test_Exponential_eval(self):
        s = ds.Exponential(C=2, alpha=1)
        with pytest.raises(ValueError):
            s.eval(0.5)
        with pytest.raises(ValueError):
            s(0.5)
        assert s.eval(0) == 2
        assert s.eval(1) == 2
        assert s.eval(-1) == 2
        assert s(0) == 2
        assert s(1) == 2
        assert s(-1) == 2
        assert s[0] == 2
        assert s[1] == 2
        assert s[-1] == 2
        assert s.eval([-1, 0, 1]) == [2, 2, 2]
        assert s([-1, 0, 1]) == [2, 2, 2]
        assert s[-1:2] == [2, 2, 2]

        s = ds.Exponential(C=-3, alpha=-1)
        with pytest.raises(ValueError):
            s.eval(0.5)
        with pytest.raises(ValueError):
            s(0.5)
        assert s.eval(0) == -3
        assert s.eval(1) == 3
        assert s.eval(-1) == 3
        assert s(0) == -3
        assert s(1) == 3
        assert s(-1) == 3
        assert s[0] == -3
        assert s[1] == 3
        assert s[-1] == 3
        assert s.eval([-1, 0, 1]) == [3, -3, 3]
        assert s([-1, 0, 1]) == [3, -3, 3]
        assert s[-1:2] == [3, -3, 3]

        s = ds.Exponential(5, sp.exp(-sp.I * 3 * sp.S.Pi / 8))
        with pytest.raises(ValueError):
            s.eval(0.5)
        with pytest.raises(ValueError):
            s(0.5)
        assert s.eval(0) == 5
        assert s.eval(1) == 5 * sp.exp(-sp.I * 3 * sp.S.Pi / 8)
        assert s.eval(-1) == 5 * sp.exp(sp.I * 3 * sp.S.Pi / 8)
        assert s(0) == 5
        assert s(1) == 5 * sp.exp(-sp.I * 3 * sp.S.Pi / 8)
        assert s(-1) == 5 * sp.exp(sp.I * 3 * sp.S.Pi / 8)
        assert s[0] == 5
        assert s[1] == 5 * sp.exp(-sp.I * 3 * sp.S.Pi / 8)
        assert s[-1] == 5 * sp.exp(sp.I * 3 * sp.S.Pi / 8)
        assert s.eval([-1, 0, 1]) == [
            5 * sp.exp(sp.I * 3 * sp.S.Pi / 8),
            5,
            5 * sp.exp(-sp.I * 3 * sp.S.Pi / 8),
        ]
        assert s([-1, 0, 1]) == [
            5 * sp.exp(sp.I * 3 * sp.S.Pi / 8),
            5,
            5 * sp.exp(-sp.I * 3 * sp.S.Pi / 8),
        ]
        assert s[-1:2] == [
            5 * sp.exp(sp.I * 3 * sp.S.Pi / 8),
            5,
            5 * sp.exp(-sp.I * 3 * sp.S.Pi / 8),
        ]

        s = ds.Exponential(C=-sp.Rational(1, 3), alpha=sp.Rational(1, 2))
        with pytest.raises(ValueError):
            s.eval(0.5)
        with pytest.raises(ValueError):
            s(0.5)
        assert s.eval(0) == -sp.Rational(1, 3)
        assert s.eval(1) == -sp.Rational(1, 6)
        assert s.eval(-1) == -sp.Rational(2, 3)
        assert s(0) == -sp.Rational(1, 3)
        assert s(1) == -sp.Rational(1, 6)
        assert s(-1) == -sp.Rational(2, 3)
        assert s[0] == -sp.Rational(1, 3)
        assert s[1] == -sp.Rational(1, 6)
        assert s[-1] == -sp.Rational(2, 3)
        assert s.eval([-1, 0, 1]) == [
            -sp.Rational(2, 3),
            -sp.Rational(1, 3),
            -sp.Rational(1, 6),
        ]
        assert s([-1, 0, 1]) == [
            -sp.Rational(2, 3),
            -sp.Rational(1, 3),
            -sp.Rational(1, 6),
        ]
        assert s[-1:2] == [
            -sp.Rational(2, 3),
            -sp.Rational(1, 3),
            -sp.Rational(1, 6),
        ]

        s = ds.Exponential(-1, sp.Rational(1, 4) * sp.exp(-sp.I * 3 * sp.S.Pi / 8))
        with pytest.raises(ValueError):
            s.eval(0.5)
        with pytest.raises(ValueError):
            s(0.5)
        assert s.eval(0) == -1
        assert s.eval(1) == -sp.exp(-sp.I * 3 * sp.S.Pi / 8) / 4
        assert s.eval(-1) == -4 * sp.exp(sp.I * 3 * sp.S.Pi / 8)
        assert s(0) == -1
        assert s(1) == -sp.exp(-sp.I * 3 * sp.S.Pi / 8) / 4
        assert s(-1) == -4 * sp.exp(sp.I * 3 * sp.S.Pi / 8)
        assert s[0] == -1
        assert s[1] == -sp.exp(-sp.I * 3 * sp.S.Pi / 8) / 4
        assert s[-1] == -4 * sp.exp(sp.I * 3 * sp.S.Pi / 8)
        assert s.eval([-1, 0, 1]) == [
            -4 * sp.exp(sp.I * 3 * sp.S.Pi / 8),
            -1,
            -sp.exp(-sp.I * 3 * sp.S.Pi / 8) / 4,
        ]
        assert s([-1, 0, 1]) == [
            -4 * sp.exp(sp.I * 3 * sp.S.Pi / 8),
            -1,
            -sp.exp(-sp.I * 3 * sp.S.Pi / 8) / 4,
        ]
        assert s[-1:2] == [
            -4 * sp.exp(sp.I * 3 * sp.S.Pi / 8),
            -1,
            -sp.exp(-sp.I * 3 * sp.S.Pi / 8) / 4,
        ]

    def test_Exponential_iv(self):
        s = ds.Exponential(alpha=1)
        assert s.is_discrete == True
        assert s.is_continuous == False
        assert s.iv == ds.n

        s = ds.Exponential(alpha=sp.Rational(1, 2))

        with pytest.raises(ValueError):
            s.shift(0.5)

        with pytest.raises(ValueError):
            s.delay(0.5)

        d = s.shift(1)
        assert d([-1, 0, 1]) == [4, 2, 1]

        d = s.flip()
        assert d([-1, 0, 1]) == [1 / 2, 1, 2]

        d = s.shift(1).flip()
        assert d([-1, 0, 1]) == [1, 2, 4]

        d = s.flip().shift(1)
        assert d([-1, 0, 1]) == [1 / 4, 1 / 2, 1]

    def test_generator(self):
        c = ds.Exponential(2, sp.Rational(1, 2))
        with pytest.raises(ValueError):
            dg = c.generate(0, step=0.1)
            next(dg)

        dg = c.generate(start=-3, size=3, overlap=1)
        assert next(dg) == [16, 8, 4]
        assert next(dg) == [4, 2, 1]
        assert next(dg) == [1, 1 / 2, 1 / 4]

    def test_Exponential_frequency(self):
        c = ds.Exponential(alpha=1)
        assert c.is_periodic == True
        assert c.period == 1
        assert c.frequency == 0
        assert c.phase == 0
        assert c.gain == 1

        c = ds.Exponential(C=3 * sp.I, alpha=1)
        assert c.is_periodic == True
        assert c.period == 1
        assert c.frequency == 0
        assert c.phase == sp.S.Pi / 2
        assert c.gain == 3

        c = ds.Exponential(alpha=-1)
        assert c.is_periodic == True
        assert c.period == 2
        assert c.frequency == sp.S.Pi
        assert c.phase == 0
        assert c.gain == 1

        c = ds.Exponential(C=-3, alpha=-1)
        assert c.is_periodic == True
        assert c.period == 2
        assert c.frequency == sp.S.Pi
        assert c.phase == sp.S.Pi
        assert c.gain == 3

        c = ds.Exponential(1, sp.exp(sp.I * sp.S.Pi / 4))
        assert c.is_periodic == True
        assert c.period == 8
        assert c.frequency == sp.S.Pi / 4
        assert c.phase == 0
        assert c.gain == 1

        c = ds.Exponential(-sp.I, sp.exp(sp.I * 3 * sp.S.Pi / 8))
        assert c.is_periodic == True
        assert c.period == 16
        assert c.frequency == 3 * sp.S.Pi / 8
        assert c.phase == -sp.S.Pi / 2
        assert c.gain == 1

        c = ds.Exponential(1, sp.exp(-sp.I * 3 * sp.S.Pi / 8))
        assert c.is_periodic == True
        assert c.period == 16
        assert c.frequency == -3 * sp.S.Pi / 8
        assert c.phase == 0
        assert c.gain == 1

        c = ds.Exponential(1, sp.exp(sp.I * 0.83 * sp.S.Pi))
        assert c.is_periodic == True
        assert c.period == 200
        assert c.frequency == 0.83 * sp.S.Pi
        assert c.phase == 0
        assert c.gain == 1

        c = ds.Exponential(alpha=sp.exp(sp.I * 3 / 8))
        assert c.is_periodic == False
        assert c.period == None
        assert c.frequency == 3 / 8
        assert c.phase == 0
        assert c.gain == 1

        c = ds.Exponential(C=-1, alpha=sp.Rational(1, 2))
        assert c.is_periodic == False
        assert c.period == None
        assert c.frequency == 0
        assert c.phase == sp.S.Pi
        assert c.gain == sp.Pow(sp.Rational(1, 2), ds.n)

        c = ds.Exponential(C=0.3, alpha=0.7)
        assert c.is_periodic == False
        assert c.period == None
        assert c.frequency == 0
        assert c.phase == 0
        assert c.gain == sp.S(0.3 * 0.7 ** ds.n)

        c = ds.Exponential(-sp.I, -2 * sp.exp(sp.I * 3 * sp.S.Pi / 8))
        assert c.is_periodic == False
        assert c.period == None
        assert c.frequency == 11 * sp.S.Pi / 8
        assert c.phase == -sp.S.Pi / 2
        assert c.gain == 2 ** ds.n

        c = ds.Exponential(1, sp.Rational(1, 4) * sp.exp(-sp.I * 3 * sp.S.Pi / 8))
        assert c.is_periodic == False
        assert c.period == None
        assert c.frequency == -3 * sp.S.Pi / 8
        assert c.phase == 0
        assert c.gain == sp.Rational(1, 4) ** ds.n

        c = ds.Exponential(1, sp.exp(-sp.I * 3 * sp.S.Pi / 8 + sp.S.Pi / 3))
        assert c.is_periodic == False
        assert c.period == None
        assert c.frequency == -3 * sp.S.Pi / 8
        assert c.phase == 0
        assert c.gain == sp.exp(sp.S.Pi / 3) ** ds.n

        c = ds.Exponential(1, -sp.exp(sp.I * 0.83 * sp.S.Pi + 0.1))
        assert c.is_periodic == False
        assert c.period == None
        assert c.frequency == 1.83 * sp.S.Pi
        assert c.phase == 0
        assert c.gain == sp.exp(0.1) ** ds.n

    def test_Exponential_misc(self):
        s = ds.Exponential(alpha=0.5)
        assert s.amplitude == sp.Pow(0.5, ds.n)
        assert s.phasor == 1
        assert s.carrier == 1

        s = ds.Exponential(alpha=-1)
        assert s.amplitude == sp.Pow(-1, ds.n)
        assert s.phasor == 1
        assert s.carrier == sp.exp(sp.I * sp.S.Pi * ds.n)

        s = ds.Exponential(2, sp.Rational(1, 2))
        assert sp.simplify(s.amplitude - 2 * sp.Pow(sp.Rational(1, 2), ds.n)) == 0
        assert s.phasor == 2
        assert s.carrier == 1

        s = ds.Exponential(
            -2 * sp.exp(sp.I * sp.S.Pi / 10), 3 * sp.exp(sp.I * sp.S.Pi / 4)
        )
        assert (
            sp.simplify(
                s.amplitude
                - ((3 ** ds.n) * sp.exp(sp.I * sp.S.Pi * ds.n / 4))
                * (-2 * (-1) ** (sp.Rational(1, 10)))
            )
            == 0
        )
        assert s.phasor == 2 * sp.exp(-9 * sp.I * sp.S.Pi / 10)
        assert s.carrier == sp.exp(sp.I * sp.S.Pi * ds.n / 4)

        omega, A = sp.symbols("omega A", real=True, positive=True)
        N = sp.Symbol("N", integer=True, positive=True)
        s = ds.Exponential(A, sp.exp(sp.I * omega))
        assert s.energy(2) == 5 * A ** 2
        assert s.energy(N) == (2 * N + 1) * A ** 2
        assert s.energy() == sp.S.Infinity
        assert s.is_energy == False
        assert s.mean_power(2) == A ** 2
        assert s.mean_power(N) == A ** 2
        assert s.mean_power() == A ** 2
        assert s.is_power == True


class Test_Undefined(object):
    def test_Undefined_constructor(self):
        d = ds.Undefined("x")
        assert d is not None

        d = ds.Undefined("y", ds.m)
        assert d is not None

        d = ds.Undefined("z", sp.Symbol("r", integer=True))
        assert d is not None

        d = ds.Undefined("x", codomain=sp.S.Complexes)
        assert d is not None

        d = ds.Undefined("x", period=10)
        assert d is not None

        d = ds.Undefined("y", ds.m, period=10)
        assert d is not None

        d = ds.Undefined("z", sp.Symbol("r", integer=True), period=10)
        assert d is not None

        d = ds.Undefined("x", duration=10, codomain=sp.S.Complexes)
        assert d is not None

        d = ds.Undefined("x", duration=10)
        assert d is not None

        d = ds.Undefined("y", ds.m, duration=10)
        assert d is not None

        d = ds.Undefined("z", sp.Symbol("r", integer=True), duration=10)
        assert d is not None

        d = ds.Undefined("x", duration=10, codomain=sp.S.Complexes)
        assert d is not None

        with pytest.raises(ValueError):
            ds.Undefined(ds.n)

        with pytest.raises(ValueError):
            ds.Undefined("x", ds.n, period=10, duration=100)

    def test_Undefined_eval(self):
        x = sp.Function("x", nargs=1)
        d = ds.Undefined("x")

        assert d[0] == x(0)
        assert d[1] == x(1)
        assert d[-1] == x(-1)
        with pytest.raises(ValueError):
            d[0.5]
        with pytest.raises(ValueError):
            d.eval(0.5)

        assert d[0:2] == [x(0), x(1)]
        assert d[-1:2] == [x(-1), x(0), x(1)]
        assert d[-4:1:2] == [x(-4), x(-2), x(0)]
        assert d[3:-2:-2] == [x(3), x(1), x(-1)]
        with pytest.raises(ValueError):
            d[0:2:0.5]

        assert d.eval(range(0, 2)) == [x(0), x(1)]
        assert d.eval(sp.Range(0, 2)) == [x(0), x(1)]
        assert d.eval(range(-1, 2)) == [x(-1), x(0), x(1)]
        assert d.eval(sp.Range(-1, 2)) == [x(-1), x(0), x(1)]
        assert d.eval(range(-4, 1, 2)) == [x(-4), x(-2), x(0)]
        assert d.eval(sp.Range(-4, 1, 2)) == [x(-4), x(-2), x(0)]
        assert d.eval(range(3, -2, -2)) == [x(3), x(1), x(-1)]
        assert d.eval(sp.Range(3, -2, -2)) == [x(3), x(1), x(-1)]

        with pytest.raises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_Undefined_iv(self):
        d = ds.Undefined("x")
        x = sp.Function("x", nargs=1)
        assert d.is_discrete == True
        assert d.is_continuous == False
        assert d.iv == ds.n
        # shift
        shift = 5
        d = ds.Undefined("x").shift(shift)
        assert d.iv == ds.n
        assert d[0] == x(-5)
        assert d[shift] == x(0)
        d = ds.Undefined("x", ds.n).shift(ds.k)
        assert d.iv == ds.n
        assert d([0, 1, 2], 1) == [x(-1), x(0), x(1)]
        # flip
        d = ds.Undefined("x").flip()
        assert d.iv == ds.n
        assert d[0] == x(0)
        # shift and flip
        shift = 5
        d = ds.Undefined("x").shift(shift).flip()
        assert d[-shift] == x(0)
        assert d[shift] == x(-10)
        # flip and shift
        shift = 5
        d = ds.Undefined("x").flip().shift(shift)
        assert d[-shift] == x(10)
        assert d[shift] == x(0)

    def test_Undefined_generator(self):
        d = ds.Undefined("x")
        x = sp.Function("x", nargs=1)
        with pytest.raises(ValueError):
            dg = d.generate(0, step=0.1)
            next(dg)

        dg = d.generate(start=-3, size=5, overlap=3)
        assert next(dg) == [x(-3), x(-2), x(-1), x(0), x(1)]
        assert next(dg) == [x(-1), x(0), x(1), x(2), x(3)]
        assert next(dg) == [x(1), x(2), x(3), x(4), x(5)]

    def test_Undefined_misc(self):
        d = ds.Undefined("x")
        x = sp.Function("x", nargs=1)
        assert d.amplitude == x(ds.n)
        assert d.real_part == sp.re(x(ds.n))
        assert d.real == ds.DiscreteSignal(
            sp.re(x(ds.n)), ds.n, None, sp.S.Integers, sp.S.Reals
        )
        assert d.imag_part == sp.im(x(ds.n))
        assert d.imag == ds.DiscreteSignal(
            sp.im(x(ds.n)), ds.n, None, sp.S.Integers, sp.S.Reals
        )
        assert d.is_periodic == False
        assert d.period == None
        assert d.support == sp.S.Integers
        assert d.duration == None

        d = ds.Undefined("x", period=10)
        assert d.amplitude == x(ds.n)
        assert d.real_part == sp.re(x(ds.n))
        assert d.real == ds.DiscreteSignal(
            sp.re(x(ds.n)), ds.n, None, sp.S.Integers, sp.S.Reals
        )
        assert d.imag_part == sp.im(x(ds.n))
        assert d.imag == ds.DiscreteSignal(
            sp.im(x(ds.n)), ds.n, None, sp.S.Integers, sp.S.Reals
        )
        assert d.is_periodic == True
        assert d.period == 10
        assert d.support == sp.S.Integers
        assert d.duration == None

        d = ds.Undefined("x", duration=10)
        assert d.amplitude == x(ds.n)
        assert d.real_part == sp.re(x(ds.n))
        assert d.real == ds.DiscreteSignal(
            sp.re(x(ds.n)), ds.n, None, sp.S.Integers, sp.S.Reals
        )
        assert d.imag_part == sp.im(x(ds.n))
        assert d.imag == ds.DiscreteSignal(
            sp.im(x(ds.n)), ds.n, None, sp.S.Integers, sp.S.Reals
        )
        assert d.is_periodic == False
        assert d.period == None
        assert d.support == sp.Range(0, 10)
        assert d.duration == 10

        f = sp.lambdify(d.iv, d.amplitude)
        with pytest.raises(NameError):
            f(0)
