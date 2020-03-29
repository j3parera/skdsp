import numpy as np
import pytest
import sympy as sp
from sympy.solvers.ode import ode_nth_linear_constant_coeff_homogeneous

from skdsp.signal.discrete import Constant, Data, Delta, Exponential, Step, n
from skdsp.system.discrete import Delay, DiscreteSystem, Identity, Zero
from skdsp.util.lccde import LCCDE
from skdsp.signal.functions import UnitStep, UnitDelta


class Test_DiscreteSystem(object):
    def test_DiscreteSystem_constructor(self):
        n = sp.Symbol("n", integer=True)
        k = sp.Symbol("k", integer=True)
        x = sp.Function("x")
        y = sp.Function("y")

        T = sp.Eq(y(n), x(n) - 2*x(n - 1))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert isinstance(lccde, LCCDE)
        assert lccde.B == [sp.S.One, -2]
        assert lccde.A == [sp.S.One]
        assert S.is_fir
        assert not S.is_iir
        assert S.is_stable

        T = sp.Eq(y(n), x(n) - y(n - 1))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert isinstance(lccde, LCCDE)
        assert lccde.B == [sp.S.One]
        assert lccde.A == [sp.S.One, sp.S.One]
        assert not S.is_fir
        assert S.is_iir
        assert not S.is_stable

        T = sp.Eq(y(n), -10 * x(n))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert isinstance(lccde, LCCDE)
        assert lccde.B == [-10]
        assert lccde.A == [sp.S.One]
        assert S.is_fir
        assert not S.is_iir
        assert S.is_stable

        T = sp.Eq(y(n), y(n - 1) + sp.Sum(x(k), (k, sp.S.NegativeInfinity, n)))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert lccde is None
        assert not S.is_fir
        assert S.is_iir
        assert S.is_stable == None

        T = sp.Eq(y(n), y(n + 1) + sp.Sum(x(k), (k, sp.S.NegativeInfinity, n)))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert lccde is None
        assert not S.is_fir
        assert S.is_iir
        assert S.is_stable == None

        T = sp.Eq(y(n), y(n + 1) + sp.Sum(x(k), (k, sp.S.NegativeInfinity, n + 1)))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert lccde is None
        assert not S.is_fir
        assert S.is_iir
        assert S.is_stable == None

        T = sp.Eq(y(n), x(n) + 3 * x(n + 4))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert lccde is None
        assert S.is_fir
        assert not S.is_iir
        assert S.is_stable

        T = sp.Eq(y(n), x(n ** 2))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert lccde is None
        assert S.is_fir
        assert not S.is_iir
        assert S.is_stable

        T = sp.Eq(y(n), x(2 * n))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert lccde is None
        assert S.is_fir
        assert not S.is_iir
        assert S.is_stable

        T = sp.Eq(y(n), x(-n))
        S = DiscreteSystem(T)
        lccde = S.as_lccde
        assert lccde is None
        assert S.is_fir
        assert not S.is_iir
        assert S.is_stable

        lccde = LCCDE(
            [1, -sp.Rational(3, 8)], [1, -sp.Rational(3, 4), -sp.Rational(1, 8)]
        )
        S = DiscreteSystem(lccde)
        assert S.is_discrete
        assert S.is_input_discrete
        assert S.is_output_discrete
        assert not S.is_continuous
        assert not S.is_input_continuous
        assert not S.is_input_continuous
        assert not S.is_hybrid
        assert not S.is_memoryless
        assert not S.is_time_invariant
        assert not S.is_linear
        assert not S.is_causal
        assert not S.is_anticausal
        assert S.is_recursive
        assert S.is_stable
        assert not S.is_fir
        assert S.is_iir

        lccde = LCCDE(
            [1, 0, -1], [1, -1]
        )
        S = DiscreteSystem(lccde)
        assert S.is_dynamic
        assert S.is_recursive
        assert S.is_fir
        assert not S.is_iir
        assert S.is_stable

        B = [1, sp.Rational(1, 2)]
        A = [1, -sp.Rational(18, 10) * sp.cos(sp.S.Pi / 16), sp.Rational(81, 100)]
        S = DiscreteSystem.from_coefficients(B, A)
        lccde = S.as_lccde
        assert lccde.B == B
        assert lccde.A == A
        assert not S.is_fir
        assert S.is_iir
        assert S.is_stable

    def test_DiscreteSystem_impulse_response(self):
        x = sp.Function("x")
        y = sp.Function("y")

        lccde = LCCDE(
            [2, -sp.Rational(3, 4)], [1, -sp.Rational(3, 4), sp.Rational(1, 8)]
        )
        S = DiscreteSystem(lccde)
        h = S.impulse_response()
        assert h == None

        h = S.impulse_response(force_lti=True)
        expected = sp.Rational(1, 2) ** n * UnitStep(n) + sp.Rational(
            1, 4
        ) ** n * UnitStep(n)
        assert sp.simplify(h.amplitude - expected) == sp.S.Zero

        h = S.impulse_response(force_lti=True, select="causal")
        expected = sp.Rational(1, 2) ** n * UnitStep(n) + sp.Rational(
            1, 4
        ) ** n * UnitStep(n)
        assert sp.simplify(h.amplitude - expected) == sp.S.Zero

        h = S.impulse_response(force_lti=True, select="anticausal")
        expected = -sp.Rational(1, 2) ** n * UnitStep(-n - 1) + -sp.Rational(
            1, 4
        ) ** n * UnitStep(-n - 1)
        assert sp.simplify(h.amplitude - expected) == sp.S.Zero

        eq = sp.Eq(y(n) - y(n - 1) / 4, x(n) - x(n - 2))
        S = DiscreteSystem(eq)
        h = S.impulse_response(force_lti=True)
        expected = (
            16 * UnitDelta(n)
            + 4 * UnitDelta(n - 1)
            + sp.Rational(1, 4) ** n * UnitStep(n)
        )
        assert sp.simplify(h.amplitude - expected) == sp.S.Zero

        eq = sp.Eq(y(n) - 3 * y(n - 1) - 4 * y(n - 2), x(n) + 2 * x(n - 1))
        S = DiscreteSystem(eq)
        h = S.impulse_response(force_lti=True)
        expected = (-sp.S(1) / 5 * (-1) ** n + sp.S(6) / 5 * (4) ** n) * UnitStep(n)
        assert sp.simplify(h.amplitude - expected) == sp.S.Zero

        eq = sp.Eq(y(n) - sp.S(1) * y(n - 1) / 4, x(n))
        S = DiscreteSystem(eq)
        h = S.impulse_response(force_lti=True, select="causal")
        expected = sp.Rational(1, 4) ** n * UnitStep(n)
        assert sp.simplify(h.amplitude - expected) is not None

        h = S.impulse_response(force_lti=True, select="anticausal")
        expected = -sp.Rational(1, 4) ** -n * UnitStep(-n - 1)
        assert sp.simplify(h.amplitude - expected) is not None


class Test_Zero(object):
    def test_Zero_misc(self):
        S = Zero()
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

    def test_Zero_impulse_response(self):
        S = Zero()
        h = S.impulse_response()
        assert h == Constant(0)


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
        h = S.impulse_response()
        assert h == Delta()


class Test_Delay(object):
    def test_Delay_misc(self):
        S = Delay(sp.Symbol("k", integer=True, positive=True))
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

        S = Delay(sp.Symbol("k", integer=True))
        assert not S.is_causal

        with pytest.raises(ValueError):
            Delay(sp.Symbol("N"))

        with pytest.raises(ValueError):
            Delay(2 * n)

    def test_Delay_impulse_response(self):
        S = Delay(3)
        h = S.impulse_response()
        assert h == Delta().shift(3)


class Test_KK(object):
    def test_filter(self):
        y3 = Exponential(sp.exp(sp.I * sp.S.Pi / 4), 1.1 * sp.exp(sp.I * sp.S.Pi / 11))
        G = y3.phasor
        z0 = y3.alpha
        x = G * Delta()
        span = range(0, 51)
        B = [1, 0]
        A = [1, -z0]
        S = DiscreteSystem(LCCDE(B, A))
        y3f = Data(S.filter(x(span)), iv=n)
        z_0 = sp.S(1) / 2 * sp.exp(sp.I * 0)
        span = range(0, 21)
        x = Delta()
        B = [1, 0]
        Ar = [1, -sp.re(z_0)]
        Sr = DiscreteSystem(LCCDE(B, Ar))
        yrf = Data(Sr.filter(x.real(span)))
        Ai = [1, -sp.im(z_0)]
        Si = DiscreteSystem(LCCDE(B, Ai))
        yif = Data(Si.filter(x.imag(span)))

    def test_dsolve(self):
        f = sp.Function("f")
        y = sp.Function("y")
        # ode = sp.Eq(sp.S(1)/8*f(n).diff(n, 2) - sp.S(3)/4*f(n).diff(n) + f(n), 10)
        ode = sp.Eq(
            f(n).diff(n, 2) - sp.S(3) / 4 * f(n).diff(n) + sp.S(1) / 8 * f(n), 0
        )
        # jint = 'nth_linear_constant_coeff_undetermined_coefficients'
        jint = "nth_linear_constant_coeff_homogeneous"
        # sol = sp.dsolve(ode, f(n), hint=jint)
        lccde_out = y(n) - sp.S(3) / 4 * y(n - 1) + sp.S(1) / 8 * y(n - 2)
        coeffs = {0: sp.S(1) / 8, 1: -sp.S(3) / 4, 2: 1}
        sol = ode_nth_linear_constant_coeff_homogeneous(
            lccde_out, f(n), len(coeffs) - 1, coeffs, returns="sol"
        )
        assert sol is not None

    def test_recursion(self):
        x = (
            sp.S(3) / 10 * Delta()
            + sp.S(6) / 10 * Delta().shift(1)
            + sp.S(3) / 10 * Delta().shift(2)
        )
        y = [0, 0]
        for k in range(0, 128):
            y.append(-sp.S(9) / 10 * y[k - 2 + 2] + x(k))
        assert y is not None

        B = [sp.S(3) / 10, sp.S(6) / 10, sp.S(3) / 10]
        A = [1, 0, sp.S(9) / 10]
        S = DiscreteSystem(LCCDE(B, A))
        Y = S.filter(Delta()(range(0, 129)))
        assert Y is not None

        x = Delta()
        xv = x[-2:51]
        yv = [0, 0]
        span = range(0, 50)
        for m in span:
            v = -A[2] * yv[m] + B[0] * xv[m + 2] + B[1] * xv[m + 1] + B[2] * xv[m]
            yv.append(v)
        from skdsp.util.util import ipystem

        ipystem(span, np.array([float(y) for y in yv[2:]]), title=r"$h[n]$")

    def test_roots(self):
        B = [sp.Rational(3, 10), sp.Rational(6, 10), sp.Rational(3, 10)]
        A = [1, 0, sp.Rational(9, 10)]
        lccde = LCCDE(B, A)
        S = DiscreteSystem(lccde)
        h = S.impulse_response(force_lti=True)
        h._repr_latex_()
        assert h is not None
