import numpy as np
import pytest
import sympy as sp
from sympy.solvers.ode import ode_nth_linear_constant_coeff_homogeneous

from skdsp.signal.discrete import Constant, Data, Delta, Exponential, Step, n
from skdsp.system.discrete import Delay, DiscreteSystem, Identity, Zero, filter
from skdsp.util.lccde import LCCDE
from skdsp.signal.functions import UnitStep, UnitDelta


class Test_DiscreteSystem(object):

    def test_DiscreteSystem_as_LCCDE(self):
        n = sp.Symbol("n", integer=True)
        k = sp.Symbol("k", integer=True)
        x = sp.Function("x")
        y = sp.Function("y")

        eq = sp.Eq(y(n) - y(n-1)/4, x(n) - x(n-2))
        S = DiscreteSystem(eq)
        h = S.impulse_response(force_lti=True)
        assert h is not None

        eq = sp.Eq(y(n) - sp.S(1)*y(n-1)/4, x(n))
        S = DiscreteSystem(eq)
        h = S.impulse_response(force_lti=True, select='causal')
        assert h is not None

        h = S.impulse_response(force_lti=True, select='anticausal')
        assert h is not None

        T = sp.Eq(y(n), x(n) - x(n - 1))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert isinstance(eq, LCCDE)
        assert eq.B == [sp.S.One, -sp.S.One] 
        assert eq.A == [sp.S.One] 

        T = sp.Eq(y(n), x(n) - y(n - 1))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert isinstance(eq, LCCDE)
        assert eq.B == [sp.S.One] 
        assert eq.A == [sp.S.One, sp.S.One] 

        T = sp.Eq(y(n), -10 * x(n))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert isinstance(eq, LCCDE)
        assert eq.B == [-10] 
        assert eq.A == [sp.S.One] 

        T = sp.Eq(y(n), y(n - 1) + sp.Sum(x(k), (k, sp.S.NegativeInfinity, n)))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert eq is None

        T = sp.Eq(y(n), y(n + 1) + sp.Sum(x(k), (k, sp.S.NegativeInfinity, n)))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert eq is None

        T = sp.Eq(y(n), y(n + 1) + sp.Sum(x(k), (k, sp.S.NegativeInfinity, n + 1)))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert eq is None

        T = sp.Eq(y(n), x(n) + 3 * x(n + 4))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert eq is None

        T = sp.Eq(y(n), x(n ** 2))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert eq is None

        T = sp.Eq(y(n), x(2 * n))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert eq is None

        T = sp.Eq(y(n), x(-n))
        S = DiscreteSystem(T)
        eq = S.as_lccde
        assert eq is None

        eq = sp.Eq(y(n) - 3*y(n-1) - 4*y(n-2), x(n) + 2*x(n-1))
        S = DiscreteSystem(eq)
        h = S.impulse_response(force_lti=True)
        expr = (-sp.S(1)/5*(-1)**n + sp.S(6)/5*(4)**n) * UnitStep(n)
        assert sp.simplify(h.amplitude - expr) == sp.S.Zero

        # eq = sp.Eq(y(n) + sp.S(9)/10*y(n-2), sp.S(3)/10*x(n) + sp.S(6)/10*x(n-1) + sp.S(3)/10*x(n-2))
        eq = sp.Eq(y(n) + sp.S(9)/10*y(n-2), sp.S(3)/5*x(n) + sp.S(3)/100*x(n-1))
        # eq = sp.Eq(10*y(n) + 9*y(n-2), 3*x(n) + 6*x(n-1) + 3*x(n-2))
        S = DiscreteSystem(eq)
        h = S.impulse_response(force_lti=True)
        # s = h.latex()
        assert h is not None

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
        y3 = Exponential(
            sp.exp(sp.I * sp.S.Pi / 4), 1.1 * sp.exp(sp.I * sp.S.Pi / 11)
        )
        G = y3.phasor
        z0 = y3.alpha
        x = G * Delta()
        span = range(0, 51)
        B = [1, 0]
        A = [1, -z0]
        y3f = Data(filter(B, A, x(span)), iv=n)

    def test_dsolve(self):
        f = sp.Function("f")
        y = sp.Function("y")
        # ode = sp.Eq(sp.S(1)/8*f(n).diff(n, 2) - sp.S(3)/4*f(n).diff(n) + f(n), 10)
        ode = sp.Eq(f(n).diff(n, 2) - sp.S(3)/4*f(n).diff(n) + sp.S(1)/8*f(n), 0)
        # jint = 'nth_linear_constant_coeff_undetermined_coefficients'
        jint = 'nth_linear_constant_coeff_homogeneous'
        # sol = sp.dsolve(ode, f(n), hint=jint)
        lccde_out = y(n) - sp.S(3)/4*y(n-1) + sp.S(1)/8*y(n-2)
        coeffs = {0: sp.S(1)/8, 1: -sp.S(3)/4, 2: 1}
        sol = ode_nth_linear_constant_coeff_homogeneous(lccde_out, f(n), len(coeffs)-1, coeffs, returns="sol")
        assert sol is not None

    def test_recursion(self):
        x = sp.S(3)/10*Delta() + sp.S(6)/10*Delta().shift(1) + sp.S(3)/10*Delta().shift(2)
        y = [0, 0]
        for k in range(0, 128):
            y.append(-sp.S(9)/10*y[k-2+2] + x(k))
        assert y is not None

        B = [sp.S(3) / 10, sp.S(6) / 10, sp.S(3) / 10]
        A = [1, 0, sp.S(9) / 10]
        Y = filter(B, A, Delta()(range(0, 129)))
        assert Y is not None

        x = Delta()
        xv = x[-2:51]
        yv = [0, 0]
        span = range(0, 50)
        for m in span:
            v = -A[2]*yv[m] + B[0]*xv[m+2] + B[1]*xv[m+1] + B[2]*xv[m]
            yv.append(v)
        from skdsp.util.util import ipystem
        ipystem(span, np.array([float(y) for y in yv[2:]]), title=r'$h[n]$');

    def test_roots(self):
        x = sp.Symbol('x')
        poly = sp.Poly(x**2 - 2*x + 1) 
        res = sp.roots(poly)
        assert res is not None
