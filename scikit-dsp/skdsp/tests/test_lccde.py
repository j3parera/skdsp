import pytest
import sympy as sp

from skdsp.util.lccde import LCCDE
from skdsp.signal.functions import UnitStep, UnitDelta, stepsimp
from skdsp.signal.discrete import n, DiscreteSignal


class Test_LCCDE(object):
    def test_LCCDE_constructor(self):
        A = [1, -sp.S(3) / 4, sp.S(1) / 8]
        B = [2, -sp.S(3) / 4]
        eq = LCCDE(B, A)
        assert eq is not None
        assert eq.iv == sp.Symbol("n", integer=True)
        assert eq.x() == sp.Function("x")(eq.iv)
        assert eq.y() == sp.Function("y")(eq.iv)
        assert eq.B == B
        assert eq.A == A
        assert eq.order == 2

    def test_LCCDE_solve_helpers(self):
        a = sp.Symbol("a")
        B = [1]
        A = [1, -a]
        lccde = LCCDE(B, A)
        n = lccde.iv
        yf = lccde.as_forward_recursion()
        assert yf == a * lccde.y(n - 1) + lccde.x(n)

        yb = lccde.as_backward_recursion()
        assert sp.simplify(yb - (lccde.y(n + 1) - lccde.x(n + 1)) / a) == sp.S.Zero

        yp = lccde.as_piecewise(ac='initial_rest')
        y = sp.Piecewise((yf, n >= 0), (0, True))
        assert sp.simplify(yp - y) == sp.S.Zero

        yp = lccde.as_piecewise(ac='final_rest')
        y = sp.Piecewise((yb, n <= -1), (0, True))
        assert sp.simplify(yp - y) == sp.S.Zero

        yp = lccde.as_piecewise(ac={-1: lccde.y(-1)})
        y = sp.Piecewise((yb, n < -1), (lccde.y(-1), sp.Eq(n, -1)), (yf, n > -1))
        assert sp.simplify(yp - y) == sp.S.Zero

        assert y.subs(n, -1) == lccde.y(-1)
        assert y.subs(n, 0) == a * lccde.y(-1) + lccde.x(0)
        assert sp.simplify(y.subs(n, -2) - (lccde.y(-1) - lccde.x(-1)) / a) == sp.S.Zero

        yh, Cs = lccde.solve_homogeneous()
        assert len(Cs) == 1
        assert yh == Cs[0] * a ** n

        A = [1, -sp.Rational(3, 4), sp.Rational(1, 8)]
        B = [1, -sp.Rational(3, 8)]
        lccde = LCCDE(B, A)
        yf = lccde.as_forward_recursion()
        expected = (
            -A[1] * lccde.y(n - 1)
            - A[2] * lccde.y(n - 2)
            + B[0] * lccde.x(n)
            + B[1] * lccde.x(n - 1)
        )
        assert sp.simplify(yf - expected) == sp.S.Zero

        yb = lccde.as_backward_recursion()
        expected = (
            -8 * lccde.y(n + 2)
            - 8 * A[1] * lccde.y(n + 1)
            + 8 * B[0] * lccde.x(n + 2)
            + 8 * B[1] * lccde.x(n + 1)
        )
        assert sp.simplify(yb - expected) == sp.S.Zero

    def test_LCCDE_solve_free(self):
        a = sp.Symbol("a")
        B = [1]
        A = [1, -a]
        lccde = LCCDE(B, A)
        n = lccde.iv

        with pytest.raises(ValueError):
            yf = lccde.solve_free("kk_rest")

        with pytest.raises(ValueError):
            yf = lccde.solve_free(None)

        yf = lccde.solve_free("initial_rest")
        assert yf == sp.S.Zero

        yf = lccde.solve_free("final_rest")
        assert yf == sp.S.Zero

        n0 = sp.Symbol("n0", integer=True)
        K = sp.Symbol("K")
        yf = lccde.solve_free({n0: K})
        assert sp.simplify(yf - K * a ** (n - n0)) == sp.S.Zero

        n0 = 3
        yf = lccde.solve_free({n0: K})
        assert sp.simplify(yf - K * a ** (n - n0)) == sp.S.Zero

        n0 = -3
        yf = lccde.solve_free({n0: K})
        assert sp.simplify(yf - K * a ** (n - n0)) == sp.S.Zero

        n0 = sp.Symbol("n0", integer=True)
        K = sp.S.Zero
        yf = lccde.solve_free({n0: K})
        assert yf == sp.S.Zero

        A = [1, -sp.Rational(3, 4), sp.Rational(1, 8)]
        B = [1, -sp.Rational(3, 8)]
        lccde = LCCDE(B, A)
        yf = lccde.solve_free(ac={3: 2, 2: 5})
        assert sp.simplify(yf - (12 * 2 ** (-n) + 32 * 4 ** (-n))) == sp.S.Zero

        yf = lccde.solve_free("initial_rest")
        assert yf == sp.S.Zero

        yf = lccde.solve_free("final_rest")
        assert yf == sp.S.Zero

        n0 = sp.Symbol("n0", integer=True)
        n1 = sp.Symbol("n1", integer=True)
        K0 = sp.Symbol("K0")
        K1 = sp.Symbol("K1")
        yf = lccde.solve_free({n0: K0, n1: K1})
        M = sp.Matrix([[2 ** -n0, 4 ** -n0], [2 ** -n1, 4 ** -n1]])
        dM = M.det()
        V = [[K0], [K1]]
        MC1 = M[:, :]
        MC1[:, 0] = V
        dMC1 = MC1.det()
        MC2 = M[:, :]
        MC2[:, 1] = V
        dMC2 = MC2.det()
        expected = dMC1 / dM * (2 ** -n) + dMC2 / dM * (4 ** -n)
        assert sp.simplify(yf - expected) == sp.S.Zero

        yf = lccde.solve_free({0: 1, 2: 2})
        assert yf

    def test_LCCDE_solve_forced(self):
        A = [1, -sp.Rational(3, 4), sp.Rational(1, 8)]
        B = [1, -sp.Rational(3, 8)]
        eq = LCCDE(B, A)
        yh, _ = eq.solve_homogeneous()
        Ca = sp.Wild("Ca")
        Cb = sp.Wild("Cb")
        roots = list(eq.roots())
        expr = Ca * roots[0] ** eq.iv + Cb * roots[1] ** eq.iv
        match = yh.match(expr)
        assert match is not None

        # TODO ac
        # yh = eq.solve_forced(UnitDelta(eq.iv), ac={0: 1, -1: -2})
        # expr = -2 * roots[0] ** eq.iv + 3 * roots[1] ** eq.iv
        # dif = sp.simplify(yh - expr)
        # assert dif == sp.S.Zero

        # TODO solve_free
        # yh = eq.solve_free(ac='initial_rest')
        # assert yh == sp.S.Zero

        # r = sp.Symbol('r', positive=True)
        # A = [1, -(1 + r)]
        # B = [-1]
        # eq = LCCDE(B, A)
        # yh = eq.solve_free({0: -1})
        # expr = -(1 + r)**eq.iv
        # dif = sp.simplify(yh - expr)
        # assert dif == sp.S.Zero

        # TODO solve_forced(u[n])
        # a1 = sp.Symbol('a1')
        # B = [1]
        # A = [1, a1]
        # eq = LCCDE(B, A)
        # y = eq.solve_forced(UnitStep(n))

    def test_LCDDE_from_expression(self):
        y = sp.Function("y")
        x = sp.Function("x")
        n = sp.Symbol("n", integer=True)
        ed = sp.Eq(
            y(n) - sp.S(3) / 4 * y(n - 1) + sp.S(1) / 8 * y(n - 2),
            2 * x(n) - sp.S(3) / 4 * x(n - 1),
        )
        eq = LCCDE.from_expression(ed, x(n), y(n))
        assert eq.A == [1, -sp.S(3) / 4, sp.S(1) / 8]
        assert eq.B == [2, -sp.S(3) / 4]
        ed2 = eq.as_expression
        y1 = sp.solve(ed, y(n))
        y2 = sp.solve(ed2, y(n))
        assert y1 == y2
