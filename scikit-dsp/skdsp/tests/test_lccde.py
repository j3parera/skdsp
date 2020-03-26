import pytest
import sympy as sp

from skdsp.util.lccde import LCCDE
from skdsp.signal.functions import UnitStep, UnitDelta, stepsimp
from skdsp.signal.discrete import n, DiscreteSignal


class Test_LCCDE(object):
    def test_LCCDE_constructor(self):
        A = [1, -sp.S(3) / 4, sp.S(1) / 8]
        B = [2, -sp.S(3) / 4]
        lccde = LCCDE(B, A)
        assert lccde is not None
        assert lccde.iv == sp.Symbol("n", integer=True)
        assert lccde.x() == sp.Function("x")(lccde.iv)
        assert lccde.y() == sp.Function("y")(lccde.iv)
        assert lccde.B == B
        assert lccde.A == A
        assert lccde.order == 2

    def test_LCDDE_from_expression(self):
        y = sp.Function("y")
        x = sp.Function("x")
        n = sp.Symbol("n", integer=True)
        ed = sp.Eq(
            y(n) - sp.S(3) / 4 * y(n - 1) + sp.S(1) / 8 * y(n - 2),
            2 * x(n) - sp.S(3) / 4 * x(n - 1),
        )
        lccde = LCCDE.from_expression(ed, x(n), y(n))
        assert lccde.A == [1, -sp.S(3) / 4, sp.S(1) / 8]
        assert lccde.B == [2, -sp.S(3) / 4]
        ed2 = lccde.as_expression
        y1 = sp.solve(ed, y(n))
        y2 = sp.solve(ed2, y(n))
        assert y1 == y2

    def test_LCCDE_solve_recursion(self):
        a = sp.Symbol("a")
        B = [1]
        A = [1, -a]
        lccde = LCCDE(B, A)
        n = lccde.iv

        yf = lccde.as_forward_recursion()
        assert yf == a * lccde.y(n - 1) + lccde.x(n)

        yb = lccde.as_backward_recursion()
        assert sp.simplify(yb - (lccde.y(n + 1) - lccde.x(n + 1)) / a) == sp.S.Zero

        res = lccde.forward_recur(range(3))
        assert res[-1] == lccde.y(-1)
        assert res[0] == a * lccde.y(-1) + lccde.x(0)
        assert res[1] == a ** 2 * lccde.y(-1) + a * lccde.x(0) + lccde.x(1)
        assert res[2] == (
            a ** 3 * lccde.y(-1) + a ** 2 * lccde.x(0) + a * lccde.x(1) + lccde.x(2)
        )

        res = lccde.backward_recur(range(-1, -4, -1))
        assert res[0] == lccde.y(0)
        assert res[-1] == -lccde.x(0) / a + lccde.y(0) / a
        assert res[-2] == -lccde.x(-1) / a - lccde.x(0) / a ** 2 + lccde.y(0) / a ** 2
        assert res[-3] == (
            -lccde.x(-2) / a
            - lccde.x(-1) / a ** 2
            + -lccde.x(0) / a ** 3
            + lccde.y(0) / a ** 3
        )

        res = lccde.forward_recur(range(1, 3))
        assert res[0] == lccde.y(0)
        assert res[1] == a * lccde.y(0) + lccde.x(1)
        assert res[2] == a ** 2 * lccde.y(0) + a * lccde.x(1) + lccde.x(2)

        res = lccde.backward_recur(range(-2, -4, -1))
        assert res[-1] == lccde.y(-1)
        assert res[-2] == -lccde.x(-1) / a + lccde.y(-1) / a
        assert res[-3] == -lccde.x(-2) / a - lccde.x(-1) / a ** 2 + lccde.y(-1) / a ** 2

        res = lccde.forward_recur(range(3), UnitDelta(n))
        assert res[-1] == lccde.y(-1)
        assert res[0] == a * lccde.y(-1) + 1
        assert res[1] == a ** 2 * lccde.y(-1) + a
        assert res[2] == a ** 3 * lccde.y(-1) + a ** 2

        res = lccde.backward_recur(range(-1, -4, -1), UnitDelta(n))
        assert res[0] == lccde.y(0)
        assert res[-1] == lccde.y(0) / a - 1 / a
        assert res[-2] == lccde.y(0) / a ** 2 - 1 / a ** 2
        assert res[-3] == lccde.y(0) / a ** 3 - 1 / a ** 3

        res = lccde.forward_recur(range(3), UnitStep(n))
        assert res[-1] == lccde.y(-1)
        assert res[0] == a * lccde.y(-1) + 1
        assert res[1] == a ** 2 * lccde.y(-1) + a + 1
        assert res[2] == a ** 3 * lccde.y(-1) + a ** 2 + a + 1

        res = lccde.backward_recur(range(-1, -4, -1), UnitStep(-n - 1))
        assert res[0] == lccde.y(0)
        assert res[-1] == lccde.y(0) / a
        assert res[-2] == -1 / a + lccde.y(0) / a ** 2
        assert res[-3] == -1 / a - 1 / a ** 2 + lccde.y(0) / a ** 3

        yp = lccde.as_piecewise(ac="initial_rest")
        y = sp.Piecewise((yf, n >= 0), (0, True))
        assert sp.simplify(yp - y) == sp.S.Zero

        yp = lccde.as_piecewise(ac="final_rest")
        y = sp.Piecewise((yb, n <= -1), (0, True))
        assert sp.simplify(yp - y) == sp.S.Zero

        yp = lccde.as_piecewise(ac={-1: lccde.y(-1)})
        y = sp.Piecewise((yb, n < -1), (lccde.y(-1), sp.Eq(n, -1)), (yf, n > -1))
        assert sp.simplify(yp - y) == sp.S.Zero

        assert yp.subs(n, -1) == lccde.y(-1)
        assert yp.subs(n, 0) == a * lccde.y(-1) + lccde.x(0)
        assert (
            sp.simplify(yp.subs(n, -2) - (lccde.y(-1) - lccde.x(-1)) / a) == sp.S.Zero
        )

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

        res = lccde.forward_recur(range(2))
        assert res[-2] == lccde.y(-2)
        assert res[-1] == lccde.y(-1)
        assert res[0] == (
            -3 * lccde.x(-1) / 8 + lccde.x(0) - lccde.y(-2) / 8 + 3 * lccde.y(-1) / 4
        )
        assert res[1] == (
            -9 * lccde.x(-1) / 32
            + 3 * lccde.x(0) / 8
            + lccde.x(1)
            - 3 * lccde.y(-2) / 32
            + 7 * lccde.y(-1) / 16
        )

        res = lccde.backward_recur(range(-1, -3, -1))
        assert res[1] == lccde.y(1)
        assert res[0] == lccde.y(0)
        assert res[-1] == (
            -3 * lccde.x(0) + 8 * lccde.x(1) + 6 * lccde.y(0) - 8 * lccde.y(1)
        )
        assert res[-2] == (
            -3 * lccde.x(-1)
            - 10 * lccde.x(0)
            + 48 * lccde.x(1)
            + 28 * lccde.y(0)
            - 48 * lccde.y(1)
        )

        res = lccde.forward_recur(range(2), UnitDelta(n))
        assert res[-2] == lccde.y(-2)
        assert res[-1] == lccde.y(-1)
        assert res[0] == -lccde.y(-2) / 8 + 3 * lccde.y(-1) / 4 + 1
        assert res[1] == -3 * lccde.y(-2) / 32 + 7 * lccde.y(-1) / 16 + sp.Rational(
            3, 8
        )

        res = lccde.backward_recur(range(-1, -3, -1), UnitDelta(n))
        assert res[1] == lccde.y(1)
        assert res[0] == lccde.y(0)
        assert res[-1] == 6 * lccde.y(0) - 8 * lccde.y(1) - 3
        assert res[-2] == 28 * lccde.y(0) - 48 * lccde.y(1) - 10

    def test_LCCDE_solve_homogeneous(self):
        a = sp.Symbol("a")
        B = [1]
        A = [1, -a]
        lccde = LCCDE(B, A)
        n = lccde.iv

        yh, Cs = lccde.solve_homogeneous()
        assert len(Cs) == lccde.order
        assert yh == Cs[0] * a ** n

        B = [1]
        A = [1, -3, -4]
        lccde = LCCDE(B, A)
        n = lccde.iv

        yh, Cs = lccde.solve_homogeneous()
        assert len(Cs) == lccde.order
        assert yh == Cs[1] * (-1) ** n + Cs[0] * (4) ** n

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
        assert lccde.check_solution(sp.S.Zero, yf)

        n0 = 3
        yf = lccde.solve_free({n0: K})
        assert sp.simplify(yf - K * a ** (n - n0)) == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        n0 = -3
        yf = lccde.solve_free({n0: K})
        assert sp.simplify(yf - K * a ** (n - n0)) == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        n0 = sp.Symbol("n0", integer=True)
        K = sp.S.Zero
        yf = lccde.solve_free({n0: K})
        assert yf == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        A = [1, -sp.Rational(3, 4), sp.Rational(1, 8)]
        B = [1, -sp.Rational(3, 8)]
        lccde = LCCDE(B, A)
        yf = lccde.solve_free(ac={3: 2, 2: 5})
        assert sp.simplify(yf - (12 * 2 ** (-n) + 32 * 4 ** (-n))) == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        yf = lccde.solve_free("initial_rest")
        assert yf == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        yf = lccde.solve_free("final_rest")
        assert yf == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

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
        assert lccde.check_solution(sp.S.Zero, yf)

        yf = lccde.solve_free({0: 1, 2: 2})
        expected = -28 / 4 ** n / 3 + 31 / 2 ** n / 3
        assert sp.simplify(yf - expected) == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        B = [1]
        A = [1, -3, -4]
        lccde = LCCDE(B, A)
        n = lccde.iv

        ym1 = lccde.y(-1)
        ym2 = lccde.y(-2)
        yf = lccde.solve_free({-1: ym1, -2: ym2})
        expected = (-1) ** n * (4 * ym2 / 5 - ym1 / 5) + 4 ** n * (
            16 * ym2 / 5 + 16 * ym1 / 5
        )
        assert sp.simplify(yf - expected) == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        yf = lccde.solve_free({-1: 5, -2: 0})
        expected = (-1) ** (n + 1) + 4 ** (n + 2)
        assert sp.simplify(yf - expected) == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        r = sp.Symbol('r', positive=True)
        B = [-1]
        A = [1, -(1 + r)]
        lccde = LCCDE(B, A)
        yf = lccde.solve_free({0: -1})
        expr = -(1 + r)**lccde.iv
        dif = sp.simplify(yf - expr)
        assert dif == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

        yf = lccde.solve_free('initial_rest')
        assert yf == sp.S.Zero
        assert lccde.check_solution(sp.S.Zero, yf)

    def test_LCCDE_solve_forced(self):
        A = [1, -sp.Rational(3, 4), sp.Rational(1, 8)]
        B = [1, -sp.Rational(3, 8)]
        lccde = LCCDE(B, A)
        yh, _ = lccde.solve_homogeneous()
        Ca = sp.Wild("Ca")
        Cb = sp.Wild("Cb")
        roots = list(lccde.y_roots())
        expr = Ca * roots[0] ** lccde.iv + Cb * roots[1] ** lccde.iv
        match = yh.match(expr)
        assert match is not None

        # TODO ac
        # yh = eq.solve_forced(UnitDelta(eq.iv), ac={0: 1, -1: -2})
        # expr = -2 * roots[0] ** eq.iv + 3 * roots[1] ** eq.iv
        # dif = sp.simplify(yh - expr)
        # assert dif == sp.S.Zero

        # TODO solve_forced(u[n])
        # a1 = sp.Symbol('a1')
        # B = [1]
        # A = [1, a1]
        # eq = LCCDE(B, A)
        # y = eq.solve_forced(UnitStep(n))

