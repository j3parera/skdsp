import pytest
import sympy as sp

from skdsp.util.lccde import LCCDE
from skdsp.signal.functions import UnitStep, UnitDelta, stepsimp

class Test_LCCDE(object):

    def test_LCCDE_constructor(self):
        A = [1, -sp.S(3) / 4, sp.S(1) / 8]
        B = [2, -sp.S(3) / 4]
        eq = LCCDE(B, A)
        assert eq is not None
        assert eq.iv == sp.Symbol('n', integer=True)
        assert eq.x == sp.Function('x')(eq.iv)
        assert eq.y == sp.Function('y')(eq.iv)
        assert eq.B == B
        assert eq.A == A
        assert eq.order == 2

    def test_LCCDE_solve_forced(self):
        A = [1, -sp.S(3) / 4, sp.S(1) / 8]
        B = [2, -sp.S(3) / 4]
        eq = LCCDE(B, A)
        yh, _ = eq.solve_homogeneous()
        Ca = sp.Wild('Ca')
        Cb = sp.Wild('Cb')
        roots = sp.Poly(A, sp.Dummy('x')).all_roots()
        expr = Ca * roots[0] ** eq.iv + Cb * roots[1] ** eq.iv
        match = yh.match(expr)
        assert match is not None

        # TODO
        # yh = eq.solve_forced(UnitDelta(eq.iv), ac={0: 1, -1: -2})
        # expr = -2 * roots[0] ** eq.iv + 3 * roots[1] ** eq.iv
        # dif = sp.simplify(yh - expr)
        # assert dif == sp.S.Zero

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

    def test_LCDDE_from_expression(self):
        y = sp.Function('y')
        x = sp.Function('x')
        n = sp.Symbol('n', integer=True)
        ed = sp.Eq(y(n) - sp.S(3)/4*y(n-1) + sp.S(1)/8*y(n-2), 2*x(n) - sp.S(3)/4*x(n-1))
        eq = LCCDE.from_expression(ed, x(n), y(n))
        assert eq.A == [1, -sp.S(3) / 4, sp.S(1) / 8]
        assert eq.B == [2, -sp.S(3) / 4]
        ed2 = eq.as_expression
        y1 = sp.solve(ed, y(n))
        y2 = sp.solve(ed2, y(n))
        assert y1 == y2