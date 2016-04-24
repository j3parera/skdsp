import unittest

import sympy as sp
import numpy as np
from skdsp.signal.discrete import Delta, Step, Sinusoid, Sine, Cosine
from skdsp.signal.util import is_discrete, is_continuous
from copy import copy


class MiscTest(unittest.TestCase):

    def test_0zero_cross(self):
        s = Sinusoid(3, sp.S.Pi/8, -sp.S.Pi/4)
        se = s.yexpr
        sv = s.xvar
        solutions = sp.solve(se, sv)
        if not len(solutions) == 0:
            print(solutions)
        print(s.is_periodic(), s.period)

        # KK
        T = sp.Symbol('T')
        print(sp.solve(se-se.subs(sv, sv-T), T))

    def test_maxmin(self):
        s = Sinusoid(3, sp.S.Pi/8, -sp.S.Pi/4)
        self.assertEqual(s.max(), 3)
        self.assertEqual(s.min(), -3)
        self.assertEqual(s.dynamic_range(), 6)
        self.assertAlmostEqual(s.dynamic_range(True).evalf(), 20*np.log10(6))
        s = Delta(-1)
        self.assertEqual(s.max(), 1)
        self.assertEqual(s.min(), 0)
        self.assertEqual(s.dynamic_range(), 1)
        self.assertAlmostEqual(s.dynamic_range(True).evalf(), 20*np.log10(1))
        s = Step(1)
        self.assertEqual(s.max(), 1)
        self.assertEqual(s.min(), 0)
        self.assertEqual(s.dynamic_range(), 1)
        self.assertAlmostEqual(s.dynamic_range(True).evalf(), 20*np.log10(1))
        s = Sine(sp.S.Pi/8, -sp.S.Pi/4)
        self.assertEqual(s.max(), 1)
        self.assertEqual(s.min(), -1)
        self.assertEqual(s.dynamic_range(), 2)
        self.assertAlmostEqual(s.dynamic_range(True).evalf(), 20*np.log10(2))
        s = Cosine(sp.S.Pi/8, -sp.S.Pi/4)
        self.assertEqual(s.max(), 1)
        self.assertEqual(s.min(), -1)
        self.assertEqual(s.dynamic_range(), 2)
        self.assertAlmostEqual(s.dynamic_range(True).evalf(), 20*np.log10(2))

    def test_minmax(self):
        # min/max de expresiones derivables con máximos mínimos finitos
        s = Sinusoid(3, sp.S.Pi/8, -sp.S.Pi/4)
        se = s.yexpr
        sv = s.xvar
        sd = se.diff(sv)
        solutions = sp.solve(sd)
        solv = [se.subs(sv, sol) for sol in solutions]
        if not len(solv) == 0:
            print(sp.Max(*solv))
            print(sp.Min(*solv))
        # self.assertTrue(False)

    def test_convolve(self):
        x = Delta()
        y = Step()
        n, k = sp.symbols('n, k', integer=True)
        xx = copy(x)
        xx.xvar = n-k
        yy = copy(y)
        yy.xvar = k
        z = sp.Sum(xx.yexpr * yy.yexpr, (k, -sp.oo, sp.oo))
        print(str(x) + ' * ' + str(y) + ' =')
        print(z.doit())

    def test_discrete(self):
        self.assertTrue(is_discrete(Delta()))

    def test_continuous(self):
        self.assertFalse(is_continuous(Delta()))


if __name__ == "__main__":
    unittest.main()
