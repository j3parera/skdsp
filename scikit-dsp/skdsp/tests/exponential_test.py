import math
from fractions import Fraction
from skdsp.signal.discrete import DiscreteFunctionSignal, DiscreteMixin, \
    Exponential
from skdsp.signal.signal import Signal, FunctionSignal
from skdsp.signal.util import is_discrete, is_continuous, is_complex
import numpy as np
import sympy as sp
import unittest


class ExponentialTest(unittest.TestCase):

#     def test_00(self):
#         N = 16
#         k = N
#         s = 0
#         for k0 in range(0, k):
#             s += Exponential(sp.exp(sp.I*2*sp.S.Pi*k0/N))
#         s = s/N
#         ns = np.arange(-25, 25)
#         print(np.real_if_close(s[ns]))

    def test_constructor(self):
        c = Exponential()
        self.assertIsInstance(c, Signal)
        self.assertIsInstance(c, FunctionSignal)
        self.assertIsInstance(c, DiscreteMixin)
        self.assertIsInstance(c, DiscreteFunctionSignal)
        self.assertIsInstance(c, Exponential)
        self.assertTrue(is_discrete(c))
        self.assertFalse(is_continuous(c))
        n = sp.symbols('n', integer=True)
        c = Exponential(-1)
        self.assertEqual(c, sp.sympify((-1)**n))
        c = Exponential(0.5)
        self.assertEqual(c, sp.sympify((0.5)**n))
        c = Exponential(1+1j)
        self.assertEqual(c, sp.sympify((1+1j)**n))

    def test_eval_sample(self):
        base = np.arange(-2, 2.5, 0.5)
        exps = np.arange(-2, 3, 1)
        for b in base:
            for e in exps:
                c = Exponential(b)
                actual = c.eval(e)
                expected = np.power(b, e)
                if b == 0 and e < 0:
                    self.assertTrue(math.isnan(actual.real))
                    self.assertTrue(math.isnan(actual.imag))
                else:
                    np.testing.assert_almost_equal(actual, expected)

    def test_eval_range(self):
        c = Exponential(2)
        n = np.arange(0, 2)
        expected = np.power(2, n)
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-1, 2)
        expected = np.power(2, n)
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-4, 1, 2)
        expected = np.power(2, n)
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(3, -2, -2)
        expected = np.power(2, n)
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_getitem_scalar(self):
        base = np.arange(-2, 2.5, 0.5)
        exps = np.arange(-2, 3, 1)
        for b in base:
            for e in exps:
                c = Exponential(b)
                actual = c[e]
                expected = np.power(b, e)
                if b == 0 and e < 0:
                    self.assertTrue(math.isnan(actual.real))
                    self.assertTrue(math.isnan(actual.imag))
                else:
                    np.testing.assert_almost_equal(actual, expected)

    def test_getitem_slice(self):
        c = Exponential(2)
        n = np.arange(0, 2)
        expected = np.power(2, n)
        actual = c[n]
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-1, 2)
        expected = np.power(2, n)
        actual = c[n]
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-4, 1, 2)
        expected = np.power(2, n)
        actual = c[n]
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(3, -2, -2)
        expected = np.power(2, n)
        actual = c[n]
        np.testing.assert_array_almost_equal(expected, actual)

    def test_dtype(self):
        for b in np.arange(-2, 0):
            c = Exponential(b)
            self.assertTrue(is_complex(c))
            with self.assertRaises(ValueError):
                c.dtype = np.int_
        for b in np.arange(0, 3):
            c = Exponential(b)
            self.assertFalse(is_complex(c))
            with self.assertRaises(ValueError):
                c.dtype = np.int_
        c = Exponential(1+1j)
        self.assertTrue(is_complex(c))
        c = Exponential(3*sp.exp(sp.I*sp.S.Pi/4))
        self.assertTrue(is_complex(c))

    def test_name(self):
        c = Exponential(3)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')

    def test_xvar(self):
        c = Exponential(0.5).delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), '0.5**(n - 3)')
        c.xvar = sp.symbols('m', integer=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), '0.5**(m - 3)')

    def test_period(self):
        c = Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = Exponential(sp.exp(sp.I*3*sp.S.Pi/8))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16)
        c = Exponential(sp.exp(sp.I*3/8))
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = Exponential(sp.exp(sp.I*1/4))
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = Exponential(sp.exp(sp.I*3/2))
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        c = Exponential(1)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 1)
        for M in np.arange(5, 205, 5):
            f = Fraction(200, M)
            c = Exponential(sp.exp(sp.I*M/100*sp.S.Pi))
            self.assertEqual(c.period, f.numerator)
            c = Exponential(1.0*sp.exp(sp.I*sp.S.Pi*M/100))
            self.assertEqual(c.period, f.numerator)

        c = Exponential(4*sp.exp(sp.I*sp.S.Pi/4))
        self.assertFalse(c.is_periodic())
        c = Exponential(1+1j)
        self.assertFalse(c.is_periodic())
        c = Exponential(1+sp.I)
        self.assertFalse(c.is_periodic())

        c = Exponential(sp.sqrt(2)/2 + sp.sqrt(2)*sp.I/2)
        self.assertTrue(c.is_periodic())

    def test_frequency(self):
        c = Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = Exponential(sp.exp(sp.I*3*sp.S.Pi/8))
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = Exponential(sp.exp(sp.I*3/8))
        self.assertEqual(c.frequency, 3/8)
        c = Exponential(sp.exp(sp.I*1/4))
        self.assertEqual(c.frequency, 1/4)
        c = Exponential(sp.exp(sp.I*3/2))
        self.assertEqual(c.frequency, 3/2)
        c = Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertEqual(c.phase_offset, 0)

if __name__ == "__main__":
    unittest.main()
