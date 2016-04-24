from fractions import Fraction
from skdsp.signal.discrete import Cosine, DiscreteFunctionSignal, DiscreteMixin
from skdsp.signal.signal import Signal, FunctionSignal
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest


class CosineTest(unittest.TestCase):

    def test_constructor(self):
        c = Cosine()
        self.assertIsInstance(c, Signal)
        self.assertIsInstance(c, FunctionSignal)
        self.assertIsInstance(c, DiscreteMixin)
        self.assertIsInstance(c, DiscreteFunctionSignal)
        self.assertTrue(is_discrete(c))
        self.assertFalse(is_continuous(c))
        c1 = Cosine(sp.S.Pi/4, sp.S.Pi/6)
        c2 = Cosine().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)

    def test_eval_sample(self):
        c = Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c.eval(0), np.cos(np.pi/6))
        self.assertAlmostEqual(c.eval(1), np.cos(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c.eval(-1), np.cos(-np.pi/4 + np.pi/6))
        c = Cosine(0, sp.S.Pi/2)
        self.assertAlmostEqual(c.eval(0), 0)

    def test_eval_range(self):
        c = Cosine(sp.S.Pi/4, sp.S.Pi/6)
        expected = np.cos(np.arange(0, 2)*np.pi/4 + np.pi/6)
        actual = c.eval(np.arange(0, 2))
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.cos(np.arange(-1, 2)*np.pi/4 + np.pi/6)
        actual = c.eval(np.arange(-1, 2))
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.cos(np.arange(-4, 1, 2)*np.pi/4 + np.pi/6)
        actual = c.eval(np.arange(-4, 1, 2))
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.cos(np.arange(3, -2, -2)*np.pi/4 + np.pi/6)
        actual = c.eval(np.arange(3, -2, -2))
        np.testing.assert_array_almost_equal(expected, actual)

    def test_getitem_scalar(self):
        c = Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c[0], np.cos(np.pi/6))
        self.assertAlmostEqual(c[1], np.cos(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c[-1], np.cos(-np.pi/4 + np.pi/6))

    def test_getitem_slice(self):
        c = Cosine(sp.S.Pi/4, sp.S.Pi/6)
        expected = np.cos(np.arange(0, 2)*np.pi/4 + np.pi/6)
        actual = c[0:2]
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.cos(np.arange(-1, 2)*np.pi/4 + np.pi/6)
        actual = c[-1:2]
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.cos(np.arange(-4, 1, 2)*np.pi/4 + np.pi/6)
        actual = c[-4:1:2]
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.cos(np.arange(3, -2, -2)*np.pi/4 + np.pi/6)
        actual = c[3:-2:-2]
        np.testing.assert_array_almost_equal(expected, actual)

    def test_dtype(self):
        c = Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_real(c))
        self.assertEqual(c.dtype, np.float_)
        c.dtype = np.complex_
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_

    def test_name(self):
        c = Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')

    def test_xvar(self):
        c = Cosine().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'cos(n - 3)')
        c.xvar = sp.symbols('m', integer=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'cos(m - 3)')

    def test_period(self):
        c = Cosine(sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = Cosine(3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16)
        c = Cosine(3/8)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = Cosine(1/4)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = Cosine(3/2)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        for M in np.arange(5, 105, 5):
            f = Fraction(200, M)
            c = Cosine(M/100*sp.S.Pi)
            self.assertEqual(c.period, f.numerator)

    def test_frequency(self):
        c = Cosine(sp.S.Pi/4)
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = Cosine(3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = Cosine(3/8)
        self.assertEqual(c.frequency, 3/8)
        c = Cosine(1/4)
        self.assertEqual(c.frequency, 1/4)
        c = Cosine(3/2)
        self.assertEqual(c.frequency, 3/2)
        c = Cosine(sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = Cosine(sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = Cosine(sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = Cosine(sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)

if __name__ == "__main__":
    unittest.main()
