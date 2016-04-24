from fractions import Fraction
from skdsp.signal.discrete import Sine, DiscreteFunctionSignal, DiscreteMixin
from skdsp.signal.signal import Signal, FunctionSignal
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest


class SineTest(unittest.TestCase):

    def test_constructor(self):
        c = Sine()
        self.assertIsInstance(c, Signal)
        self.assertIsInstance(c, FunctionSignal)
        self.assertIsInstance(c, DiscreteMixin)
        self.assertIsInstance(c, DiscreteFunctionSignal)
        self.assertTrue(is_discrete(c))
        self.assertFalse(is_continuous(c))
        c1 = Sine(sp.S.Pi/4, sp.S.Pi/6)
        c2 = Sine().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)

    def test_eval_sample(self):
        c = Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c.eval(0), np.sin(np.pi/6))
        self.assertAlmostEqual(c.eval(1), np.sin(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c.eval(-1), np.sin(-np.pi/4 + np.pi/6))
        c = Sine(0, sp.S.Pi/2)
        self.assertEqual(c.eval(0), 1.0)

    def test_eval_range(self):
        c = Sine(sp.S.Pi/4, sp.S.Pi/6)
        expected = np.sin(np.arange(0, 2)*np.pi/4 + np.pi/6)
        actual = c.eval(np.arange(0, 2))
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.sin(np.arange(-1, 2)*np.pi/4 + np.pi/6)
        actual = c.eval(np.arange(-1, 2))
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.sin(np.arange(-4, 1, 2)*np.pi/4 + np.pi/6)
        actual = c.eval(np.arange(-4, 1, 2))
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.sin(np.arange(3, -2, -2)*np.pi/4 + np.pi/6)
        actual = c.eval(np.arange(3, -2, -2))
        np.testing.assert_array_almost_equal(expected, actual)

    def test_getitem_scalar(self):
        c = Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c[0], np.sin(np.pi/6))
        self.assertAlmostEqual(c[1], np.sin(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c[-1], np.sin(-np.pi/4 + np.pi/6))

    def test_getitem_slice(self):
        c = Sine(sp.S.Pi/4, sp.S.Pi/6)
        expected = np.sin(np.arange(0, 2)*np.pi/4 + np.pi/6)
        actual = c[0:2]
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.sin(np.arange(-1, 2)*np.pi/4 + np.pi/6)
        actual = c[-1:2]
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.sin(np.arange(-4, 1, 2)*np.pi/4 + np.pi/6)
        actual = c[-4:1:2]
        np.testing.assert_array_almost_equal(expected, actual)
        expected = np.sin(np.arange(3, -2, -2)*np.pi/4 + np.pi/6)
        actual = c[3:-2:-2]
        np.testing.assert_array_almost_equal(expected, actual)

    def test_dtype(self):
        c = Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_real(c))
        self.assertEqual(c.dtype, np.float_)
        c.dtype = np.complex_
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_

    def test_name(self):
        c = Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')

    def test_xvar(self):
        c = Sine().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'sin(n - 3)')
        c.xvar = sp.symbols('m', integer=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'sin(m - 3)')

    def test_period(self):
        c = Sine(sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = Sine(3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16)
        c = Sine(3/8)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = Sine(1/4)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = Sine(3/2)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        for M in np.arange(5, 105, 5):
            f = Fraction(200, M)
            c = Sine(M/100*sp.S.Pi)
            self.assertEqual(c.period, f.numerator)

    def test_frequency(self):
        c = Sine(sp.S.Pi/4)
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = Sine(3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = Sine(3/8)
        self.assertEqual(c.frequency, 3/8)
        c = Sine(1/4)
        self.assertEqual(c.frequency, 1/4)
        c = Sine(3/2)
        self.assertEqual(c.frequency, 3/2)
        c = Sine(sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = Sine(sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = Sine(sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = Sine(sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)

if __name__ == "__main__":
    unittest.main()
