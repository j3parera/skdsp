from cmath import rect
from fractions import Fraction
from skdsp.signal.discrete import DiscreteFunctionSignal, DiscreteMixin, \
    ComplexSinusoid, Exponential
from skdsp.signal.signal import Signal, FunctionSignal
from skdsp.signal.util import is_discrete, is_continuous, is_complex
import numpy as np
import sympy as sp
import unittest


class ComplexSinusoidTest(unittest.TestCase):

    def test_constructor(self):
        c = ComplexSinusoid()
        self.assertIsInstance(c, Signal)
        self.assertIsInstance(c, FunctionSignal)
        self.assertIsInstance(c, DiscreteMixin)
        self.assertIsInstance(c, DiscreteFunctionSignal)
        self.assertTrue(is_discrete(c))
        self.assertFalse(is_continuous(c))
        c1 = ComplexSinusoid(3, sp.S.Pi/4, sp.S.Pi/6)
        c2 = 3*ComplexSinusoid().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)

    def test_eval_sample(self):
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        As = 3*sp.exp(sp.I*phi0s)
        An = rect(3, phi0n)
        c = ComplexSinusoid(As, omega0s)
        np.testing.assert_almost_equal(c.eval(0), An)
        np.testing.assert_almost_equal(c.eval(1), An*(np.exp(1j*omega0n)))
        np.testing.assert_almost_equal(c.eval(-1), An*(np.exp(-1j*omega0n)))
        c = ComplexSinusoid(1, 0, sp.S.Pi/2)
        np.testing.assert_almost_equal(c.eval(0), 1j)

    def test_eval_range(self):
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        c = ComplexSinusoid(1, omega0s, phi0s)
        n = np.arange(0, 2)
        expected = np.exp(1j*(n*omega0n + phi0n))
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-1, 2)
        expected = np.exp(1j*(n*omega0n + phi0n))
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-4, 1, 2)
        expected = np.exp(1j*(n*omega0n + phi0n))
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(3, -2, -2)
        expected = np.exp(1j*(n*omega0n + phi0n))
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_getitem_scalar(self):
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        An = rect(1, phi0n)
        c = ComplexSinusoid(1, omega0s, phi0s)
        self.assertAlmostEqual(c[0], An)
        self.assertAlmostEqual(c[1], An*np.exp(1j*omega0n))
        self.assertAlmostEqual(c[-1], An*np.exp(-1j*omega0n))

    def test_getitem_slice(self):
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        An = rect(1, phi0n)
        c = ComplexSinusoid(1, omega0s, phi0s)
        n = np.arange(0, 2)
        expected = An*np.exp(1j*omega0n*n)
        np.testing.assert_array_almost_equal(expected, c[n])
        n = np.arange(-1, 2)
        expected = An*np.exp(1j*omega0n*n)
        np.testing.assert_array_almost_equal(expected, c[n])
        n = np.arange(-4, 1, 2)
        expected = An*np.exp(1j*omega0n*n)
        np.testing.assert_array_almost_equal(expected, c[-4:1:2])
        n = np.arange(3, -2, -2)
        expected = An*np.exp(1j*omega0n*n)
        np.testing.assert_array_almost_equal(expected, c[3:-2:-2])

    def test_dtype(self):
        c = ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_

    def test_name(self):
        c = ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')

    def test_xvar(self):
        c = ComplexSinusoid().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'exp(j*(n - 3))')
        c.xvar = sp.symbols('m', integer=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'exp(j*(m - 3))')

    def test_period(self):
        c = ComplexSinusoid(1, sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = ComplexSinusoid(1, 3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16)
        c = ComplexSinusoid(1, 3/8)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ComplexSinusoid(1, 1/4)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ComplexSinusoid(1, 3/2)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        for M in np.arange(5, 105, 5):
            f = Fraction(200, M)
            c = ComplexSinusoid(1, M/100*sp.S.Pi)
            self.assertEqual(c.period, f.numerator)

    def test_frequency(self):
        c = ComplexSinusoid(1, sp.S.Pi/4)
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = ComplexSinusoid(1, 3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = ComplexSinusoid(1, 3/8)
        self.assertEqual(c.frequency, 3/8)
        c = ComplexSinusoid(1, 1/4)
        self.assertEqual(c.frequency, 1/4)
        c = ComplexSinusoid(1, 3/2)
        self.assertEqual(c.frequency, 3/2)
        c = ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = ComplexSinusoid(1, sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = ComplexSinusoid(1, sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = ComplexSinusoid(1, sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)

if __name__ == "__main__":
    unittest.main()
