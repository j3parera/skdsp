import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.printer as pt
import skdsp.signal.signal as sg
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest
from fractions import Fraction


class SineTest(unittest.TestCase):

    def test_constructor(self):
        ''' Sine (discrete/continuous): constructors '''
        # seno discreto
        c = ds.Sine()
        self.assertIsInstance(c, sg.Signal)
        self.assertIsInstance(c, sg.FunctionSignal)
        self.assertIsInstance(c, ds.DiscreteFunctionSignal)
        self.assertTrue(is_discrete(c))
        self.assertFalse(is_continuous(c))
        c1 = ds.Sine(sp.S.Pi/4, sp.S.Pi/6)
        c2 = ds.Sine().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)
        # seno continuo
        c = cs.Sine()
        self.assertIsInstance(c, sg.Signal)
        self.assertIsInstance(c, sg.FunctionSignal)
        self.assertIsInstance(c, cs.ContinuousFunctionSignal)
        self.assertFalse(is_discrete(c))
        self.assertTrue(is_continuous(c))
        c1 = ds.Sine(sp.S.Pi/4, sp.S.Pi/6)
        c2 = ds.Sine().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)

    def test_eval_sample(self):
        ''' Sine (discrete/continuous): eval(scalar) '''
        # seno discreto
        c = ds.Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c.eval(0), np.sin(np.pi/6))
        self.assertAlmostEqual(c.eval(1), np.sin(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c.eval(-1), np.sin(-np.pi/4 + np.pi/6))
        c = ds.Sine(0, sp.S.Pi/2)
        self.assertEqual(c.eval(0), 1.0)
        # seno continuo
        c = cs.Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c.eval(0), np.sin(np.pi/6))
        self.assertAlmostEqual(c.eval(1), np.sin(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c.eval(-1), np.sin(-np.pi/4 + np.pi/6))
        c = cs.Sine(0, sp.S.Pi/2)
        self.assertAlmostEqual(c.eval(0), 1.0)

    def test_eval_range(self):
        ''' Sine (discrete/continuous): eval(array) '''
        # seno discreto
        c = ds.Sine(sp.S.Pi/4, sp.S.Pi/6)
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
        # seno continuo
        c = cs.Sine(sp.S.Pi/4, sp.S.Pi/6)
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
        ''' Sine (discrete/continuous): eval[scalar] '''
        # seno discreto
        c = ds.Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c[0], np.sin(np.pi/6))
        self.assertAlmostEqual(c[1], np.sin(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c[-1], np.sin(-np.pi/4 + np.pi/6))
        # seno continuo
        c = cs.Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c[0], np.sin(np.pi/6))
        self.assertAlmostEqual(c[1], np.sin(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c[-1], np.sin(-np.pi/4 + np.pi/6))

    def test_getitem_slice(self):
        ''' Sine (discrete/continuous): eval[:] '''
        # seno discreto
        c = ds.Sine(sp.S.Pi/4, sp.S.Pi/6)
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
        # seno discreto
        c = cs.Sine(sp.S.Pi/4, sp.S.Pi/6)
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
        ''' Sine (discrete/continuous): dtype '''
        # seno discreto
        c = ds.Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_real(c))
        self.assertEqual(c.dtype, np.float_)
        c.dtype = np.complex_
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_
        # seno discreto
        c = cs.Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_real(c))
        self.assertEqual(c.dtype, np.float_)
        c.dtype = np.complex_
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_

    def test_name(self):
        ''' Sine (discrete/continuous): name '''
        # seno discreto
        c = ds.Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')
        # seno continuo
        c = cs.Sine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')

    def test_xvar(self):
        ''' Sine (discrete/continuous): free variable '''
        # seno discreto
        c = ds.Sine().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'sin(n - 3)')
        c.xvar = sp.symbols('m', integer=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'sin(m - 3)')
        # seno continuo
        c = cs.Sine().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'sin(t - 3)')
        c.xvar = sp.symbols('u', real=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'sin(u - 3)')

    def test_period(self):
        ''' Sine (discrete/continuous): period '''
        # seno discreto
        c = ds.Sine(sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = ds.Sine(3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16)
        c = ds.Sine(3/8)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ds.Sine(1/4)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ds.Sine(3/2)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        for M in np.arange(5, 105, 5):
            f = Fraction(200, M)
            c = ds.Sine(M/100*sp.S.Pi)
            self.assertEqual(c.period, f.numerator)
        # seno continuo
        c = cs.Sine(sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = cs.Sine(3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16/3)
        c = cs.Sine(3/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 16*np.pi/3)
        c = cs.Sine(1/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 8*np.pi)
        c = cs.Sine(3/2)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 4*np.pi/3)

    def test_frequency(self):
        ''' Sine (discrete/continuous): frequency '''
        # seno discreto
        c = ds.Sine(sp.S.Pi/4)
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = ds.Sine(3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = ds.Sine(3/8)
        self.assertEqual(c.frequency, 3/8)
        c = ds.Sine(1/4)
        self.assertEqual(c.frequency, 1/4)
        c = ds.Sine(3/2)
        self.assertEqual(c.frequency, 3/2)
        c = ds.Sine(sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = ds.Sine(sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = ds.Sine(sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = ds.Sine(sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        # seno continuo
        c = cs.Sine(sp.S.Pi/4)
        self.assertEqual(c.frequency, 1/8)
        self.assertEqual(c.angular_frequency, sp.S.Pi/4)
        c = cs.Sine(3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3/16)
        self.assertEqual(c.angular_frequency, 3*sp.S.Pi/8)
        c = cs.Sine(3/8)
        self.assertEqual(c.frequency.evalf(), 3/(16*np.pi))
        self.assertEqual(c.angular_frequency, 3/8)
        c = cs.Sine(1/4)
        self.assertEqual(c.frequency.evalf(), 1/(8*np.pi))
        self.assertEqual(c.angular_frequency, 1/4)
        c = cs.Sine(3/2)
        self.assertEqual(c.angular_frequency, 3/2)
        c = cs.Sine(sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = cs.Sine(sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = cs.Sine(sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = cs.Sine(sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)

    def test_latex(self):
        ''' Sine (discrete/continuous): latex '''
        # seno discreto
        d = ds.Sine()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\sin\left[n\right]$')
        d = ds.Sine(sp.S.Pi/4)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\sin\left[\pi n / 4\right]$')
        d = ds.Sine(sp.S.Pi/4, sp.S.Pi/8)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\sin\left[\pi n / 4 + \pi / 8\right]$')
        # seno continuo
        d = cs.Sine()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\sin\left(t\right)$')
        d = cs.Sine(sp.S.Pi/4)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\sin\left(\pi t / 4\right)$')
        d = cs.Sine(sp.S.Pi/4, sp.S.Pi/8)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\sin\left(\pi t / 4 + \pi / 8\right)$')

if __name__ == "__main__":
    unittest.main()
