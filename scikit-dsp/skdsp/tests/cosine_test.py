import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.printer as pt
import skdsp.signal.signals as sg
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest
from fractions import Fraction


class CosineTest(unittest.TestCase):

    def test_constructor(self):
        ''' Cosine (discrete/continuous): constructors '''
        # coseno discreto
        c = ds.Cosine()
        self.assertIsInstance(c, sg.Signal)
        self.assertIsInstance(c, sg.FunctionSignal)
        self.assertIsInstance(c, ds.DiscreteFunctionSignal)
        self.assertTrue(is_discrete(c))
        self.assertFalse(is_continuous(c))
        c1 = ds.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        c2 = ds.Cosine().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)
        # coseno continuo
        c = cs.Cosine()
        self.assertIsInstance(c, sg.Signal)
        self.assertIsInstance(c, sg.FunctionSignal)
        self.assertIsInstance(c, cs.ContinuousFunctionSignal)
        self.assertFalse(is_discrete(c))
        self.assertTrue(is_continuous(c))
        c1 = ds.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        c2 = ds.Cosine().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)

    def test_eval_sample(self):
        ''' Cosine (discrete/continuous): eval(scalar) '''
        # coseno discreto
        c = ds.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c.eval(0), np.cos(np.pi/6))
        self.assertAlmostEqual(c.eval(1), np.cos(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c.eval(-1), np.cos(-np.pi/4 + np.pi/6))
        c = ds.Cosine(0, sp.S.Pi/2)
        self.assertAlmostEqual(c.eval(0), 0)
        # coseno continuo
        c = cs.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c.eval(0), np.cos(np.pi/6))
        self.assertAlmostEqual(c.eval(1), np.cos(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c.eval(-1), np.cos(-np.pi/4 + np.pi/6))
        c = cs.Cosine(0, sp.S.Pi/2)
        self.assertAlmostEqual(c.eval(0), 0)

    def test_eval_range(self):
        ''' Cosine (discrete/continuous): eval(array) '''
        # coseno discreto
        c = ds.Cosine(sp.S.Pi/4, sp.S.Pi/6)
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
        # coseno continuo
        c = cs.Cosine(sp.S.Pi/4, sp.S.Pi/6)
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
        ''' Cosine (discrete/continuous): eval[scalar] '''
        # coseno discreto
        c = ds.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c[0], np.cos(np.pi/6))
        self.assertAlmostEqual(c[1], np.cos(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c[-1], np.cos(-np.pi/4 + np.pi/6))
        # coseno continuo
        c = cs.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c[0], np.cos(np.pi/6))
        self.assertAlmostEqual(c[1], np.cos(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c[-1], np.cos(-np.pi/4 + np.pi/6))

    def test_getitem_slice(self):
        ''' Cosine (discrete/continuous): eval[:] '''
        # coseno discreto
        c = ds.Cosine(sp.S.Pi/4, sp.S.Pi/6)
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
        # coseno discreto
        c = cs.Cosine(sp.S.Pi/4, sp.S.Pi/6)
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
        ''' Cosine (discrete/continuous): dtype '''
        # coseno discreto
        c = ds.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_real(c))
        self.assertEqual(c.dtype, np.float_)
        c.dtype = np.complex_
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_
        # coseno discreto
        c = cs.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_real(c))
        self.assertEqual(c.dtype, np.float_)
        c.dtype = np.complex_
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_

    def test_name(self):
        ''' Cosine (discrete/continuous): name '''
        # coseno discreto
        c = ds.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')
        # coseno continuo
        c = cs.Cosine(sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')

    def test_xvar(self):
        ''' Cosine (discrete/continuous): free variable '''
        # coseno discreto
        c = ds.Cosine().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'cos(n - 3)')
        c.xvar = sp.symbols('m', integer=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'cos(m - 3)')
        # coseno continuo
        c = cs.Cosine().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'cos(t - 3)')
        c.xvar = sp.symbols('u', real=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'cos(u - 3)')

    def test_period(self):
        ''' Cosine (discrete/continuous): period '''
        # coseno discreto
        c = ds.Cosine(sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = ds.Cosine(3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16)
        c = ds.Cosine(3/8)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ds.Cosine(1/4)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ds.Cosine(3/2)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        for M in np.arange(5, 105, 5):
            f = Fraction(200, M)
            c = ds.Cosine(M/100*sp.S.Pi)
            self.assertEqual(c.period, f.numerator)
        # coseno continuo
        c = cs.Cosine(sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = cs.Cosine(3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16/3)
        c = cs.Cosine(3/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 16*np.pi/3)
        c = cs.Cosine(1/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 8*np.pi)
        c = cs.Cosine(3/2)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 4*np.pi/3)

    def test_frequency(self):
        ''' Cosine (discrete/continuous): frequency '''
        # coseno discreto
        c = ds.Cosine(sp.S.Pi/4)
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = ds.Cosine(3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = ds.Cosine(3/8)
        self.assertEqual(c.frequency, 3/8)
        c = ds.Cosine(1/4)
        self.assertEqual(c.frequency, 1/4)
        c = ds.Cosine(3/2)
        self.assertEqual(c.frequency, 3/2)
        c = ds.Cosine(sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = ds.Cosine(sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = ds.Cosine(sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = ds.Cosine(sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        # coseno continuo
        c = cs.Cosine(sp.S.Pi/4)
        self.assertEqual(c.frequency, 1/8)
        self.assertEqual(c.angular_frequency, sp.S.Pi/4)
        c = cs.Cosine(3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3/16)
        self.assertEqual(c.angular_frequency, 3*sp.S.Pi/8)
        c = cs.Cosine(3/8)
        self.assertEqual(c.frequency.evalf(), 3/(16*np.pi))
        self.assertEqual(c.angular_frequency, 3/8)
        c = cs.Cosine(1/4)
        self.assertEqual(c.frequency.evalf(), 1/(8*np.pi))
        self.assertEqual(c.angular_frequency, 1/4)
        c = cs.Cosine(3/2)
        self.assertEqual(c.angular_frequency, 3/2)
        c = cs.Cosine(sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = cs.Cosine(sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = cs.Cosine(sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = cs.Cosine(sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)

    def test_latex(self):
        ''' Cosine (discrete/continuous): latex '''
        # coseno discreto
        d = ds.Cosine()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\cos\left(n\right)$')
        d = ds.Cosine(sp.S.Pi/4)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\cos\left(\pi n / 4\right)$')
        d = ds.Cosine(sp.S.Pi/4, sp.S.Pi/8)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\cos\left(\pi n / 4 + \pi / 8\right)$')
        d = 3*ds.Cosine(sp.S.Pi/4, sp.S.Pi/8)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$3*\cos\left(\pi n / 4 + \pi / 8\right)$')
        # coseno continuo
        d = cs.Cosine()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\cos\left(t\right)$')
        d = cs.Cosine(sp.S.Pi/4)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\cos\left(\pi t / 4\right)$')
        d = cs.Cosine(sp.S.Pi/4, sp.S.Pi/8)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\cos\left(\pi t / 4 + \pi / 8\right)$')


if __name__ == "__main__":
    unittest.main()
