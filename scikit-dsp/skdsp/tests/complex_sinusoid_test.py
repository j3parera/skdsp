from cmath import rect
import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.printer as pt
import skdsp.signal.signal as sg
from skdsp.signal.util import is_discrete, is_continuous, is_complex
import numpy as np
import sympy as sp
import unittest
from fractions import Fraction


class ComplexSinusoidTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        np.seterr('ignore')

    def test_constructor(self):
        ''' Complex sinusoid (discrete/continuous): constructors '''
        # sinusoide compleja discreta
        c = ds.ComplexSinusoid()
        self.assertIsInstance(c, sg.Signal)
        self.assertIsInstance(c, sg.FunctionSignal)
        self.assertIsInstance(c, ds.DiscreteFunctionSignal)
        self.assertTrue(is_discrete(c))
        self.assertFalse(is_continuous(c))
        c1 = ds.ComplexSinusoid(3, sp.S.Pi/4, sp.S.Pi/6)
        c2 = 3*ds.ComplexSinusoid().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)
        # sinusoide compleja continua
        c = cs.ComplexSinusoid()
        self.assertIsInstance(c, sg.Signal)
        self.assertIsInstance(c, sg.FunctionSignal)
        self.assertIsInstance(c, cs.ContinuousFunctionSignal)
        self.assertFalse(is_discrete(c))
        self.assertTrue(is_continuous(c))
        c1 = cs.ComplexSinusoid(3, sp.S.Pi/4, sp.S.Pi/6)
        c2 = 3*cs.ComplexSinusoid().delay(-sp.S.Pi/6).scale(sp.S.Pi/4)
        self.assertEqual(c1, c2)

    def test_eval_sample(self):
        ''' Complex sinusoid (discrete/continuous): eval(scalar) '''
        # sinusoide compleja discreta
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        As = 3*sp.exp(sp.I*phi0s)
        An = rect(3, phi0n)
        c = ds.ComplexSinusoid(As, omega0s)
        np.testing.assert_almost_equal(c.eval(0), An)
        np.testing.assert_almost_equal(c.eval(1), An*(np.exp(1j*omega0n)))
        np.testing.assert_almost_equal(c.eval(-1), An*(np.exp(-1j*omega0n)))
        c = ds.ComplexSinusoid(1, 0, sp.S.Pi/2)
        np.testing.assert_almost_equal(c.eval(0), 1j)
        # sinusoide compleja continua
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        As = 3*sp.exp(sp.I*phi0s)
        An = rect(3, phi0n)
        c = cs.ComplexSinusoid(As, omega0s)
        np.testing.assert_almost_equal(c.eval(0), An)
        np.testing.assert_almost_equal(c.eval(1), An*(np.exp(1j*omega0n)))
        np.testing.assert_almost_equal(c.eval(-1), An*(np.exp(-1j*omega0n)))
        c = cs.ComplexSinusoid(1, 0, sp.S.Pi/2)
        np.testing.assert_almost_equal(c.eval(0), 1j)

    def test_eval_range(self):
        ''' Complex sinusoid (discrete/continuous): eval(array) '''
        # sinusoide compleja discreta
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        c = ds.ComplexSinusoid(1, omega0s, phi0s)
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
        # sinusoide compleja continua
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        c = cs.ComplexSinusoid(1, omega0s, phi0s)
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
        ''' Complex sinusoid (discrete/continuous): eval[scalar] '''
        # sinusoide compleja discreta
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        An = rect(1, phi0n)
        c = ds.ComplexSinusoid(1, omega0s, phi0s)
        self.assertAlmostEqual(c[0], An)
        self.assertAlmostEqual(c[1], An*np.exp(1j*omega0n))
        self.assertAlmostEqual(c[-1], An*np.exp(-1j*omega0n))
        # sinusoide compleja continua
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        An = rect(1, phi0n)
        c = cs.ComplexSinusoid(1, omega0s, phi0s)
        self.assertAlmostEqual(c[0], An)
        self.assertAlmostEqual(c[1], An*np.exp(1j*omega0n))
        self.assertAlmostEqual(c[-1], An*np.exp(-1j*omega0n))

    def test_getitem_slice(self):
        ''' Complex sinusoid (discrete/continuous): eval[:] '''
        # sinusoide compleja discreta
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        An = rect(1, phi0n)
        c = ds.ComplexSinusoid(1, omega0s, phi0s)
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
        # sinusoide compleja continua
        omega0s = sp.S.Pi/4
        omega0n = np.pi/4
        phi0s = sp.S.Pi/6
        phi0n = np.pi/6
        An = rect(1, phi0n)
        c = cs.ComplexSinusoid(1, omega0s, phi0s)
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
        ''' Complex sinusoid (discrete/continuous): dtype '''
        # sinusoide compleja discreta
        c = ds.ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_
        # sinusoide compleja continua
        c = cs.ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi/6)
        self.assertTrue(is_complex(c))
        self.assertEqual(c.dtype, np.complex_)
        with self.assertRaises(ValueError):
            c.dtype = np.int_

    def test_name(self):
        ''' Complex sinusoid (discrete/continuous): name '''
        # sinusoide compleja discreta
        c = ds.ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')
        # sinusoide compleja continua
        c = cs.ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi/6)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')

    def test_xvar(self):
        ''' Complex sinusoid (discrete/continuous): free variable '''
        # sinusoide compleja discreta
        c = ds.ComplexSinusoid().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'exp(j*(n - 3))')
        c.xvar = sp.symbols('m', integer=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'exp(j*(m - 3))')
        # sinusoide compleja continua
        c = cs.ComplexSinusoid().delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'exp(j*(t - 3))')
        c.xvar = sp.symbols('u', real=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), 'exp(j*(u - 3))')

    def test_period(self):
        ''' Complex sinusoid (discrete/continuous): period '''
        # sinusoide compleja discreta
        c = ds.ComplexSinusoid(1, sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = ds.ComplexSinusoid(1, 3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16)
        c = ds.ComplexSinusoid(1, 3/8)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ds.ComplexSinusoid(1, 1/4)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ds.ComplexSinusoid(1, 3/2)
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        for M in np.arange(5, 105, 5):
            f = Fraction(200, M)
            c = ds.ComplexSinusoid(1, M/100*sp.S.Pi)
            self.assertEqual(c.period, f.numerator)
        # sinusoide compleja continua
        c = cs.ComplexSinusoid(1, sp.S.Pi/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = cs.ComplexSinusoid(1, 3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16/3)
        c = cs.ComplexSinusoid(1, 3/8)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 16*np.pi/3)
        c = cs.ComplexSinusoid(1, 1/4)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 8*np.pi)
        c = cs.ComplexSinusoid(1, 3/2)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 4*np.pi/3)

    def test_frequency(self):
        ''' Complex sinusoid (discrete/continuous): frecuency '''
        # sinusoide compleja discreta
        c = ds.ComplexSinusoid(1, sp.S.Pi/4)
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = ds.ComplexSinusoid(1, 3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = ds.ComplexSinusoid(1, 3/8)
        self.assertEqual(c.frequency, 3/8)
        c = ds.ComplexSinusoid(1, 1/4)
        self.assertEqual(c.frequency, 1/4)
        c = ds.ComplexSinusoid(1, 3/2)
        self.assertEqual(c.frequency, 3/2)
        c = ds.ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = ds.ComplexSinusoid(1, sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = ds.ComplexSinusoid(1, sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = ds.ComplexSinusoid(1, sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        # sinusoide compleja continua
        c = cs.ComplexSinusoid(1, sp.S.Pi/4)
        self.assertEqual(c.angular_frequency, sp.S.Pi/4)
        self.assertEqual(c.frequency, 1/8)
        c = cs.ComplexSinusoid(1, 3*sp.S.Pi/8)
        self.assertEqual(c.angular_frequency, 3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3/16)
        c = cs.ComplexSinusoid(1, 3/8)
        self.assertEqual(c.angular_frequency, 3/8)
        self.assertEqual(c.frequency.evalf(), 3/(16*np.pi))
        c = cs.ComplexSinusoid(1, 1/4)
        self.assertEqual(c.angular_frequency, 1/4)
        self.assertEqual(c.frequency.evalf(), 1/(8*np.pi))
        c = cs.ComplexSinusoid(1, 3/2)
        self.assertEqual(c.angular_frequency, 3/2)
        self.assertEqual(c.frequency.evalf(), 3/(4*np.pi))
        c = cs.ComplexSinusoid(1, sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)
        c = cs.ComplexSinusoid(1, sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase_offset, 1/2)
        c = cs.ComplexSinusoid(1, sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase_offset, 0)
        c = cs.ComplexSinusoid(1, sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase_offset, -sp.S.Pi)

    def test_latex(self):
        ''' Complex sinusoid (discrete/continuous): latex '''
        # sinusoide compleja discreta
        d = ds.ComplexSinusoid().flip()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$1^(- n)$')
        d = ds.ComplexSinusoid(3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$3^n$')
        d = ds.ComplexSinusoid(sp.exp(-sp.I*2*sp.S.Pi/3))
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'${\rm{e}}^{-{{\rm{{j}}}}(2 \pi / 3)n}$')
        d = ds.ComplexSinusoid(sp.exp(sp.I*3/8))
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'${\rm{e}}^{\,{{\rm{{j}}}}\frac{3}{8}n}$')
        # sinusoide compleja continua
        d = cs.ComplexSinusoid().flip()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$1^(- t)$')
        d = cs.ComplexSinusoid(3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$3^t$')
        d = cs.ComplexSinusoid(sp.exp(-sp.I*2*sp.S.Pi/3))
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'${\rm{e}}^{-{{\rm{{j}}}}(2 \pi / 3)t}$')
        d = cs.ComplexSinusoid(sp.exp(sp.I*3/8))
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'${\rm{e}}^{\,{{\rm{{j}}}}\frac{3}{8}t}$')


if __name__ == "__main__":
    unittest.main()
