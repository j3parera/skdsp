import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.printer as pt
import skdsp.signal.signals as sg
from skdsp.signal.util import is_discrete, is_continuous, is_complex
import numpy as np
import sympy as sp
import unittest
from fractions import Fraction


class ExponentialTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        np.seterr('ignore')

    def test_constructor(self):
        ''' Exponential (discrete/continuous): constructors '''
        # exponencial discreta
        c = ds.Exponential()
        self.assertIsInstance(c, sg.Signal)
        self.assertIsInstance(c, sg.FunctionSignal)
        self.assertIsInstance(c, ds.DiscreteFunctionSignal)
        self.assertIsInstance(c, ds.Exponential)
        self.assertTrue(is_discrete(c))
        self.assertFalse(is_continuous(c))
        n = sp.symbols('n', integer=True)
        c = ds.Exponential(-1)
        self.assertEqual(c, sp.sympify((-1)**n))
        c = ds.Exponential(0.5)
        self.assertEqual(c, sp.sympify((0.5)**n))
        c = ds.Exponential(1+1j)
        self.assertEqual(c, sp.sympify((1+1j)**n))
        # exponencial continua
        c = cs.Exponential()
        self.assertIsInstance(c, sg.Signal)
        self.assertIsInstance(c, sg.FunctionSignal)
        self.assertIsInstance(c, cs.ContinuousFunctionSignal)
        self.assertIsInstance(c, cs.Exponential)
        self.assertFalse(is_discrete(c))
        self.assertTrue(is_continuous(c))
        t = sp.symbols('t', integer=True)
        c = cs.Exponential(-1)
        self.assertEqual(c, sp.sympify((-1)**t))
        c = cs.Exponential(0.5)
        self.assertEqual(c, sp.sympify((0.5)**t))
        c = cs.Exponential(1+1j)
        self.assertEqual(c, sp.sympify((1+1j)**t))

    def test_eval_sample(self):
        ''' Exponential (discrete/continuous): eval(scalar) '''
        # exponencial discreta
        base = np.arange(-2, 2.5, 0.5)
        exps = np.arange(-2, 3, 1)
        for b in base:
            for e in exps:
                c = ds.Exponential(b)
                actual = c.eval(e)
                expected = np.power(b, e)
                if b == 0 and e < 0:
                    self.assertTrue(np.isnan(actual.real))
                    self.assertTrue(np.isnan(actual.imag))
                else:
                    np.testing.assert_almost_equal(actual, expected)
        # exponencial continua
        base = np.arange(-2, 2.5, 0.5)
        exps = np.arange(-2, 3, 1)
        for b in base:
            for e in exps:
                c = cs.Exponential(b)
                actual = c.eval(e)
                expected = np.power(b, e)
                if b == 0 and e < 0:
                    self.assertTrue(np.isnan(actual.real))
                    self.assertTrue(np.isnan(actual.imag))
                else:
                    np.testing.assert_almost_equal(actual, expected)

    def test_eval_range(self):
        ''' Exponential (discrete/continuous): eval(array) '''
        # exponencial discreta
        c = ds.Exponential(2)
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
        # exponencial continua
        c = cs.Exponential(2)
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
        ''' Exponential (discrete/continuous): eval[scalar] '''
        # exponencial discreta
        base = np.arange(-2, 2.5, 0.5)
        exps = np.arange(-2, 3, 1)
        for b in base:
            for e in exps:
                c = ds.Exponential(b)
                actual = c[e]
                expected = np.power(b, e)
                if b == 0 and e < 0:
                    self.assertTrue(np.isnan(actual.real))
                    self.assertTrue(np.isnan(actual.imag))
                else:
                    np.testing.assert_almost_equal(actual, expected)
        # exponencial continua
        base = np.arange(-2, 2.5, 0.5)
        exps = np.arange(-2, 3, 1)
        for b in base:
            for e in exps:
                c = cs.Exponential(b)
                actual = c[e]
                expected = np.power(b, e)
                if b == 0 and e < 0:
                    self.assertTrue(np.isnan(actual.real))
                    self.assertTrue(np.isnan(actual.imag))
                else:
                    np.testing.assert_almost_equal(actual, expected)

    def test_getitem_slice(self):
        ''' Exponential (discrete/continuous): eval[:] '''
        # exponencial discreta
        c = ds.Exponential(2)
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
        # exponencial continua
        c = cs.Exponential(2)
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
        ''' Exponential (discrete/continuous): eval[:] '''
        # exponencial discreta
        for b in np.arange(-2, 0):
            c = ds.Exponential(b)
            self.assertTrue(is_complex(c))
            with self.assertRaises(ValueError):
                c.dtype = np.int_
        for b in np.arange(0, 3):
            c = ds.Exponential(b)
            self.assertFalse(is_complex(c))
            with self.assertRaises(ValueError):
                c.dtype = np.int_
        c = ds.Exponential(1+1j)
        self.assertTrue(is_complex(c))
        c = ds.Exponential(3*sp.exp(sp.I*sp.S.Pi/4))
        self.assertTrue(is_complex(c))
        # exponencial continua
        for b in np.arange(-2, 0):
            c = cs.Exponential(b)
            self.assertTrue(is_complex(c))
            with self.assertRaises(ValueError):
                c.dtype = np.int_
        for b in np.arange(0, 3):
            c = cs.Exponential(b)
            self.assertFalse(is_complex(c))
            with self.assertRaises(ValueError):
                c.dtype = np.int_
        c = cs.Exponential(1+1j)
        self.assertTrue(is_complex(c))
        c = cs.Exponential(3*sp.exp(sp.I*sp.S.Pi/4))
        self.assertTrue(is_complex(c))

    def test_name(self):
        ''' Exponential (discrete/continuous): name '''
        # exponencial discreta
        c = ds.Exponential(3)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')
        # exponencial continua
        c = cs.Exponential(3)
        self.assertEqual(c.name, 'x')
        c.name = 'z'
        self.assertEqual(c.name, 'z')

    def test_xvar(self):
        ''' Exponential (discrete/continuous): free variable '''
        # exponencial discreta
        c = ds.Exponential(0.5).delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), '0.5**(n - 3)')
        c.xvar = sp.symbols('m', integer=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), '0.5**(m - 3)')
        # exponencial continua
        c = cs.Exponential(0.5).delay(3)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), '0.5**(t - 3)')
        c.xvar = sp.symbols('u', real=True)
        self.assertEqual(c.name, 'x')
        self.assertEqual(str(c), '0.5**(u - 3)')

    def test_period(self):
        ''' Exponential (discrete/continuous): period '''
        # exponencial discreta
        c = ds.Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = ds.Exponential(sp.exp(sp.I*3*sp.S.Pi/8))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16)
        c = ds.Exponential(sp.exp(sp.I*3/8))
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ds.Exponential(sp.exp(sp.I*1/4))
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        c = ds.Exponential(sp.exp(sp.I*3/2))
        self.assertFalse(c.is_periodic())
        self.assertEqual(c.period, np.Inf)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        c = ds.Exponential(1)
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 1)
        for M in np.arange(5, 205, 5):
            f = Fraction(200, M)
            c = ds.Exponential(sp.exp(sp.I*M/100*sp.S.Pi))
            self.assertEqual(c.period, f.numerator)
            c = ds.Exponential(1.0*sp.exp(sp.I*sp.S.Pi*M/100))
            self.assertEqual(c.period, f.numerator)

        c = ds.Exponential(4*sp.exp(sp.I*sp.S.Pi/4))
        self.assertFalse(c.is_periodic())
        c = ds.Exponential(1+1j)
        self.assertFalse(c.is_periodic())
        c = ds.Exponential(1+sp.I)
        self.assertFalse(c.is_periodic())
        c = ds.Exponential(sp.sqrt(2)/2 + sp.sqrt(2)*sp.I/2)
        self.assertTrue(c.is_periodic())
        # exponencial continua
        c = cs.Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 8)
        c = cs.Exponential(sp.exp(sp.I*3*sp.S.Pi/8))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period, 16/3)
        c = cs.Exponential(sp.exp(sp.I*3/8))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 16*np.pi/3)
        c = cs.Exponential(sp.exp(sp.I*1/4))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 8*np.pi)
        c = cs.Exponential(sp.exp(sp.I*3/2))
        self.assertTrue(c.is_periodic())
        self.assertEqual(c.period.evalf(), 4*np.pi/3)
        c = cs.Exponential(4*sp.exp(sp.I*sp.S.Pi/4))
        self.assertFalse(c.is_periodic())
        c = cs.Exponential(1+1j)
        self.assertFalse(c.is_periodic())
        c = cs.Exponential(1+sp.I)
        self.assertFalse(c.is_periodic())
        c = cs.Exponential(sp.sqrt(2)/2 + sp.sqrt(2)*sp.I/2)
        self.assertTrue(c.is_periodic())

    def test_frequency(self):
        ''' Exponential (discrete/continuous): frequency '''
        # exponencial discreta
        c = ds.Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = ds.Exponential(sp.exp(sp.I*3*sp.S.Pi/8))
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = ds.Exponential(sp.exp(sp.I*3/8))
        self.assertEqual(c.frequency, 3/8)
        c = ds.Exponential(sp.exp(sp.I*1/4))
        self.assertEqual(c.frequency, 1/4)
        c = ds.Exponential(sp.exp(sp.I*3/2))
        self.assertEqual(c.frequency, 3/2)
        c = ds.Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertEqual(c.phase_offset, 0)
        # exponencial continua
        c = cs.Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertEqual(c.angular_frequency, sp.S.Pi/4)
        self.assertEqual(c.frequency, 1/8)
        c = cs.Exponential(sp.exp(sp.I*3*sp.S.Pi/8))
        self.assertEqual(c.angular_frequency, 3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3/16)
        c = cs.Exponential(sp.exp(sp.I*3/8))
        self.assertEqual(c.angular_frequency, 3/8)
        self.assertEqual(c.frequency.evalf(), 3/(16*np.pi))
        c = cs.Exponential(sp.exp(sp.I*1/4))
        self.assertEqual(c.angular_frequency, 1/4)
        self.assertEqual(c.frequency.evalf(), 1/(8*np.pi))
        c = cs.Exponential(sp.exp(sp.I*3/2))
        self.assertEqual(c.angular_frequency, 3/2)
        self.assertEqual(c.frequency.evalf(), 3/(4*np.pi))
        c = cs.Exponential(sp.exp(sp.I*sp.S.Pi/4))
        self.assertEqual(c.phase_offset, 0)

    def test_latex(self):
        ''' Exponential (discrete/continuous): latex '''
        # exponencial discreta
        d = ds.Exponential().flip()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$1^(- n)$')
        d = ds.Exponential(3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$3^n$')
        d = ds.Exponential(sp.exp(-sp.I*2*sp.S.Pi/3))
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'${\rm{e}}^{\,{\rm{j}}(- 2 \pi / 3)n}$')
        d = ds.Exponential(sp.exp(sp.I*3/8))
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'${\rm{e}}^{\,{\rm{j}}\frac{3}{8}n}$')
        # exponencial continua
        d = cs.Exponential().flip()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$1^(- t)$')
        d = cs.Exponential(3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$3^t$')
        d = cs.Exponential(sp.exp(-sp.I*2*sp.S.Pi/3))
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'${\rm{e}}^{\,{\rm{j}}(- 2 \pi / 3)t}$')
        d = cs.Exponential(sp.exp(sp.I*3/8))
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'${\rm{e}}^{\,{\rm{j}}\frac{3}{8}t}$')

if __name__ == "__main__":
    unittest.main()
