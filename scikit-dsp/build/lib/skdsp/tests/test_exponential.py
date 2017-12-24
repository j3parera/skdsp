from fractions import Fraction
from skdsp.signal._signal import _Signal, _FunctionSignal
import numpy as np
import skdsp.signal.discrete as ds
from skdsp.signal.printer import latex
import sympy as sp
import unittest


class ExponentialTest(unittest.TestCase):

    def test_constructor(self):
        ''' Exponential (discrete): constructors '''
        # exponencial discreta
        c = ds.Exponential()
        self.assertIsInstance(c, _Signal)
        self.assertIsInstance(c, _FunctionSignal)
        self.assertIsInstance(c, ds.DiscreteFunctionSignal)
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        n = sp.symbols('n', integer=True)
        c = ds.Exponential(1, -1)
        self.assertEqual(c.yexpr, sp.sympify((-1)**n))
        c = ds.Exponential(2, 0.5)
        self.assertEqual(c.yexpr, sp.sympify(2*(0.5)**n))
        c = ds.Exponential(3, 1+1j)
        self.assertEqual(c.yexpr, sp.sympify(3*(1+1j)**n))

    def test_name(self):
        ''' Exponential: name.
        '''
        # exponencial discreta
        c = ds.Exponential(name='y0')
        self.assertEqual(c.name, 'y0')
        self.assertEqual(c.latex_name, 'y_{0}')
        c.name = 'z'
        self.assertEqual(c.name, 'z')
        self.assertEqual(c.latex_name, 'z')
        with self.assertRaises(ValueError):
            c.name = 'x0'
        with self.assertRaises(ValueError):
            c.name = 'y0'
        c = ds.Exponential(name='y0')
        self.assertEqual(c.name, 'y0')
        self.assertEqual(c.latex_name, 'y_{0}')
        del c
        c = ds.Exponential(name='yupi')
        self.assertEqual(c.name, 'yupi')
        self.assertEqual(c.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' Exponential: independent variable and expression.
        '''
        # exponencial discreta
        c = ds.Exponential()
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, c.xvar)
        # shift
        shift = 5
        c = ds.Exponential().shift(shift)
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, c.xvar - shift)
        # flip
        c = ds.Exponential().flip()
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, -c.xvar)
        # shift and flip
        shift = 5
        c = ds.Exponential().shift(shift).flip()
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, -c.xvar - shift)
        # flip and shift
        shift = 5
        c = ds.Exponential().flip().shift(shift)
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, -c.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' Exponential: function expression.
        '''
        # exponencial discreta
        c = ds.Exponential(2, sp.exp(sp.I*10))
        # expresión
        n = ds._DiscreteMixin.default_xvar()
        self.assertEqual(c.yexpr, 2*sp.exp(sp.I*10*n))
        self.assertTrue(np.issubdtype(c.dtype, np.complex))
        self.assertFalse(c.is_real)
        self.assertTrue(c.is_complex)
        self.assertEqual(2*sp.cos(10*n), c.real.yexpr)
        self.assertEqual(2*sp.sin(10*n), c.imag.yexpr)

    def test_dtype(self):
        ''' Exponential (discrete/continuous): eval[:] '''
        # exponencial discreta
        for b in np.arange(-2, 0):
            c = ds.Exponential(1, b)
            self.assertFalse(c.is_complex)
            with self.assertRaises(ValueError):
                c.dtype = np.int_
        for b in np.arange(0, 3):
            c = ds.Exponential(1, b)
            self.assertFalse(c.is_complex)
            with self.assertRaises(ValueError):
                c.dtype = np.int_
        c = ds.Exponential(1, 1+1j)
        self.assertTrue(c.is_complex)
        c = ds.Exponential(1, 3*sp.exp(sp.I*sp.S.Pi/4))
        self.assertTrue(c.is_complex)

    def test_period(self):
        ''' Exponential: period.
        '''
        # exponencial discreta
        c = ds.Exponential()
        self.assertFalse(c.is_periodic)
        self.assertEqual(c.period, sp.oo)
        c = ds.Exponential(1, sp.exp(sp.I*sp.S.Pi/4))
        self.assertTrue(c.is_periodic)
        self.assertEqual(c.period, 8)
        c = ds.Exponential(1, sp.exp(sp.I*3*sp.S.Pi/8))
        self.assertTrue(c.is_periodic)
        self.assertEqual(c.period, 16)
        c = ds.Exponential(1, sp.exp(-sp.I*3*sp.S.Pi/8))
        self.assertTrue(c.is_periodic)
        self.assertEqual(c.period, 16)
        c = ds.Exponential(1, sp.exp(sp.I*0.83*sp.S.Pi))
        self.assertTrue(c.is_periodic)
        self.assertEqual(c.period, 200)
        c = ds.Exponential(1, sp.exp(sp.I*3/8))
        self.assertFalse(c.is_periodic)
        self.assertEqual(c.period, sp.oo)
        c = ds.Exponential(1, sp.exp(sp.I*1/4))
        self.assertFalse(c.is_periodic)
        self.assertEqual(c.period, sp.oo)
        c = ds.Exponential(1, sp.exp(sp.I*3/2))
        self.assertFalse(c.is_periodic)
        self.assertEqual(c.period, sp.oo)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        for M in np.arange(5, 105, 5):
            f = Fraction(200, M)
            c = ds.Exponential(1, sp.exp(sp.I*M/100*sp.S.Pi))
            self.assertEqual(c.period, f.numerator)

    def test_repr_str_latex(self):
        ''' Exponential: repr, str and latex.
        '''
        # exponencial discreta
        c = ds.Exponential(3, sp.exp(sp.I*sp.S.Pi/4))
        # repr
        self.assertEqual(repr(c), 'Exponential(3, exp(j*pi/4))')
        # str
        self.assertEqual(str(c), '3*exp(j*pi*n/4)')
        # latex
        # TODO e --> \mathrm{e}, i -- \mathrm{j}
        self.assertEqual(latex(c, mode='inline'),
                         r'$3 {\rm{e}}^{\,{\rm{j}}(pi / 4)n}$')
        # exponencial discreta
        c = ds.Exponential(-5, sp.exp(sp.I*10))
        # repr
        self.assertEqual(repr(c), 'Exponential(-5, exp(10*j))')
        # str
        self.assertEqual(str(c), '-5*exp(10*j*n)')
        # latex
        self.assertEqual(latex(c, mode='inline'),
                         r'- 5 e^{10 i n}')
        c = ds.Exponential(1, -1)
        # repr
        self.assertEqual(repr(c), 'Exponential(1, -1)')
        # str
        self.assertEqual(str(c), '(-1)**n')
        # latex
        self.assertEqual(latex(c, mode='inline'), r'\left(-1\right)^{n}')
        c = ds.Exponential(3, sp.exp(sp.I*sp.S.Pi/4)).flip().delay(2)
        # repr
        self.assertEqual(repr(c), 'Exponential(3, exp(j*pi/4))')
        # str
        self.assertEqual(str(c), '3*exp(-j*pi*(n - 2)/4)')
        # latex
        self.assertEqual(latex(c, mode='inline'),
                         r'3 e^{- \frac{i \pi}{4} \left(n - 2\right)}')
        c = ds.Exponential(1+1j, sp.exp(sp.I*sp.S.Pi/4)).flip().delay(2)
        # repr
        self.assertEqual(repr(c), 'Exponential(1.0 + 1.0*j, exp(j*pi/4))')
        # str
        self.assertEqual(str(c), '(1.0 + 1.0*j)*exp(-j*pi*(n - 2)/4)')
        # latex
        self.assertEqual(latex(c, mode='inline'),
                         (r'\left(1.0 + 1.0 i\right) e^{- \frac{i \pi}{4} ' +
                          r'\left(n - 2\right)}'))

    def test_eval_sample(self):
        ''' Exponential (discrete): eval(scalar) '''
        # exponencial discreta
        base = np.arange(-2, 2.5, 0.5)
        exps = np.arange(-2, 3, 1)
        for b in base:
            for e in exps:
                c = ds.Exponential(1, b)
                actual = c.eval(e)
                expected = np.power(b, e)
                if b == 0 and e < 0:
                    self.assertTrue(np.isnan(actual.real))
                    self.assertTrue(np.isnan(actual.imag))
                else:
                    np.testing.assert_almost_equal(actual, expected)
                with self.assertRaises(ValueError):
                    c.eval(0.5)

    def test_eval_range(self):
        ''' Exponential (discrete): eval(range) '''
        # exponencial discreta
        c = ds.Exponential(1, 2)
        n = np.arange(0, 2)
        expected = np.power(2, n)
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-1, 2)
        expected = np.power(2.0, n)
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-4, 1, 2)
        expected = np.power(2.0, n)
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(3, -2, -2)
        expected = np.power(2.0, n)
        actual = c.eval(n)
        np.testing.assert_array_almost_equal(expected, actual)

    def test_getitem_scalar(self):
        ''' Exponential (discrete): eval[scalar] '''
        # exponencial discreta
        base = np.arange(-2, 2.5, 0.5)
        exps = np.arange(-2, 3, 1)
        for b in base:
            for e in exps:
                c = ds.Exponential(1, b)
                actual = c[e]
                expected = np.power(b, e)
                if b == 0 and e < 0:
                    self.assertTrue(np.isnan(actual.real))
                    self.assertTrue(np.isnan(actual.imag))
                else:
                    np.testing.assert_almost_equal(actual, expected)

    def test_getitem_slice(self):
        ''' Exponential (discrete): eval[:] '''
        # exponencial discreta
        c = ds.Exponential(1, 2)
        n = np.arange(0, 2)
        expected = np.power(2, n)
        actual = c[n]
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-1, 2)
        expected = np.power(2.0, n)
        actual = c[n]
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(-4, 1, 2)
        expected = np.power(2.0, n)
        actual = c[n]
        np.testing.assert_array_almost_equal(expected, actual)
        n = np.arange(3, -2, -2)
        expected = np.power(2.0, n)
        actual = c[n]
        np.testing.assert_array_almost_equal(expected, actual)

    def test_generator(self):
        ''' Exponential (discrete): generate '''
        c = ds.Exponential(2, sp.Rational(1, 2))
        with self.assertRaises(ValueError):
            c.generate(0, step=0.1)
        dg = c.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg),
                                      2 * np.power(0.5, np.arange(-3, 2)))
        np.testing.assert_array_equal(next(dg),
                                      2 * np.power(0.5, np.arange(-2, 3)))
        np.testing.assert_array_equal(next(dg),
                                      2 * np.power(0.5, np.arange(-1, 4)))

    def test_flip(self):
        ''' Exponential (discrete): flip '''
        d = ds.Exponential(1, 1).flip()
        np.testing.assert_array_equal(d[-3:3],
                                      np.power(1.0, np.arange(3, -3, -1)))
        d = ds.Exponential(2, 2).flip()
        np.testing.assert_array_equal(d[-3:3],
                                      2*np.power(2.0, np.arange(3, -3, -1)))
        d = ds.Exponential(3, -1).flip()
        np.testing.assert_array_equal(d[-3:3],
                                      3*np.power(-1.0, np.arange(3, -3, -1)))

    def test_shift_delay(self):
        ''' Exponential (discrete): shift, delay '''
        d = ds.Exponential()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.Exponential(1, 2).shift(2)
        np.testing.assert_array_equal(d[-3:3], np.power(2.0, np.arange(-5, 1)))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.Exponential(1, sp.Rational(1, 4)).delay(-2)
        np.testing.assert_array_equal(d[-3:3],
                                      np.power(0.25, np.arange(-1, 5)))

    def test_scale(self):
        ''' Exponential (discrete): shift, delay '''
        d = ds.Exponential()
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.Exponential(1, sp.Rational(1, 2)).scale(3)
        np.testing.assert_array_equal(d[-3:3],
                                      np.power(0.5, np.arange(-3, 3)))
        d = ds.Exponential(1, sp.Rational(1, 2)).scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.power(0.5, np.arange(-12, 12, 4)))
        d = ds.Exponential(1, sp.Rational(1, 2)).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.power(0.5, np.arange(-12, 12, 4)))

    def test_frequency(self):
        ''' Exponential (discrete): frequency '''
        # exponencial discreta
        c = ds.Exponential(1, sp.exp(sp.I*sp.S.Pi/4))
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = ds.Exponential(1, sp.exp(sp.I*3*sp.S.Pi/8))
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = ds.Exponential(1, sp.exp(sp.I*3/8))
        self.assertEqual(c.frequency, 3/8)
        c = ds.Exponential(1, sp.exp(sp.I*1/4))
        self.assertEqual(c.frequency, 1/4)
        c = ds.Exponential(1, sp.exp(sp.I*3/2))
        self.assertEqual(c.frequency, 3/2)
        c = ds.Exponential(1, sp.exp(sp.I*sp.S.Pi/4))
        self.assertEqual(c.phase, 0)
        c = ds.Exponential(-1, 1)
        self.assertEqual(c.phase, -sp.S.Pi)

    def test_amplitude(self):
        ''' Exponential (discrete): amplitude '''
        # exponencial discreta
        s = ds.Exponential(3, 2)
        self.assertEqual(3, s.amplitude)
        s = ds.Exponential(4, 2)
        self.assertEqual(4, s.amplitude)
        s = ds.Exponential((3+3j), 2)
        self.assertEqual((3+3j), s.amplitude)

#     def test_components(self):
#         ''' Exponential (discrete): components '''
#         # exponencial discreta
#         s = ds.Exponential()
#         s0 = s.phasor
#         self.assertEqual(s0, ds.Constant(1))
#         s1 = s.carrier
#         self.assertEqual(s1, ds.Constant(1))


if __name__ == "__main__":
    unittest.main()
