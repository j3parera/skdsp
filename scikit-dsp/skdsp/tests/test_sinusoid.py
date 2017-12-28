from fractions import Fraction
from skdsp.signal._signal import _Signal, _FunctionSignal
import numpy as np
import skdsp.signal.discrete as ds
import sympy as sp
import unittest


class SinusoidTest(unittest.TestCase):

    def test_constructor(self):
        ''' Sinusoid (discrete): constructors '''
        # sinusoide discreta
        c = ds.Sinusoid()
        self.assertIsInstance(c, _Signal)
        self.assertIsInstance(c, _FunctionSignal)
        self.assertIsInstance(c, ds.DiscreteFunctionSignal)
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        with self.assertRaises(ValueError):
            c = ds.Sinusoid(3+3j)

    def test_name(self):
        ''' Sinusoid: name.
        '''
        # sinusoide discreta
        c = ds.Sinusoid(name='y0')
        self.assertEqual(c.name, 'y0')
        self.assertEqual(c.latex_name, 'y_{0}')
        c.name = 'z'
        self.assertEqual(c.name, 'z')
        self.assertEqual(c.latex_name, 'z')
        with self.assertRaises(ValueError):
            c.name = 'x0'
        with self.assertRaises(ValueError):
            c.name = 'y0'
        c = ds.Sinusoid(name='y0')
        self.assertEqual(c.name, 'y0')
        self.assertEqual(c.latex_name, 'y_{0}')
        del c
        c = ds.Sinusoid(name='yupi')
        self.assertEqual(c.name, 'yupi')
        self.assertEqual(c.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' Sinusoid: independent variable and expression.
        '''
        # sinusoide discreta
        c = ds.Sinusoid()
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, c.xvar)
        # shift
        shift = 5
        c = ds.Sinusoid().shift(shift)
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, c.xvar - shift)
        # flip
        c = ds.Sinusoid().flip()
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, -c.xvar)
        # shift and flip
        shift = 5
        c = ds.Sinusoid().shift(shift).flip()
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, -c.xvar - shift)
        # flip and shift
        shift = 5
        c = ds.Sinusoid().flip().shift(shift)
        self.assertTrue(c.is_discrete)
        self.assertFalse(c.is_continuous)
        self.assertEqual(c.xvar, c.default_xvar())
        self.assertEqual(c.xexpr, -c.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' Sinusoid: function expression.
        '''
        # sinusoide discreta
        c = ds.Sinusoid()
        # expresión
        self.assertEqual(c.yexpr, sp.cos(ds._DiscreteMixin.default_xvar()))
        self.assertTrue(np.issubdtype(c.dtype, np.float))
        self.assertFalse(c.is_real)
        self.assertFalse(c.is_complex)
        self.assertEqual(c, c.real)
        self.assertEqual(ds.Constant(0), c.imag)

    def test_period(self):
        ''' Sinusoid: period.
        '''
        # sinusoide discreta
        c = ds.Sinusoid()
        self.assertFalse(c.is_periodic)
        self.assertEqual(c.period, sp.oo)
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4)
        self.assertTrue(c.is_periodic)
        self.assertEqual(c.period, 8)
        c = ds.Sinusoid(ds.n, 1, 3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic)
        self.assertEqual(c.period, 48)
        c = ds.Sinusoid(ds.n, 1, -3*sp.S.Pi/8)
        self.assertTrue(c.is_periodic)
        self.assertEqual(c.period, 208)
        c = ds.Sinusoid(ds.n, 1, 0.83*sp.S.Pi)
        self.assertTrue(c.is_periodic)
        self.assertEqual(c.period, 16600)
        c = ds.Sinusoid(ds.n, 1, 3/8)
        self.assertFalse(c.is_periodic)
        self.assertEqual(c.period, sp.oo)
        c = ds.Sinusoid(ds.n, 1, 1/4)
        self.assertFalse(c.is_periodic)
        self.assertEqual(c.period, sp.oo)
        c = ds.Sinusoid(ds.n, 1, 3/2)
        self.assertFalse(c.is_periodic)
        self.assertEqual(c.period, sp.oo)
        # unos cuantos casos, p.ej 0.35*pi -> 40, 0.75*pi -> 8
        for M in np.arange(5, 105, 5):
            f = Fraction(200, M)
            c = ds.Sinusoid(ds.n, 1, M/100*sp.S.Pi)
            self.assertEqual(c.period, f.numerator * f.denominator)

    def test_repr_str_latex(self):
        ''' Sinusoid: repr, str and latex.
        '''
        # sinusoide discreta
        c = ds.Sinusoid(ds.n, 3, sp.S.Pi/4, sp.S.Pi/8)
        # repr
        self.assertEqual(repr(c), 'Sinusoid(3, pi/4, pi/8)')
        # str
        self.assertEqual(str(c), '3*cos(pi/4*n + pi/8)')
        # sinusoide discreta
        c = ds.Sinusoid(ds.n, -5, 10)
        # repr
        self.assertEqual(repr(c), 'Sinusoid(-5, 10, 0)')
        # str
        self.assertEqual(str(c), '-5*cos(10*n)')
        c = ds.Sinusoid(ds.n, 3, sp.S.Pi/4, sp.S.Pi/8).flip().delay(2)
        # repr
        self.assertEqual(repr(c), 'Sinusoid(3, pi/4, pi/8)')
        # str
        self.assertEqual(str(c), '3*cos(pi/4*(-n + 2) + pi/8)')

    def test_eval_sample(self):
        ''' Sinusoid (discrete): eval(scalar) '''
        # sinusoide discreta
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c.eval(0), np.cos(np.pi/6))
        self.assertAlmostEqual(c.eval(1), np.cos(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c.eval(-1), np.cos(-np.pi/4 + np.pi/6))
        c = ds.Sinusoid(ds.n, 1, 0, sp.S.Pi/2)
        self.assertAlmostEqual(c.eval(0), 0)
        with self.assertRaises(ValueError):
            c.eval(0.5)

    def test_eval_range(self):
        ''' Sinusoid (discrete): eval(range) '''
        # sinusoide discreta
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4, sp.S.Pi/6)
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
        ''' Sinusoid (discrete): eval[scalar] '''
        # sinusoide discreta
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4, sp.S.Pi/6)
        self.assertAlmostEqual(c[0], np.cos(np.pi/6))
        self.assertAlmostEqual(c[1], np.cos(np.pi/4 + np.pi/6))
        self.assertAlmostEqual(c[-1], np.cos(-np.pi/4 + np.pi/6))

    def test_getitem_slice(self):
        ''' Sinusoid (discrete): eval[:] '''
        # sinusoide discreta
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4, sp.S.Pi/6)
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

    def test_generator(self):
        ''' Sinusoid (discrete): generate '''
        c = ds.Sinusoid()
        with self.assertRaises(ValueError):
            c.generate(0, step=0.1)
        dg = c.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.cos(np.arange(-3, 2)))
        np.testing.assert_array_equal(next(dg), np.cos(np.arange(-2, 3)))
        np.testing.assert_array_equal(next(dg), np.cos(np.arange(-1, 4)))

    def test_flip(self):
        ''' Sinusoid (discrete): flip '''
        d = ds.Sinusoid().flip()
        np.testing.assert_array_equal(d[-3:3], np.cos(np.arange(3, -3, -1)))
        d = ds.Sinusoid(ds.n, 2).flip()
        np.testing.assert_array_equal(d[-3:3], 2*np.cos(np.arange(3, -3, -1)))
        d = ds.Sinusoid(ds.n, -1).flip()
        np.testing.assert_array_equal(d[-3:3], -1*np.cos(np.arange(3, -3, -1)))

    def test_shift_delay(self):
        ''' Sinusoid (discrete): shift, delay '''
        d = ds.Sinusoid()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.Sinusoid().shift(2)
        np.testing.assert_array_equal(d[-3:3], np.cos(np.arange(-5, 1)))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.Sinusoid().delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.cos(np.arange(-1, 5)))

    def test_scale(self):
        ''' Sinusoid (discrete): shift, delay '''
        d = ds.Sinusoid()
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.Sinusoid().scale(3)
        np.testing.assert_array_equal(d[-3:3], np.cos(np.arange(-3, 3)))
        d = ds.Sinusoid().scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.cos(np.arange(-12, 12, 4)))
        d = ds.Sinusoid(ds.n, 1).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.cos(np.arange(-12, 12, 4)))

    def test_frequency(self):
        ''' Sinusoid (discrete): frequency '''
        # sinusoide discreta
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4)
        self.assertEqual(c.frequency, sp.S.Pi/4)
        c = ds.Sinusoid(ds.n, 1, 3*sp.S.Pi/8)
        self.assertEqual(c.frequency, 3*sp.S.Pi/8)
        c = ds.Sinusoid(ds.n, 1, 3/8)
        self.assertEqual(c.frequency, 3/8)
        c = ds.Sinusoid(ds.n, 1, 1/4)
        self.assertEqual(c.frequency, 1/4)
        c = ds.Sinusoid(ds.n, 1, 3/2)
        self.assertEqual(c.frequency, 3/2)
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4, sp.S.Pi)
        self.assertEqual(c.phase, -sp.S.Pi)
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4, 1/2)
        self.assertEqual(c.phase, 1/2)
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4, 2*sp.S.Pi)
        self.assertEqual(c.phase, 0)
        c = ds.Sinusoid(ds.n, 1, sp.S.Pi/4, 3*sp.S.Pi)
        self.assertEqual(c.phase, -sp.S.Pi)

    def test_as_euler(self):
        ''' Sinusoid (discrete): as_euler '''
        # sinusoide discreta
        s = ds.Sinusoid(ds.n, 0.5, sp.S.Pi/4, sp.S.Pi/3)
        self.assertEqual('0.25*exp(j*(-pi*n/4 - pi/3))' +
                         ' + 0.25*exp(j*(pi*n/4 + pi/3))', str(s.as_euler()))

    def test_amplitude(self):
        ''' Sinusoid (discrete): amplitude '''
        # sinusoide discreta
        s = ds.Sinusoid(ds.n, 3, sp.S.Pi/4, sp.S.Pi/12)
        self.assertEqual(3, s.amplitude)
        with self.assertRaises(ValueError):
            s = ds.Sinusoid(3+3j, sp.S.Pi/4, sp.S.Pi/12)

    def test_magnitude(self):
        ''' Sinusoid (discrete): magnitude '''
        # sinusoide discreta
        s = ds.Sinusoid(ds.n, 1, sp.S.Pi, 0).magnitude()
        np.testing.assert_equal(np.r_[[1]*5], s[0:10:2])
        np.testing.assert_equal(np.r_[[1]*5], s[1:11:2])
        s = ds.Sinusoid(ds.n, 3, sp.S.Pi/8, sp.S.Pi/4).magnitude()
        np.testing.assert_equal(np.r_[[3]*4], s[-2:49:16])
        np.testing.assert_equal(np.r_[[3]*4], s[14:65:16])
        # en dBs
        s = ds.Sinusoid(ds.n, 1, sp.S.Pi, 0).magnitude(dB=True)
        np.testing.assert_equal(np.r_[[0]*5], s[0:10:2])
        np.testing.assert_equal(np.r_[[0]*5], s[1:11:2])
        s = ds.Sinusoid(ds.n, 3, sp.S.Pi/8, sp.S.Pi/4).magnitude(True)
        np.testing.assert_equal(20*np.log10(np.r_[[3]*4]), s[-2:49:16])
        np.testing.assert_equal(20*np.log10(np.r_[[3]*4]), s[14:65:16])

    def test_components(self):
        ''' Sinusoid (discrete): components '''
        # sinusoide discreta
        s = ds.Sinusoid()
        s0 = s.in_phase
        self.assertEqual(s0, ds.Sinusoid())
        s0 = s.i
        self.assertEqual(s0, ds.Sinusoid())
        s1 = s.in_quadrature
        self.assertEqual(s1, ds.Constant(0))
        s1 = s.q
        self.assertEqual(s1, ds.Constant(0))
        s = ds.Sinusoid(ds.n, 3, sp.S.Pi/8, sp.S.Pi/4)
        s0 = s.in_phase
        self.assertEqual(s0, ds.Sinusoid(ds.n, 3*sp.cos(sp.S.Pi/4),
                                         sp.S.Pi/8, 0))
        s0 = s.i
        self.assertEqual(s0, ds.Sinusoid(ds.n, 3*sp.cos(sp.S.Pi/4),
                                         sp.S.Pi/8, 0))
        s1 = s.in_quadrature
        self.assertEqual(s1, ds.Sinusoid(ds.n, -3*sp.sin(sp.S.Pi/4),
                                         sp.S.Pi/8, -sp.S.Pi/2))
        s1 = s.q
        self.assertEqual(s1, ds.Sinusoid(ds.n, -3*sp.sin(sp.S.Pi/4),
                                         sp.S.Pi/8, -sp.S.Pi/2))


if __name__ == "__main__":
    unittest.main()
