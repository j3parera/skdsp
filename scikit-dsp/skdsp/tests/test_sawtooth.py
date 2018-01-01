from skdsp.signal._signal import _Signal, _FunctionSignal
from skdsp.signal.printer import latex
import numpy as np
import skdsp.signal.discrete as ds
import sympy as sp
import unittest


class SawtoothTest(unittest.TestCase):

    def saw(self, N, w, n0):
        n0 = n0 % N
        if n0 < w:
            return -1 + 2*n0/w
        else:
            return 1 - 2*(n0-w)/(N-w)

    def test_constructor(self):
        ''' Sawtooth: constructors.
        '''
        # sawtooth discreta
        d = ds.Sawtooth()
        self.assertIsNotNone(d)
        # sawtooth discreta
        d = ds.Sawtooth(ds.n)
        self.assertIsNotNone(d)
        # sawtooth discreta
        d = ds.Sawtooth(ds.n-3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # retardo simbólico
        d = ds.Sawtooth(sp.Symbol('r', integer=True))
        self.assertIsNotNone(d)
        d = ds.Sawtooth(ds.m)
        self.assertIsNotNone(d)
        d = ds.Sawtooth(ds.n-ds.m)
        self.assertIsNotNone(d)
        # retardo no entero
        with self.assertRaises(ValueError):
            ds.Sawtooth(sp.Symbol('z', real=True))
        with self.assertRaises(ValueError):
            ds.Sawtooth(ds.n-1.5)
        with self.assertRaises(ValueError):
            ds.Sawtooth(ds.n/3)

    def test_name(self):
        ''' Sawtooth: name.
        '''
        # sawtooth discreta
        d = ds.Sawtooth(ds.n-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name, 'z')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = ds.Sawtooth(ds.n-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        del d
        d = ds.Sawtooth(ds.n-3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' Sawtooth: independent variable and expression.
        '''
        # sawtooth discreta
        d = ds.Sawtooth()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5
        d = ds.Sawtooth().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        d = ds.Sawtooth(ds.n-shift)
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = ds.Sawtooth().flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5
        d = ds.Sawtooth().shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        d = ds.Sawtooth(ds.n-shift).flip()
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5
        d = ds.Sawtooth().flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' Sawtooth: function expression.
        '''
        # sawtooth discreta
        d = ds.Sawtooth(sp.Symbol('k', integer=True))
        # expresión
        nm = sp.Mod(d.xexpr, d.period)
        yexpr = sp.Piecewise((-1+(2*nm)/d.width, nm < d.width),
                             (1-2*(nm-d.width)/(d.period-d.width),
                              nm < d.period))
        self.assertEqual(d.yexpr, yexpr)
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertFalse(d.is_integer)
        self.assertTrue(d.is_real)
        self.assertTrue(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)

    def test_period(self):
        ''' Sawtooth: period.
        '''
        # sawtooth discreta
        d = ds.Sawtooth(ds.n-3)
        # periodicidad
        self.assertTrue(d.is_periodic)
        self.assertEqual(d.period, 16)
        # sawtooth discreta
        d = ds.Sawtooth(ds.n, 27)
        # periodicidad
        self.assertTrue(d.is_periodic)
        self.assertEqual(d.period, 27)

    def test_repr_str_latex(self):
        ''' Sawtooth: repr, str and latex.
        '''
        # sawtooth discreta
        d = ds.Sawtooth(ds.n)
        # repr
        self.assertEqual(repr(d), 'Sawtooth(n, 16, 8)')
        # str
        self.assertEqual(str(d), 'saw[((n))16/8]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\mathrm{Sawtooth}_{8}\left[((n))_{16}\right]$')
        # sawtooth discreta
        d = ds.Sawtooth(ds.n, sp.Symbol('L', integer=True, positive=True))
        # repr
        self.assertEqual(repr(d), 'Sawtooth(n, L, 8)')
        # str
        self.assertEqual(str(d), 'saw[((n))L/8]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\mathrm{Sawtooth}_{8}\left[((n))_{L}\right]$')
        # sawtooth discreta
        d = ds.Sawtooth(ds.n+5, 25, sp.Symbol('W', integer=True, positive=True))
        # repr
        self.assertEqual(repr(d), 'Sawtooth(n + 5, 25, W)')
        # str
        self.assertEqual(str(d), 'saw[((n + 5))25/W]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\mathrm{Sawtooth}_{W}\left[((n + 5))_{25}\right]$')
        # sawtooth discreta
        d = ds.Sawtooth(ds.n-5, 25, 10)
        # repr
        self.assertEqual(repr(d), 'Sawtooth(n - 5, 25, 10)')
        # str
        self.assertEqual(str(d), 'saw[((n - 5))25/10]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\mathrm{Sawtooth}_{10}\left[((n - 5))_{25}\right]$')

    def test_eval_sample(self):
        ''' Sawtooth: eval(scalar), eval(range)
        '''
        # scalar
        d = ds.Sawtooth()
        self.assertAlmostEqual(d.eval(0), -1)
        self.assertAlmostEqual(d.eval(1), -0.75)
        self.assertAlmostEqual(d.eval(-1), -0.75)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([-1, -0.75]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([-0.75, -1, -0.75]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, -0.5, -1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([-0.25, -0.75, -0.75]))
        # scalar
        d = ds.Sawtooth(ds.n-1)
        self.assertAlmostEqual(d.eval(0), -0.75)
        self.assertAlmostEqual(d.eval(1), -1)
        self.assertAlmostEqual(d.eval(-1), -0.5)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([-0.75, -1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([-0.5, -0.75, -1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0.25, -0.25, -0.75]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([-0.5, -1, -0.5]))
        with self.assertRaises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        ''' Sawtooth (discrete): eval[scalar], eval[slice] '''
        # scalar
        d = ds.Sawtooth()
        self.assertAlmostEqual(d[0], -1)
        self.assertAlmostEqual(d[1], -0.75)
        self.assertAlmostEqual(d[-1], -0.75)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([-1, -0.75]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([-0.75, -1, -0.75]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, -0.5, -1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([-0.25, -0.75, -0.75]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]
        # scalar
        d = ds.Sawtooth(ds.n+1)
        self.assertAlmostEqual(d[0], -0.75)
        self.assertAlmostEqual(d[1], -0.5)
        self.assertAlmostEqual(d[-1], -1)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([-0.75, -0.5]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([-1, -0.75, -0.5]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([-0.25, -0.75, -0.75]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([0, -0.5, -1]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]

    def test_generator(self):
        ''' Sawtooth (discrete): generate '''
        d = ds.Sawtooth()
        with self.assertRaises(ValueError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.array([-0.25, -0.5, -0.75,
                                                          -1, -0.75]))
        np.testing.assert_array_equal(next(dg), np.array([-0.5, -0.75, -1,
                                                          -0.75, -0.5]))
        np.testing.assert_array_equal(next(dg), np.array([-0.75, -1, -0.75,
                                                          -0.5, -0.25]))

    def test_flip(self):
        ''' Sawtooth (discrete): flip '''
        d = ds.Sawtooth()
        np.testing.assert_array_equal(d[-3:3], d.flip()[-3:3])
        d = ds.Sawtooth(ds.n-1).flip()
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([-0.5, -0.75, -1, -0.75, -0.5,
                                                -0.25]))
        d = ds.Sawtooth(ds.n+1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([0, -0.25, -0.5, -0.75,
                                                         -1, -0.75]))

    def test_shift_delay(self):
        ''' Sawtooth (discrete): shift, delay '''
        d = ds.Sawtooth()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.Sawtooth().shift(2)
        np.testing.assert_array_equal(d[-3:3], np.array([0.25, 0, -0.25,
                                                         -0.5, -0.75, -1]))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.Sawtooth().delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.array([-0.75, -1, -0.75,
                                                         -0.5, -0.25,  0]))

    def test_scale(self):
        ''' Sawtooth (discrete): shift, delay '''
        d = ds.Sawtooth()
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.Sawtooth().scale(3)
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([-0.25, -0.5, -0.75, -1,
                                                -0.75, -0.5]))
        d = ds.Sawtooth().scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 1, 0, -1, 0, 1]))
        d = ds.Sawtooth(ds.n-1).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([-0.25, 0.75, 0.25, -0.75,
                                                -0.25,  0.75]))


if __name__ == "__main__":
    unittest.main()
