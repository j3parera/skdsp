from skdsp.signal._signal import _Signal, _FunctionSignal
from skdsp.signal.printer import latex
import numpy as np
import skdsp.signal.discrete as ds
import sympy as sp
import unittest


class DeltaTrainTest(unittest.TestCase):

    def test_constructor(self):
        ''' DeltaTrainTrain: constructors.
        '''
        # delta discreta
        d = ds.DeltaTrain()
        self.assertIsNotNone(d)
        # delta discreta
        d = ds.DeltaTrain(ds.n)
        self.assertIsNotNone(d)
        # delta discreta
        d = ds.DeltaTrain(ds.n-3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # retardo simbólico
        d = ds.DeltaTrain(sp.Symbol('r', integer=True))
        self.assertIsNotNone(d)
        d = ds.DeltaTrain(ds.m)
        self.assertIsNotNone(d)
        d = ds.DeltaTrain(ds.n-ds.m)
        self.assertIsNotNone(d)
        # retardo no entero
        with self.assertRaises(ValueError):
            ds.DeltaTrain(sp.Symbol('z', real=True))
        with self.assertRaises(ValueError):
            ds.DeltaTrain(ds.n-1.5)
        with self.assertRaises(ValueError):
            ds.DeltaTrain(ds.n/3)

    def test_name(self):
        ''' DeltaTrain: name.
        '''
        # delta discreta
        d = ds.DeltaTrain(ds.n-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name, 'z')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = ds.DeltaTrain(ds.n-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        del d
        d = ds.DeltaTrain(ds.n-3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' DeltaTrain: independent variable and expression.
        '''
        # delta discreta
        d = ds.DeltaTrain()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5
        d = ds.DeltaTrain().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        d = ds.DeltaTrain(ds.n-shift)
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = ds.DeltaTrain().flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5
        d = ds.DeltaTrain().shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        d = ds.DeltaTrain(ds.n-shift).flip()
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5
        d = ds.DeltaTrain().flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' DeltaTrain: function expression.
        '''
        # delta discreta
        d = ds.DeltaTrain(sp.Symbol('k', integer=True))
        # expresión
        self.assertEqual(d.yexpr,
                         sp.Piecewise((1, sp.Eq(sp.Mod(d._xvar, d.period), 0)),
                                      (0, True)))
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_integer)
        self.assertTrue(d.is_real)
        self.assertTrue(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)

    def test_period(self):
        ''' DeltaTrain: period.
        '''
        # delta discreta
        d = ds.DeltaTrain(ds.n-3)
        # periodicidad
        self.assertTrue(d.is_periodic)
        self.assertEqual(d.period, 16)
        # delta discreta
        d = ds.DeltaTrain(ds.n, 27)
        # periodicidad
        self.assertTrue(d.is_periodic)
        self.assertEqual(d.period, 27)

    def test_repr_str_latex(self):
        ''' DeltaTrain: repr, str and latex.
        '''
        # delta discreta
        d = ds.DeltaTrain(ds.n)
        # repr
        self.assertEqual(repr(d), 'DeltaTrain(n, 16)')
        # str
        self.assertEqual(str(d), 'delta[n, 16]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\sum_{k=-\infty}^{\infty} \delta [n + 16k]$')
        # delta discreta
        d = ds.DeltaTrain(ds.n, sp.Symbol('L', integer=True, positive=True))
        # repr
        self.assertEqual(repr(d), 'DeltaTrain(n, L)')
        # str
        self.assertEqual(str(d), 'delta[n, L]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\sum_{k=-\infty}^{\infty} \delta [n + kL]$')
        # delta discreta
        d = ds.DeltaTrain(ds.n+5, 25)
        # repr
        self.assertEqual(repr(d), 'DeltaTrain(n + 5, 25)')
        # str
        self.assertEqual(str(d), 'delta[n + 5, 25]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\sum_{k=-\infty}^{\infty} \delta [(n + 5) + 25k]$')

        # delta discreta
        d = ds.DeltaTrain(ds.n-5, 25)
        # repr
        self.assertEqual(repr(d), 'DeltaTrain(n - 5, 25)')
        # str
        self.assertEqual(str(d), 'delta[n - 5, 25]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\sum_{k=-\infty}^{\infty} \delta [(n - 5) + 25k]$')

    def test_eval_sample(self):
        ''' DeltaTrain: eval(scalar), eval(range)
        '''
        # scalar
        d = ds.DeltaTrain()
        self.assertAlmostEqual(d.eval(0), 1)
        self.assertAlmostEqual(d.eval(1), 0)
        self.assertAlmostEqual(d.eval(-1), 0)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([1, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([0, 1, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([0, 0, 0]))
        # scalar
        d = ds.DeltaTrain(ds.n-1)
        self.assertAlmostEqual(d.eval(0), 0)
        self.assertAlmostEqual(d.eval(1), 1)
        self.assertAlmostEqual(d.eval(-1), 0)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([0, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([0, 1, 0]))
        with self.assertRaises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        ''' DeltaTrain (discrete): eval[scalar], eval[slice] '''
        # scalar
        d = ds.DeltaTrain()
        self.assertAlmostEqual(d[0], 1)
        self.assertAlmostEqual(d[1], 0)
        self.assertAlmostEqual(d[-1], 0)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([1, 0]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([0, 1, 0]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([0, 0, 0]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]
        # scalar
        d = ds.DeltaTrain(ds.n+1)
        self.assertAlmostEqual(d[0], 0)
        self.assertAlmostEqual(d[1], 0)
        self.assertAlmostEqual(d[-1], 1)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([0, 0]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([1, 0, 0]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([0, 0, 1]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]

    def test_generator(self):
        ''' DeltaTrain (discrete): generate '''
        d = ds.DeltaTrain()
        with self.assertRaises(ValueError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 0, 1, 0]))
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 1, 0, 0]))
        np.testing.assert_array_equal(next(dg), np.array([0, 1, 0, 0, 0]))

    def test_flip(self):
        ''' DeltaTrain (discrete): flip '''
        d = ds.DeltaTrain()
        np.testing.assert_array_equal(d[-3:3], d.flip()[-3:3])
        d = ds.DeltaTrain(ds.n-1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 1, 0, 0, 0]))
        d = ds.DeltaTrain(ds.n+1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 0, 1, 0]))

    def test_shift_delay(self):
        ''' DeltaTrain (discrete): shift, delay '''
        d = ds.DeltaTrain()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.DeltaTrain().shift(2)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 0, 0, 1]))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.DeltaTrain().delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 1, 0, 0, 0, 0]))

    def test_scale(self):
        ''' DeltaTrain (discrete): shift, delay '''
        d = ds.DeltaTrain()
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.DeltaTrain().scale(3)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 1, 0, 0]))
        d = ds.DeltaTrain().scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 1, 0, 0]))
        d = ds.DeltaTrain(ds.n-1).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
