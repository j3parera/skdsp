from skdsp.signal._signal import _Signal, _FunctionSignal
from skdsp.signal.printer import latex
import numpy as np
import skdsp.signal.continuous as cs
import skdsp.signal.discrete as ds
import sympy as sp
import unittest


class DeltaTest(unittest.TestCase):

    def test_constructor(self):
        ''' Delta: constructors.
        '''
        # delta discreta
        d = ds.Delta()
        self.assertIsNotNone(d)
        # delta discreta
        d = ds.Delta(ds.n)
        self.assertIsNotNone(d)
        # delta discreta
        d = ds.Delta(ds.n-3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # otras variables simbólicas
        d = ds.Delta(sp.Symbol('r', integer=True))
        self.assertIsNotNone(d)
        d = ds.Delta(ds.m)
        self.assertIsNotNone(d)
        d = ds.Delta(ds.n-ds.m)
        self.assertIsNotNone(d)
        # variables no enteras
        with self.assertRaises(ValueError):
            ds.Delta(sp.Symbol('z', real=True))
        with self.assertRaises(ValueError):
            ds.Delta(ds.n-1.5)
        with self.assertRaises(ValueError):
            ds.Delta(ds.n/3)

        # delta continua
        d = cs.Delta()
        self.assertIsNotNone(d)
        # delta continua
        d = cs.Delta(cs.t)
        self.assertIsNotNone(d)
        # delta continua
        d = cs.Delta(cs.t-3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, cs.ContinuousFunctionSignal)
        # otras variables simbólicas
        with self.assertRaises(ValueError):
            cs.Delta(sp.Symbol('r', integer=True))
        d = cs.Delta(cs.tau)
        self.assertIsNotNone(d)
        d = cs.Delta(cs.t-cs.tau)
        self.assertIsNotNone(d)
        # variables no entero
        d = cs.Delta(sp.Symbol('z', real=True))
        self.assertIsNotNone(d)
        d = cs.Delta(cs.t-1.5)
        self.assertIsNotNone(d)
        d = cs.Delta(cs.t/3)
        self.assertIsNotNone(d)

    def test_name(self):
        ''' Delta: name.
        '''
        # delta discreta
        d = ds.Delta(ds.n-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name, 'z')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = ds.Delta(ds.n-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        del d
        d = ds.Delta(ds.n-3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name, 'yupi')

        # delta continua
        d = cs.Delta(cs.t-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name, 'z')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = cs.Delta(cs.t-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        del d
        d = cs.Delta(cs.t-3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' Delta: independent variable and expression.
        '''
        # delta discreta
        d = ds.Delta()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5
        d = ds.Delta().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        d = ds.Delta(ds.n-shift)
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = ds.Delta().flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5
        d = ds.Delta().shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        d = ds.Delta(ds.n-shift).flip()
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5
        d = ds.Delta().flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

        # delta continua
        d = cs.Delta()
        self.assertFalse(d.is_discrete)
        self.assertTrue(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5.5
        d = cs.Delta().shift(shift)
        self.assertFalse(d.is_discrete)
        self.assertTrue(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        d = cs.Delta(cs.t-shift)
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = cs.Delta().flip()
        self.assertFalse(d.is_discrete)
        self.assertTrue(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5.5
        d = cs.Delta().shift(shift).flip()
        self.assertFalse(d.is_discrete)
        self.assertTrue(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        d = cs.Delta(cs.t-shift).flip()
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5.5
        d = cs.Delta().flip().shift(shift)
        self.assertFalse(d.is_discrete)
        self.assertTrue(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' Delta: function expression.
        '''
        # delta discreta
        d = ds.Delta(sp.Symbol('k', integer=True)-3)
        # expresión
        self.assertEqual(d.yexpr, ds.UnitDelta(d.xexpr))
        self.assertEqual(d.yexpr.rewrite(sp.Piecewise),
                         sp.Piecewise((1, sp.Eq(d.xexpr, 0)),
                                      (0, True)))
        self.assertEqual(d.yexpr.rewrite(ds.UnitStep),
                         ds.UnitStep(d.xexpr) - ds.UnitStep(d.xexpr - 1))
        self.assertTrue(np.issubdtype(d.dtype, np.float_))
        self.assertTrue(d.is_integer)
        self.assertTrue(d.is_real)
        self.assertTrue(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)

        # delta continua
        d = cs.Delta(sp.Symbol('k', real=True)-3)
        # expresión
        self.assertEqual(d.yexpr, sp.DiracDelta(d.xexpr))
        self.assertEqual(d.yexpr.rewrite(sp.Piecewise),
                         sp.Piecewise((sp.DiracDelta(0), sp.Eq(d.xexpr, 0)),
                                      (0, True)))
        self.assertEqual(d.yexpr.rewrite(sp.SingularityFunction),
                         sp.SingularityFunction(d.xvar,
                                                sp.solve(d.xexpr, d.xvar)[0],
                                                -1))
        self.assertTrue(np.issubdtype(d.dtype, np.float_))
        self.assertTrue(d.is_integer)
        self.assertTrue(d.is_real)
        self.assertTrue(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(cs.Constant(0), d.imag)

    def test_period(self):
        ''' Delta: period.
        '''
        # delta discreta
        d = ds.Delta(ds.n-3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

        # delta continua
        d = cs.Delta(cs.t-3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

    def test_repr_str_latex(self):
        ''' Delta: repr, str and latex.
        '''
        # delta discreta
        d = ds.Delta(ds.n-3)
        # repr
        self.assertEqual(repr(d), 'Delta(n - 3)')
        # str
        self.assertEqual(str(d), 'delta[n - 3]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\delta \left[ n - 3 \right]$')
        # delta discreta
        d = ds.Delta(ds.n+5)
        # repr
        self.assertEqual(repr(d), 'Delta(n + 5)')
        # str
        self.assertEqual(str(d), 'delta[n + 5]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\delta \left[ n + 5 \right]$')

        # delta continua
        d = cs.Delta(cs.t-3.5)
        # repr
        self.assertEqual(repr(d), 'Delta(t - 3.5)')
        # str
        self.assertEqual(str(d), 'delta(t - 3.5)')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\delta \left( t - 3.5 \right)$')
        # delta continua
        d = cs.Delta(cs.t+5.5)
        # repr
        self.assertEqual(repr(d), 'Delta(t + 5.5)')
        # str
        self.assertEqual(str(d), 'delta(t + 5.5)')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\delta \left( t + 5.5 \right)$')

    def test_eval_sample(self):
        ''' Delta: eval(scalar), eval(range)
        '''
        # scalar
        d = ds.Delta()
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
        d = ds.Delta(ds.n-1)
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

        # scalar
        d = cs.Delta()
        self.assertTrue(np.isnan(d.eval(0)))
        self.assertAlmostEqual(d.eval(1), 0)
        self.assertAlmostEqual(d.eval(-1), 0)
        self.assertAlmostEqual(d.eval(0.5), 0)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([np.nan, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([0, np.nan, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, 0, np.nan]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([0, 0, 0]))
        # scalar
        d = cs.Delta(cs.t-1)
        self.assertAlmostEqual(d.eval(0), 0)
        self.assertTrue(np.isnan(d.eval(1)))
        self.assertAlmostEqual(d.eval(-1), 0)
        self.assertAlmostEqual(d.eval(0.5), 0)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([0, np.nan]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([0, 0, np.nan]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([0, np.nan, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2, 0.5)),
                                             np.array([0, 0, 0, 0, np.nan, 0]))

    def test_getitem_scalar(self):
        ''' Delta (discrete): eval[scalar], eval[slice] '''
        # scalar
        d = ds.Delta()
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
        d = ds.Delta(ds.n+1)
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

        # scalar
        d = cs.Delta()
        self.assertTrue(np.isnan(d[0]))
        self.assertAlmostEqual(d[1], 0)
        self.assertAlmostEqual(d[-1], 0)
        self.assertAlmostEqual(d[0.5], 0)
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([np.nan, 0]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([0, np.nan, 0]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, 0, np.nan]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d[0:2:0.5],
                                             np.array([np.nan, 0, 0, 0]))

        # scalar
        d = cs.Delta(cs.t+1)
        self.assertAlmostEqual(d[0], 0)
        self.assertAlmostEqual(d[1], 0)
        self.assertTrue(np.isnan(d[-1]))
        self.assertAlmostEqual(d[0.5], 0)
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([0, 0]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([np.nan, 0, 0]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([0, 0, np.nan]))
        np.testing.assert_array_almost_equal(d[-1:1:0.5],
                                             np.array([np.nan, 0, 0, 0]))

    def test_generator(self):
        ''' Delta (discrete): generate '''
        d = ds.Delta()
        with self.assertRaises(ValueError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 0, 1, 0]))
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 1, 0, 0]))
        np.testing.assert_array_equal(next(dg), np.array([0, 1, 0, 0, 0]))

        ''' Delta (continuous): generate '''
        d = cs.Delta()
        dg = d.generate(s0=-1.5, size=5, overlap=2, step=0.5)
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 0, np.nan, 0]))
        np.testing.assert_array_equal(next(dg), np.array([0, 0, np.nan, 0, 0]))
        np.testing.assert_array_equal(next(dg), np.array([0, np.nan, 0, 0, 0]))

    def test_flip(self):
        ''' Delta (discrete): flip '''
        d = ds.Delta()
        np.testing.assert_array_equal(d[-3:3], d.flip()[-3:3])
        d = ds.Delta(ds.n-1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 1, 0, 0, 0]))
        d = ds.Delta(ds.n+1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 0, 1, 0]))

        ''' Delta (continuous): flip '''
        d = cs.Delta()
        np.testing.assert_array_equal(d[-3:3], d.flip()[-3:3])
        d = cs.Delta(cs.t-1).flip()
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([0, 0, np.nan, 0, 0, 0]))
        d = cs.Delta(cs.t+1).flip()
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([0, 0, 0, 0, np.nan, 0]))

    def test_shift_delay(self):
        ''' Delta (discrete): shift, delay '''
        d = ds.Delta()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.Delta().shift(2)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 0, 0, 1]))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.Delta().delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 1, 0, 0, 0, 0]))

        ''' Delta (continuous): shift, delay '''
        d = cs.Delta()
        d = cs.Delta().shift(2)
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([0, 0, 0, 0, 0, np.nan]))
        d = cs.Delta().delay(-2)
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([0, np.nan, 0, 0, 0, 0]))

    def test_scale(self):
        ''' Delta (discrete): shift, delay '''
        d = ds.Delta()
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.Delta().scale(3)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 1, 0, 0]))
        d = ds.Delta().scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 1, 0, 0]))
        d = ds.Delta(ds.n-1).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 0, 0, 0]))

        ''' Delta (continuous): shift, delay '''
        d = cs.Delta()
        d = cs.Delta().scale(3)
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([0, 0, 0, np.nan, 0, 0]))
        d = cs.Delta().scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, np.nan, 0, 0]))
        d = cs.Delta(cs.t-1).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
