from skdsp.signal._signal import _Signal, _FunctionSignal
import numpy as np
import skdsp.signal.discrete as ds
import sympy as sp
import unittest


class RampTest(unittest.TestCase):

    def test_constructor(self):
        ''' Ramp: constructors.
        '''
        # rampa discreta
        d = ds.Ramp()
        self.assertIsNotNone(d)
        # rampa discreta
        d = ds.Ramp(3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # retardo simbólico
        d = ds.Ramp(sp.Symbol('n0', integer=True))
        self.assertIsNotNone(d)
        # retardo no entero
        with self.assertRaises(ValueError):
            d = ds.Ramp(sp.Symbol('x0', real=True))
        with self.assertRaises(ValueError):
            d = ds.Ramp(1.5)

    def test_name(self):
        ''' Ramp: name.
        '''
        # rampa discreta
        d = ds.Ramp(3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name(), 'y_{0}')
        self.assertEqual(d.latex_name('inline'), '$y_{0}$')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name(), 'z')
        self.assertEqual(d.latex_name('inline'), '$z$')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = ds.Ramp(3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name(), 'y_{0}')
        self.assertEqual(d.latex_name('inline'), '$y_{0}$')
        del d
        d = ds.Ramp(3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name(), 'yupi')
        self.assertEqual(d.latex_name('inline'), '$yupi$')

    def test_xvar_xexpr(self):
        ''' Ramp: independent variable and expression.
        '''
        # rampa discreta
        d = ds.Ramp()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5
        d = ds.Ramp().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        d = ds.Ramp(shift)
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = ds.Ramp().flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5
        d = ds.Ramp().shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        d = ds.Ramp(shift).flip()
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5
        d = ds.Ramp().flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' Ramp: function expression.
        '''
        # rampa discreta
        d = ds.Ramp()
        # expresión
        self.assertEqual(d.yexpr, ds.Ramp._DiscreteRamp(
            ds._DiscreteMixin._default_xvar()))
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_real)
        self.assertFalse(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)

    def test_period(self):
        ''' Ramp: period.
        '''
        # rampa discreta
        d = ds.Ramp(3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

    def test_repr_str_latex(self):
        ''' Ramp: repr, str and latex.
        '''
        # rampa discreta
        d = ds.Ramp(3)
        # repr
        self.assertEqual(repr(d), 'Ramp(3)')
        # str
        self.assertEqual(str(d), 'r[n - 3]')
        # latex
        self.assertEqual(d.latex_yexpr(), 'r\\left[n - 3\\right]')
        # rampa discreta
        d = ds.Ramp(-5)
        # repr
        self.assertEqual(repr(d), 'Ramp(-5)')
        # str
        self.assertEqual(str(d), 'r[n + 5]')
        # latex
        self.assertEqual(d.latex_yexpr(), 'r\\left[n + 5\\right]')

    def test_eval_sample(self):
        ''' Ramp: eval(scalar), eval(range)
        '''
        # scalar
        d = ds.Ramp()
        self.assertAlmostEqual(d.eval(0), 0)
        self.assertAlmostEqual(d.eval(1), 1)
        self.assertAlmostEqual(d.eval(-1), 0)
        with self.assertRaises(TypeError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([0, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([3, 1, 0]))
        # scalar
        d = ds.Ramp(1)
        self.assertAlmostEqual(d.eval(0), 0)
        self.assertAlmostEqual(d.eval(1), 0)
        self.assertAlmostEqual(d.eval(-1), 0)
        with self.assertRaises(TypeError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([0, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([2, 0, 0]))
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        ''' Ramp (discrete): eval[scalar], eval[slice] '''
        # scalar
        d = ds.Ramp()
        self.assertAlmostEqual(d[0], 0)
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[-1], 0)
        with self.assertRaises(TypeError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([0, 1]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([3, 1, 0]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]
        # scalar
        d = ds.Ramp(-1)
        self.assertAlmostEqual(d[0], 1)
        self.assertAlmostEqual(d[1], 2)
        self.assertAlmostEqual(d[-1], 0)
        with self.assertRaises(TypeError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([1, 2]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([0, 1, 2]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([4, 2, 0]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]

    def test_generator(self):
        ''' Ramp (discrete): generate '''
        d = ds.Ramp()
        with self.assertRaises(TypeError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 0, 0, 1]))
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 0, 1, 2]))
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 1, 2, 3]))

    def test_flip(self):
        ''' Ramp (discrete): flip '''
        d = ds.Ramp().flip()
        np.testing.assert_array_equal(d[-3:3], np.array([3, 2, 1, 0, 0, 0]))
        d = ds.Ramp(1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([2, 1, 0, 0, 0, 0]))
        d = ds.Ramp(-1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([4, 3, 2, 1, 0, 0]))

    def test_shift_delay(self):
        ''' Ramp (discrete): shift, delay '''
        d = ds.Ramp()
        with self.assertRaises(TypeError):
            d.shift(0.5)
        d = ds.Ramp().shift(2)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 0, 0, 0]))
        with self.assertRaises(TypeError):
            d.delay(0.5)
        d = ds.Ramp().delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 1, 2, 3, 4]))

    def test_scale(self):
        ''' Ramp (discrete): shift, delay '''
        d = ds.Ramp()
        with self.assertRaises(TypeError):
            d.scale(sp.pi)
        d = ds.Ramp().scale(3)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 0, 1, 2]))
        d = ds.Ramp().scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 0, 4, 8]))
        d = ds.Ramp(1).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 0, 3, 7]))

if __name__ == "__main__":
    unittest.main()
