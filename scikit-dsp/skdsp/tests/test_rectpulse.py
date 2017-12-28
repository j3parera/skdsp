from ..signal._signal import _Signal, _FunctionSignal
from ..signal.printer import latex
import numpy as np
import skdsp.signal.discrete as ds
import sympy as sp
import unittest


class RectPulseTest(unittest.TestCase):

    def test_constructor(self):
        ''' RectPulse: constructors.
        '''
        # pulso rectangular discreto
        d = ds.RectPulse()
        self.assertIsNotNone(d)
        # pulso rectangular discreto
        d = ds.RectPulse(ds.n-3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # anchura simbólica
        d = ds.RectPulse(sp.Symbol('L', integer=True, positive=True))
        self.assertIsNotNone(d)
        # retardo no entero
        with self.assertRaises(ValueError):
            ds.RectPulse(sp.Symbol('z', real=True))
        with self.assertRaises(ValueError):
            ds.RectPulse(ds.n-1.5)
        with self.assertRaises(ValueError):
            ds.RectPulse(ds.n/3)

    def test_name(self):
        ''' RectPulse: name.
        '''
        # pulso rectangular discreto
        d = ds.RectPulse(ds.n, 3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name, 'z')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = ds.RectPulse(ds.n, 3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        del d
        d = ds.RectPulse(ds.n, 3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' RectPulse: independent variable and expression.
        '''
        # pulso rectangular discreto
        d = ds.RectPulse()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5
        d = ds.RectPulse().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = ds.RectPulse().flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5
        d = ds.RectPulse().shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5
        d = ds.RectPulse().flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' RectPulse: function expression.
        '''
        # pulso rectangular discreto
        d = ds.RectPulse()
        # expresión
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_real)
        self.assertTrue(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)

    def test_period(self):
        ''' RectPulse: period.
        '''
        # pulso rectangular discreto
        d = ds.RectPulse(ds.n, 3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

    def test_repr_str_latex(self):
        ''' RectPulse: repr, str and latex.
        '''
        # pulso rectangular discreto
        d = ds.RectPulse(ds.n, 3)
        # repr
        self.assertEqual(repr(d), 'RectPulse(n, 3)')
        # str
        self.assertEqual(str(d), 'Pi3[n]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\Pi_{3} \left[ n \right]$')
        # pulso rectangular discreto
        d = ds.RectPulse(ds.n, 4)
        # repr
        self.assertEqual(repr(d), 'RectPulse(n, 4)')
        # str
        self.assertEqual(str(d), 'Pi4[n]')
        # latex
        self.assertEqual(latex(d.shift(1), mode='inline'),
                         r'$\Pi_{4} \left[ n - 1 \right]$')

    def test_eval_sample(self):
        ''' RectPulse: eval(scalar), eval(range)
        '''
        # scalar
        d = ds.RectPulse()
        self.assertAlmostEqual(d.eval(0), 1)
        self.assertAlmostEqual(d.eval(1), 1)
        self.assertAlmostEqual(d.eval(-1), 1)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([1, 1, 1]))
        # scalar
        d = ds.RectPulse(ds.n, 1)
        self.assertAlmostEqual(d.eval(0), 1)
        self.assertAlmostEqual(d.eval(1), 1)
        self.assertAlmostEqual(d.eval(-1), 1)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([0, 1, 1]))
        with self.assertRaises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        ''' RectPulse (discrete): eval[scalar], eval[slice] '''
        # scalar
        d = ds.RectPulse()
        self.assertAlmostEqual(d[0], 1)
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[-1], 1)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([1, 1, 1]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]
        # scalar
        d = ds.RectPulse(ds.n, 1)
        self.assertAlmostEqual(d[0], 1)
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[-1], 1)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([0, 1, 1]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]

    def test_generator(self):
        ''' RectPulse (discrete): generate '''
        d = ds.RectPulse(ds.n, 1)
        with self.assertRaises(ValueError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 1, 1, 1]))
        np.testing.assert_array_equal(next(dg), np.array([0, 1, 1, 1, 0]))
        np.testing.assert_array_equal(next(dg), np.array([1, 1, 1, 0, 0]))

    def test_flip(self):
        ''' RectPulse (discrete): flip '''
        d = ds.RectPulse(ds.n).flip()
        np.testing.assert_array_equal(d[-3:3], d.flip()[-3:3])
        d = ds.RectPulse(ds.n, 1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 1, 1, 1, 0]))
        d = ds.RectPulse(ds.n, 2).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([0, 1, 1, 1, 1, 1]))

    def test_shift_delay(self):
        ''' RectPulse (discrete): shift, delay '''
        d = ds.RectPulse()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.RectPulse(ds.n, 8).shift(2)
        np.testing.assert_array_equal(d[-3:3], np.array([1, 1, 1, 1, 1, 1]))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.RectPulse(ds.n, 8).delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.array([1, 1, 1, 1, 1, 1]))

    def test_scale(self):
        ''' RectPulse (discrete): shift, delay '''
        d = ds.RectPulse(ds.n, 8)
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.RectPulse(ds.n, 8).scale(3)
        np.testing.assert_array_equal(d[-3:3], np.array([1, 1, 1, 1, 1, 1]))
        d = ds.RectPulse(ds.n, 8).scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 1, 1, 1, 1, 1]))
        d = ds.RectPulse(ds.n, 8).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 1, 1, 1, 1, 1]))

    def test_width(self):
        ''' RectPulse (discrete): shift, delay '''
        d = ds.RectPulse(ds.n, 8)
        self.assertEqual(d.width, 8)
        N = sp.Symbol('N', integer=True, positive=True)
        d = ds.RectPulse(ds.n, N)
        self.assertEqual(d.width, N)
        with self.assertRaises(ValueError):
            d = ds.RectPulse(-8)
            d = ds.RectPulse(sp.Symbol('L', integer=True))


if __name__ == "__main__":
    unittest.main()
