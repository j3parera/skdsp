from skdsp.signal._signal import _Signal, _FunctionSignal
import numpy as np
import skdsp.signal.discrete as ds
import sympy as sp
import unittest


class TriangTest(unittest.TestCase):

    def test_constructor(self):
        ''' TriangPulse: constructors.
        '''
        # pulso triangular discreto
        d = ds.TriangPulse()
        self.assertIsNotNone(d)
        # pulso triangular discreto
        d = ds.TriangPulse(3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # anchura simbólica
        d = ds.TriangPulse(sp.Symbol('L', integer=True, positive=True))
        self.assertIsNotNone(d)
        # retardo no entero
        with self.assertRaises(ValueError):
            d = ds.TriangPulse(sp.Symbol('x0', real=True))
        with self.assertRaises(ValueError):
            d = ds.TriangPulse(1.5)

    def test_name(self):
        ''' TriangPulse: name.
        '''
        # pulso triangular discreto
        d = ds.TriangPulse(3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name, 'z')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = ds.TriangPulse(3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        del d
        d = ds.TriangPulse(3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' TriangPulse: independent variable and expression.
        '''
        # pulso triangular discreto
        d = ds.TriangPulse()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5
        d = ds.TriangPulse().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = ds.TriangPulse().flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5
        d = ds.TriangPulse().shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5
        d = ds.TriangPulse().flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' TriangPulse: function expression.
        '''
        # pulso triangular discreto
        d = ds.TriangPulse()
        # expresión
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_real)
        self.assertFalse(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)

    def test_period(self):
        ''' TriangPulse: period.
        '''
        # pulso triangular discreto
        d = ds.TriangPulse(3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

    def test_repr_str_latex(self):
        ''' TriangPulse: repr, str and latex.
        '''
        # pulso triangular discreto
        d = ds.TriangPulse(3)
        # repr
        self.assertEqual(repr(d), 'TriangPulse(3)')
        # str
        self.assertEqual(str(d), 'Delta3[n]')
        # pulso triangular discreto
        d = ds.TriangPulse(4)
        # repr
        self.assertEqual(repr(d), 'TriangPulse(4)')
        # str
        self.assertEqual(str(d), 'Delta4[n]')

    def test_eval_sample(self):
        ''' TriangPulse: eval(scalar), eval(range)
        '''
        # scalar
        d = ds.TriangPulse()
        self.assertAlmostEqual(d.eval(0), 1)
        self.assertAlmostEqual(d.eval(1), 1 - 1/16)
        self.assertAlmostEqual(d.eval(-1), 1 - 1/16)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([1, 1 - 1/16]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([1 - 1/16, 1, 1 - 1/16]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([1 - 4/16, 1 - 2/16, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([1 - 3/16, 1 - 1/16,
                                                       1 - 1/16]))
        # scalar
        d = ds.TriangPulse(1)
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
        with self.assertRaises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        ''' TriangPulse (discrete): eval[scalar], eval[slice] '''
        # scalar
        d = ds.TriangPulse()
        self.assertAlmostEqual(d[0], 1)
        self.assertAlmostEqual(d[1], 1 - 1/16)
        self.assertAlmostEqual(d[-1], 1 - 1/16)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([1, 1 - 1/16]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([1 - 1/16, 1, 1 - 1/16]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([1 - 4/16, 1 - 2/16, 1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([1 - 3/16, 1 - 1/16,
                                                       1 - 1/16]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]
        # scalar
        d = ds.TriangPulse(1)
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

    def test_generator(self):
        ''' TriangPulse (discrete): generate '''
        d = ds.TriangPulse(1)
        with self.assertRaises(ValueError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 0, 1, 0]))
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 1, 0, 0]))
        np.testing.assert_array_equal(next(dg), np.array([0, 1, 0, 0, 0]))

    def test_flip(self):
        ''' TriangPulse (discrete): flip '''
        d = ds.TriangPulse().flip()
        np.testing.assert_array_equal(d[-3:3], d.flip()[-3:3])
        d = ds.TriangPulse(1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 1, 0, 0]))
        d = ds.TriangPulse(2).flip()
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([0, 0, 1 - 1/2, 1, 1 - 1/2, 0]))

    def test_shift_delay(self):
        ''' TriangPulse (discrete): shift, delay '''
        d = ds.TriangPulse()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.TriangPulse(8).shift(2)
        np.testing.assert_array_equal(d[-3:3], np.array([1 - 5/8, 1 - 4/8,
                                                         1 - 3/8, 1 - 2/8,
                                                         1 - 1/8, 1]))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.TriangPulse(8).delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.array([1 - 1/8, 1, 1 - 1/8,
                                                         1 - 2/8, 1 - 3/8,
                                                         1 - 4/8]))

    def test_scale(self):
        ''' TriangPulse (discrete): shift, delay '''
        d = ds.TriangPulse(8)
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.TriangPulse(8).scale(3)
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([1 - 6/16, 1 - 4/16, 1 - 2/16,
                                                1, 1 - 2/16, 1 - 4/16]))
        d = ds.TriangPulse(8).scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 1 - 1/2, 1, 1 - 1/2, 0]))
        d = ds.TriangPulse(8).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 1 - 1/2, 1, 1 - 1/2, 0]))

    def test_width(self):
        ''' TriangPulse (discrete): shift, delay '''
        d = ds.TriangPulse(8)
        self.assertEqual(d.width, 8)
        N = sp.Symbol('N', integer=True, positive=True)
        d = ds.TriangPulse(N)
        self.assertEqual(d.width, N)
        with self.assertRaises(ValueError):
            d = ds.TriangPulse(-8)
            d = ds.TriangPulse(sp.Symbol('L', integer=True))

if __name__ == "__main__":
    unittest.main()
