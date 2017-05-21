from skdsp.signal._signal import _Signal, _FunctionSignal
import numpy as np
import skdsp.signal.discrete as ds
import sympy as sp
import unittest


class StepTest(unittest.TestCase):

    def test_constructor(self):
        ''' Step: constructors.
        '''
        # escalón discreto
        d = ds.Step()
        self.assertIsNotNone(d)
        # escalón discreto
        d = ds.Step(3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # retardo simbólico
        d = ds.Step(sp.Symbol('n0', integer=True))
        self.assertIsNotNone(d)
        # retardo no entero
        with self.assertRaises(ValueError):
            d = ds.Step(sp.Symbol('x0', real=True))
        with self.assertRaises(ValueError):
            d = ds.Step(1.5)

    def test_name(self):
        ''' Step: name.
        '''
        # escalón discreto
        d = ds.Step(3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name, 'z')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = ds.Step(3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        del d
        d = ds.Step(3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' Step: independent variable and expression.
        '''
        # escalón discreto
        d = ds.Step()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5
        d = ds.Step().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        d = ds.Step(shift)
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = ds.Step().flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5
        d = ds.Step().shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        d = ds.Step(shift).flip()
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5
        d = ds.Step().flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' Step: function expression.
        '''
        # escalón discreto
        d = ds.Step()
        # expresión
        self.assertEqual(d.yexpr, ds.Step._DiscreteStep(
            ds._DiscreteMixin.default_xvar()))
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_real)
        self.assertFalse(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)

    def test_period(self):
        ''' Step: period.
        '''
        # escalón discreto
        d = ds.Step(3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

    def test_repr_str_latex(self):
        ''' Step: repr, str and latex.
        '''
        # escalón discreto
        d = ds.Step(3)
        # repr
        self.assertEqual(repr(d), 'Step(3)')
        # str
        self.assertEqual(str(d), 'u[n - 3]')
        # escalón discreto
        d = ds.Step(-5)
        # repr
        self.assertEqual(repr(d), 'Step(-5)')
        # str
        self.assertEqual(str(d), 'u[n + 5]')

    def test_eval_sample(self):
        ''' Step: eval(scalar), eval(range)
        '''
        # scalar
        d = ds.Step()
        self.assertAlmostEqual(d.eval(0), 1)
        self.assertAlmostEqual(d.eval(1), 1)
        self.assertAlmostEqual(d.eval(-1), 0)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([0, 1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([1, 1, 0]))
        # scalar
        d = ds.Step(1)
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
                                             np.array([1, 1, 0]))
        with self.assertRaises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        ''' Step (discrete): eval[scalar], eval[slice] '''
        # scalar
        d = ds.Step()
        self.assertAlmostEqual(d[0], 1)
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[-1], 0)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([0, 1, 1]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0, 0, 1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([1, 1, 0]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]
        # scalar
        d = ds.Step(-1)
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
                                             np.array([1, 1, 1]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]

    def test_generator(self):
        ''' Step (discrete): generate '''
        d = ds.Step()
        with self.assertRaises(ValueError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 0, 1, 1]))
        np.testing.assert_array_equal(next(dg), np.array([0, 0, 1, 1, 1]))
        np.testing.assert_array_equal(next(dg), np.array([0, 1, 1, 1, 1]))

    def test_flip(self):
        ''' Step (discrete): flip '''
        d = ds.Step().flip()
        np.testing.assert_array_equal(d[-3:3], np.array([1, 1, 1, 1, 0, 0]))
        d = ds.Step(1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([1, 1, 1, 0, 0, 0]))
        d = ds.Step(-1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([1, 1, 1, 1, 1, 0]))

    def test_shift_delay(self):
        ''' Step (discrete): shift, delay '''
        d = ds.Step()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.Step().shift(2)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 0, 0, 1]))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.Step().delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 1, 1, 1, 1, 1]))

    def test_scale(self):
        ''' Step (discrete): shift, delay '''
        d = ds.Step()
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.Step().scale(3)
        np.testing.assert_array_equal(d[-3:3], np.array([0, 0, 0, 1, 1, 1]))
        d = ds.Step().scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 1, 1, 1]))
        d = ds.Step(1).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([0, 0, 0, 0, 1, 1]))

if __name__ == "__main__":
    unittest.main()
