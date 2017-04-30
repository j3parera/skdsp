from skdsp.signal._signal import _Signal, _FunctionSignal
import numpy as np
import skdsp.signal.discrete as ds
import sympy as sp
import unittest
from skdsp.signal.discrete import _DiscreteMixin


class ConstantTest(unittest.TestCase):

    def test_constructor(self):
        ''' Constant: constructors.
        '''
        # constante discreta
        d = ds.Constant()
        self.assertIsNotNone(d)
        # constante discreta
        d = ds.Constant(3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # no constante
        with self.assertRaises(ValueError):
            d = ds.Constant(_DiscreteMixin._default_xvar())

    def test_name(self):
        ''' Constant: name.
        '''
        # constante discreta
        d = ds.Constant(3, name='y0')
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
        d = ds.Constant(3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name(), 'y_{0}')
        self.assertEqual(d.latex_name('inline'), '$y_{0}$')
        del d
        d = ds.Constant(3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name(), 'yupi')
        self.assertEqual(d.latex_name('inline'), '$yupi$')

    def test_xvar_xexpr(self):
        ''' Constant: independent variable and expression.
        '''
        # constante discreta
        d = ds.Constant(3)
        # variable independiente
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, sp.Expr(d.xvar))
        # shift
        shift = 5
        d = ds.Constant(3).shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, sp.Expr(d.xvar - shift))
        # flip
        d = ds.Constant(3).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, sp.Expr(-d.xvar))
        # shift and flip
        shift = 5
        d = ds.Constant(3).shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, sp.Expr(-d.xvar - shift))
        # flip and shift
        shift = 5
        d = ds.Constant(3).flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, sp.Expr(-d.xvar + shift))

    def test_yexpr_real_imag(self):
        ''' Constant: function expression.
        '''
        # constante discreta
        cte = 3.0
        d = ds.Constant(cte)
        # expresión
        self.assertEqual(d.yexpr, cte)
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_real)
        self.assertFalse(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)
        # constante discreta
        cte = -5-3j
        d = ds.Constant(cte)
        # expresión
        self.assertEqual(d.yexpr, cte)
        self.assertTrue(np.issubdtype(d.dtype, np.complex))
        self.assertFalse(d.is_real)
        self.assertTrue(d.is_complex)
        self.assertEqual(ds.Constant(-5.0), d.real)
        self.assertEqual(ds.Constant(-3.0), d.imag)

    def test_period(self):
        ''' Constant: period.
        '''
        # constante discreta
        d = ds.Constant(3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

    def test_repr_str_latex(self):
        ''' Constant: repr, str and latex.
        '''
        # constante discreta
        d = ds.Constant(3)
        # repr
        self.assertEqual(repr(d), 'Constant(3)')
        # str
        self.assertEqual(str(d), '3')
        # latex
        self.assertEqual(d.latex_yexpr(), '3')
        # constante discreta
        d = ds.Constant(-5.10-3.03j)
        # repr
        self.assertEqual(repr(d), 'Constant(-5.1 - 3.03*j)')
        # str
        self.assertEqual(str(d), '-5.1 - 3.03*j')
        # latex
        self.assertEqual(d.latex_yexpr(), '-5.1 - 3.03*\\mathrm{j}')
        # constante discreta
        d = ds.Constant(-3j)
        # repr
        self.assertEqual(repr(d), 'Constant(-3.0*j)')
        # str
        self.assertEqual(str(d), '-3.0*j')
        # latex
        self.assertEqual(d.latex_yexpr(), '-3.0*\\mathrm{j}')

    def test_eval_sample(self):
        ''' Constant: eval(scalar), eval(range)
        '''
        # scalar
        cte = complex(np.random.random(), np.random.random())
        d = ds.Constant(cte)
        self.assertAlmostEqual(d.eval(0), cte)
        self.assertAlmostEqual(d.eval(1), cte)
        self.assertAlmostEqual(d.eval(-1), cte)
        with self.assertRaises(TypeError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([cte, cte]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([cte, cte, cte]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([cte, cte, cte]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([cte, cte, cte]))
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        ''' Constant (discrete): eval[scalar], eval[slice] '''
        # scalar
        cte = complex(np.random.random(), np.random.random())
        d = ds.Constant(cte)
        self.assertAlmostEqual(d[0], cte)
        self.assertAlmostEqual(d[1], cte)
        self.assertAlmostEqual(d[-1], cte)
        with self.assertRaises(TypeError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([cte, cte]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([cte, cte, cte]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([cte, cte, cte]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([cte, cte, cte]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]

    def test_generator(self):
        ''' Constant (discrete): generate '''
        d = ds.Constant(123.456)
        with self.assertRaises(TypeError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=3)
        np.testing.assert_array_equal(next(dg), np.full(5, 123.456))
        np.testing.assert_array_equal(next(dg), np.full(5, 123.456))
        np.testing.assert_array_equal(next(dg), np.full(5, 123.456))

    def test_flip(self):
        ''' Constant (discrete): flip '''
        d = ds.Constant(123.456)
        np.testing.assert_array_equal(d[-10:10], d.flip()[-10:10])

    def test_shift_delay(self):
        ''' Constant (discrete): shift, delay '''
        d = ds.Constant(123.456)
        with self.assertRaises(TypeError):
            d.shift(0.5)
        np.testing.assert_array_equal(d[-10:10], d.shift(3)[-10:10])
        with self.assertRaises(TypeError):
            d.delay(0.5)
        np.testing.assert_array_equal(d[-10:10], d.delay(3)[-10:10])

    def test_scale(self):
        ''' Constant (discrete): shift, delay '''
        d = ds.Constant(123.456)
        with self.assertRaises(TypeError):
            d.scale(sp.pi)
        np.testing.assert_array_equal(d[-10:10], d.scale(3)[-10:10])
        np.testing.assert_array_equal(d[-10:10], d.scale(0.25)[-10:10])

if __name__ == "__main__":
    unittest.main()
