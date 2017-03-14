from skdsp.signal._signal import _Signal, _FunctionSignal
import skdsp.signal.discrete as ds
# import skdsp.signal.continuous as cs
# import skdsp.signal.printer as pt
# import skdsp.signal._util as u
import numpy as np
import sympy as sp
import unittest


class ConstantTest(unittest.TestCase):

    def test_constructor(self):
        ''' Constant (discrete/continuous): constructors.
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

    def test_name(self):
        ''' Constant (discrete/continuous): name.
        '''
        # constante discreta
        d = ds.Constant(3)
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar_xexpr(self):
        ''' Constant (discrete/continuous): temporal variable and expression.
        '''
        # constante discreta
        d = ds.Constant(3)
        # variable temporal
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

    def test_yexpr(self):
        ''' Constant (discrete/continuous): function expression.
        '''
        # constante discreta
        cte = 3
        d = ds.Constant(cte)
        # variable dependiente
        self.assertEqual(d.yexpr, sp.Expr(cte))
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_real)
        self.assertFalse(d.is_complex)
        # constante discreta
        cte = 3j
        d = ds.Constant(cte)
        # variable dependiente
        self.assertEqual(d.yexpr, sp.Expr(cte))
        self.assertTrue(np.issubdtype(d.dtype, np.complex))
        self.assertFalse(d.is_real)
        self.assertTrue(d.is_complex)

    def test_period(self):
        ''' Constant (discrete/continuous): period.
        '''
        # constante discreta
        d = ds.Constant(3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

    def test_repr_str_latex(self):
        ''' Constant (discrete/continuous): repr, str and latex.
        '''
        # constante discreta
        d = ds.Constant(3)
        # repr
        self.assertEqual(repr(d), 'Constant(3)')
        # str
        self.assertEqual(str(d), '3')
        # latex
        self.assertEqual(d.latex(), '3')
        # constante discreta
        d = ds.Constant(-5.10-3.03j)
        # repr
        self.assertEqual(repr(d), 'Constant(-5.1 - 3.03*j)')
        # str
        self.assertEqual(str(d), '-5.1 - 3.03*j')
        # latex
        self.assertEqual(d.latex(), '-5.1 - 3.03*\\mathrm{j}')
        # constante discreta
        d = ds.Constant(-3j)
        # repr
        self.assertEqual(repr(d), 'Constant(-3.0*j)')
        # str
        self.assertEqual(str(d), '-3.0*j')
        # latex
        self.assertEqual(d.latex(), '-3.0*\\mathrm{j}')

    def test_repr_str_latex(self):
        ''' Constant (discrete/continuous): repr, str and latex.
        '''

#     def test_eval_sample(self):
#         ''' Rect (discrete/continuous): eval(scalar) '''
#         # rectángulo discreto
#         d = ds.Rect(3)
#         self.assertEqual(d.eval(0), 1.0)
#         self.assertEqual(d.eval(1), 1.0)
#         self.assertEqual(d.eval(-1), 1.0)
#         with self.assertRaises(TypeError):
#             d.eval(0.5)
#         # rectángulo continuo
#         d = cs.Rect(3)
#         self.assertEqual(d.eval(0), 1.0)
#         self.assertEqual(d.eval(1), 1.0)
#         self.assertEqual(d.eval(-1), 1.0)

#     def test_eval_range(self):
#         ''' Rect (discrete/continuous): eval(array) '''
#         # rectángulo discreto
#         d = ds.Rect(3)
#         np.testing.assert_array_equal(d.eval(np.arange(0, 2)),
#                                       np.array([1.0, 1.0]))
#         np.testing.assert_array_equal(d.eval(np.arange(-1, 2)),
#                                       np.array([1.0, 1.0, 1.0]))
#         np.testing.assert_array_equal(d.eval(np.arange(-4, 1, 2)),
#                                       np.array([0.0, 0.0, 1.0]))
#         np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
#                                       np.array([0.0, 1.0, 1.0]))
#         with self.assertRaises(TypeError):
#             d.eval(np.arange(0, 2, 0.5))
#         # rectángulo continuo
#         d = cs.Rect(3)
#         np.testing.assert_array_equal(d.eval(np.arange(0, 2)),
#                                       np.array([1.0, 1.0]))
#         np.testing.assert_array_equal(d.eval(np.arange(-1, 2)),
#                                       np.array([1.0, 1.0, 1.0]))
#         np.testing.assert_array_equal(d.eval(np.arange(-4, 1, 2)),
#                                       np.array([0.0, 0.0, 1.0]))
#         np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
#                                       np.array([0.0, 1.0, 1.0]))

#     def test_getitem_scalar(self):
#         ''' Rect (discrete/continuous): eval[scalar] '''
#         # rectángulo discreto
#         d = ds.Rect(3)
#         self.assertEqual(d[0], 1.0)
#         self.assertEqual(d[1], 1.0)
#         self.assertEqual(d[-1], 1.0)
#         with self.assertRaises(TypeError):
#             d[0.5]
#         # rectángulo discreto
#         d = cs.Rect(3)
#         self.assertEqual(d[0], 1.0)
#         self.assertEqual(d[1], 1.0)
#         self.assertEqual(d[-1], 1.0)

#     def test_getitem_slice(self):
#         ''' Rect (discrete/continuous): eval[:] '''
#         # rectángulo discreto
#         d = ds.Rect(3)
#         np.testing.assert_array_equal(d[0:2], np.array([1.0, 1.0]))
#         np.testing.assert_array_equal(d[-1:2], np.array([1.0, 1.0, 1.0]))
#         np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, 1.0]))
#         np.testing.assert_array_equal(d[3:-2:-2], np.array([0.0, 1.0, 1.0]))
#         with self.assertRaises(TypeError):
#             d[0:2:0.5]
#         # rectángulo continuo
#         d = cs.Rect(3)
#         np.testing.assert_array_equal(d[0:2], np.array([1.0, 1.0]))
#         np.testing.assert_array_equal(d[-1:2], np.array([1.0, 1.0, 1.0]))
#         np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, 1]))
#         np.testing.assert_array_equal(d[3:-2:-2], np.array([0.0, 1.0, 1.0]))

if __name__ == "__main__":
    unittest.main()
