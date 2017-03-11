from skdsp.signal._signal import _Signal, _FunctionSignal
from skdsp.signal.discrete import DiscreteFunctionSignal
from skdsp.signal.discrete import Constant as dConstant
# import skdsp.signal.continuous as cs
# import skdsp.signal.printer as pt
# import skdsp.signal._util as u
import numpy as np
import sympy as sp
import unittest


class ConstantTest(unittest.TestCase):

    def test_constructor(self):
        ''' Constant (discrete/continuous): constructors '''
        # constante discreta
        d = dConstant(3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, DiscreteFunctionSignal)

    def test_name(self):
        ''' Constant (discrete/continuous): name '''
        # constante discreta
        d = dConstant(3)
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
 
#     def test_xvar(self):
#         ''' Rect (discrete/continuous): free variable '''
#         # rectángulo discreto
#         d = ds.Rect(3) >> 3
#         self.assertEqual(d.name, 'x')
#         self.assertEqual(str(d), 'Pi[n - 3, {0}]'.format(d.width))
#         d.xvar = sp.symbols('m', integer=True)
#         self.assertEqual(d.name, 'x')
#         self.assertEqual(str(d), 'Pi[m - 3, {0}]'.format(d.width))

    def test_xvar(self):
        ''' Constant (discrete/continuous): temporal variable '''
        # constante discreta
        d = dConstant(3)
        # variable temporal
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, sp.Expr(d.xvar))
        # shift
        shift = 5
        d = d >> shift
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d._default_xvar())
        self.assertEqual(d.xexpr, sp.Expr(d.xvar - shift))

    def test_yexpr(self):
        ''' Constant (discrete/continuous): expression '''
        # constante discreta
        cte = 3
        d = dConstant(cte)
        # variable dependiente
        self.assertEqual(d.yexpr, sp.Expr(cte))
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_real)
        self.assertFalse(d.is_complex)
        # constante discreta
        cte = 3j
        d = dConstant(cte)
        # variable dependiente
        self.assertEqual(d.yexpr, sp.Expr(cte))
        self.assertTrue(np.issubdtype(d.dtype, np.complex))
        self.assertFalse(d.is_real)
        self.assertTrue(d.is_complex)

    def test_period(self):
        ''' Constant (discrete/continuous): period '''
        # constante discreta
        d = dConstant(3)
        # periodicidad
        self.assertFalse(d.is_periodic)
        self.assertEqual(d.period, sp.oo)

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
# 
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
# 
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
# 
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

#     def test_latex(self):
#         ''' Rect (discrete/continuous): latex '''
#         # rectángulo discreto
#         d = ds.Rect()
#         self.assertEqual(pt.latex(d, mode='inline'),
#                          r'$\Pi\left[n/16\right]$')
#         d = ds.Rect(8, 3)
#         self.assertEqual(pt.latex(d, mode='inline'),
#                          r'$\Pi\left[(n - 3)/8\right]$')
#         # rectángulo continuo
#         d = cs.Rect()
#         self.assertEqual(pt.latex(d, mode='inline'),
#                          r'$\Pi\left(t/16\right)$')
#         d = cs.Rect(8, 3)
#         self.assertEqual(pt.latex(d, mode='inline'),
#                          r'$\Pi\left((t - 3)/8\right)$')

if __name__ == "__main__":
    unittest.main()
