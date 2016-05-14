import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.printer as pt
import skdsp.signal.signal as sg
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest


class RectTest(unittest.TestCase):

    def test_constructor(self):
        ''' Rect (discrete/continuous): constructors '''
        # rectángulo discreto
        d = ds.Rect(3)
        self.assertIsInstance(d, sg.Signal)
        self.assertIsInstance(d, sg.FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        self.assertTrue(is_discrete(d))
        self.assertFalse(is_continuous(d))
        # rectángulo continuo
        d = cs.Rect(3)
        self.assertIsInstance(d, sg.Signal)
        self.assertIsInstance(d, sg.FunctionSignal)
        self.assertIsInstance(d, cs.ContinuousFunctionSignal)
        self.assertFalse(is_discrete(d))
        self.assertTrue(is_continuous(d))

    def test_eval_sample(self):
        ''' Rect (discrete/continuous): eval(scalar) '''
        # rectángulo discreto
        d = ds.Rect(3)
        self.assertEqual(d.eval(0), 1.0)
        self.assertEqual(d.eval(1), 1.0)
        self.assertEqual(d.eval(-1), 1.0)
        with self.assertRaises(TypeError):
            d.eval(0.5)
        # rectángulo continuo
        d = cs.Rect(3)
        self.assertEqual(d.eval(0), 1.0)
        self.assertEqual(d.eval(1), 1.0)
        self.assertEqual(d.eval(-1), 1.0)

    def test_eval_range(self):
        ''' Rect (discrete/continuous): eval(array) '''
        # rectángulo discreto
        d = ds.Rect(3)
        np.testing.assert_array_equal(d.eval(np.arange(0, 2)),
                                      np.array([1.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-1, 2)),
                                      np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-4, 1, 2)),
                                      np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
                                      np.array([0.0, 1.0, 1.0]))
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))
        # rectángulo continuo
        d = cs.Rect(3)
        np.testing.assert_array_equal(d.eval(np.arange(0, 2)),
                                      np.array([1.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-1, 2)),
                                      np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-4, 1, 2)),
                                      np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
                                      np.array([0.0, 1.0, 1.0]))

    def test_getitem_scalar(self):
        ''' Rect (discrete/continuous): eval[scalar] '''
        # rectángulo discreto
        d = ds.Rect(3)
        self.assertEqual(d[0], 1.0)
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d[-1], 1.0)
        with self.assertRaises(TypeError):
            d[0.5]
        # rectángulo discreto
        d = cs.Rect(3)
        self.assertEqual(d[0], 1.0)
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d[-1], 1.0)

    def test_getitem_slice(self):
        ''' Rect (discrete/continuous): eval[:] '''
        # rectángulo discreto
        d = ds.Rect(3)
        np.testing.assert_array_equal(d[0:2], np.array([1.0, 1.0]))
        np.testing.assert_array_equal(d[-1:2], np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([0.0, 1.0, 1.0]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]
        # rectángulo continuo
        d = cs.Rect(3)
        np.testing.assert_array_equal(d[0:2], np.array([1.0, 1.0]))
        np.testing.assert_array_equal(d[-1:2], np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, 1]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([0.0, 1.0, 1.0]))

    def test_dtype(self):
        ''' Rect (discrete/continuous): dtype '''
        # rectángulo discreto
        d = ds.Rect(3)
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_
        # rectángulo continuo
        d = cs.Rect(3)
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_

    def test_name(self):
        ''' Rect (discrete/continuous): name '''
        # rectángulo discreto
        d = ds.Rect(3)
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        # rectángulo continuo
        d = ds.Rect(3)
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar(self):
        ''' Rect (discrete/continuous): free variable '''
        # rectángulo discreto
        d = ds.Rect(3) >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'Pi[n - 3, {0}]'.format(d.width))
        d.xvar = sp.symbols('m', integer=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'Pi[m - 3, {0}]'.format(d.width))
        # rectángulo continuo
        d = cs.Rect(3) >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'Pi(t - 3, {0})'.format(d.width))
        d.xvar = sp.symbols('x', real=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'Pi(x - 3, {0})'.format(d.width))

    def test_latex(self):
        ''' Rect (discrete/continuous): latex '''
        # rectángulo discreto
        d = ds.Rect()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\Pi\left[n/16\right]$')
        d = ds.Rect(8, 3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\Pi\left[(n - 3)/8\right]$')
        # rectángulo continuo
        d = cs.Rect()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\Pi\left(t/16\right)$')
        d = cs.Rect(8, 3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\Pi\left((t - 3)/8\right)$')

if __name__ == "__main__":
    unittest.main()
