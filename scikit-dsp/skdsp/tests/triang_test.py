import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.printer as pt
import skdsp.signal.signals as sg
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest


class TriangTest(unittest.TestCase):

    def test_constructor(self):
        ''' Triang (discrete/continuous): constructors '''
        # triángulo discreto
        d = ds.Triang(5)
        self.assertIsInstance(d, sg.Signal)
        self.assertIsInstance(d, sg.FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        self.assertTrue(is_discrete(d))
        self.assertFalse(is_continuous(d))
        # triángulo continuo
        d = cs.Triang(5)
        self.assertIsInstance(d, sg.Signal)
        self.assertIsInstance(d, sg.FunctionSignal)
        self.assertIsInstance(d, cs.ContinuousFunctionSignal)
        self.assertFalse(is_discrete(d))
        self.assertTrue(is_continuous(d))

    def test_eval_sample(self):
        ''' Triang (discrete/continuous): eval(scalar) '''
        # triángulo discreto
        d = ds.Triang(5)
        self.assertEqual(d.eval(0), 1.0)
        self.assertEqual(d.eval(1), 0.6)
        self.assertEqual(d.eval(-1), 0.6)
        with self.assertRaises(TypeError):
            d.eval(0.5)
        # triángulo continuo
        d = cs.Triang(5)
        self.assertEqual(d.eval(0), 1.0)
        self.assertEqual(d.eval(1), 0.6)
        self.assertEqual(d.eval(-1), 0.6)

    def test_eval_range(self):
        ''' Triang (discrete/continuous): eval(array) '''
        # triángulo discreto
        d = ds.Triang(5)
        np.testing.assert_array_equal(d.eval(np.arange(0, 2)),
                                      np.array([1.0, 0.6]))
        np.testing.assert_array_equal(d.eval(np.arange(-1, 2)),
                                      np.array([0.6, 1.0, 0.6]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0.0, 0.2, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
                                      np.array([0.0, 0.6, 0.6]))
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))
        # triángulo continuo
        d = cs.Triang(5)
        np.testing.assert_array_equal(d.eval(np.arange(0, 2)),
                                      np.array([1.0, 0.6]))
        np.testing.assert_array_equal(d.eval(np.arange(-1, 2)),
                                      np.array([0.6, 1.0, 0.6]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([0.0, 0.2, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
                                      np.array([0.0, 0.6, 0.6]))

    def test_getitem_scalar(self):
        ''' Triang (discrete/continuous): eval[scalar] '''
        # triángulo discreto
        d = ds.Triang(5)
        self.assertEqual(d[0], 1.0)
        self.assertEqual(d[1], 0.6)
        self.assertEqual(d[-1], 0.6)
        with self.assertRaises(TypeError):
            d[0.5]
        # triángulo discreto
        d = cs.Triang(5)
        self.assertEqual(d[0], 1.0)
        self.assertEqual(d[1], 0.6)
        self.assertEqual(d[-1], 0.6)

    def test_getitem_slice(self):
        ''' Triang (discrete/continuous): eval[:] '''
        # triángulo discreto
        d = ds.Triang(5)
        np.testing.assert_array_equal(d[0:2], np.array([1.0, 0.6]))
        np.testing.assert_array_equal(d[-1:2], np.array([0.6, 1.0, 0.6]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0.0, 0.2, 1.0]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([0.0, 0.6, 0.6]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]
        # triángulo continuo
        d = cs.Triang(5)
        np.testing.assert_array_equal(d[0:2], np.array([1.0, 0.6]))
        np.testing.assert_array_equal(d[-1:2], np.array([0.6, 1.0, 0.6]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([0.0, 0.2, 1.0]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([0.0, 0.6, 0.6]))

    def test_dtype(self):
        ''' Triang (discrete/continuous): dtype '''
        # triángulo discreto
        d = ds.Triang(5)
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_
        # triángulo continuo
        d = cs.Triang(5)
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_

    def test_name(self):
        ''' Triang (discrete/continuous): name '''
        # triángulo discreto
        d = ds.Triang(5)
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        # triángulo continuo
        d = ds.Triang(5)
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar(self):
        ''' Triang (discrete/continuous): free variable '''
        # triángulo discreto
        d = ds.Triang(5) >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'Delta[n - 3, {0}]'.format(d.width))
        d.xvar = sp.symbols('m', integer=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'Delta[m - 3, {0}]'.format(d.width))
        # triángulo continuo
        d = cs.Triang(5) >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'Delta(t - 3, {0})'.format(d.width))
        d.xvar = sp.symbols('x', real=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'Delta(x - 3, {0})'.format(d.width))

    def test_latex(self):
        ''' Triang (discrete/continuous): latex '''
        # triángulo discreto
        d = ds.Triang()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\Delta\left[n/16\right]$')
        d = ds.Triang(8, 3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\Delta\left[(n - 3)/8\right]$')
        # triángulo continuo
        d = cs.Triang()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\Delta\left(t/16\right)$')
        d = cs.Triang(8, 3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$\Delta\left((t - 3)/8\right)$')

if __name__ == "__main__":
    unittest.main()
