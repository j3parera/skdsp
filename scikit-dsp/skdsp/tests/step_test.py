import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.printer as pt
import skdsp.signal._signal as sg
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest


class StepTest(unittest.TestCase):

    def test_constructor(self):
        ''' Step (discrete/continuous): constructors '''
        # escalón discreto
        d = ds.Step()
        self.assertIsInstance(d, sg._Signal)
        self.assertIsInstance(d, sg._FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        self.assertTrue(is_discrete(d))
        self.assertFalse(is_continuous(d))
        # escalón continuo
        d = cs.Step()
        self.assertIsInstance(d, sg._Signal)
        self.assertIsInstance(d, sg._FunctionSignal)
        self.assertIsInstance(d, cs.ContinuousFunctionSignal)
        self.assertFalse(is_discrete(d))
        self.assertTrue(is_continuous(d))

    def test_eval_sample(self):
        ''' Step (discrete/continuous): eval(scalar) '''
        # escalón discreto
        d = ds.Step()
        self.assertEqual(d.eval(0), 1.0)
        self.assertEqual(d.eval(1), 1.0)
        self.assertEqual(d.eval(-1), 0.0)
        with self.assertRaises(TypeError):
            d.eval(0.5)
        # escalón continuo
        d = cs.Step()
        self.assertTrue(np.isnan(d.eval(0)))
        self.assertEqual(d.eval(1), 1.0)
        self.assertEqual(d.eval(-1), 0.0)

    def test_eval_range(self):
        ''' Step (discrete/continuous): eval(array) '''
        # escalón discreto
        d = ds.Step()
        np.testing.assert_array_equal(d.eval(np.arange(0, 2)),
                                      np.array([1.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-1, 2)),
                                      np.array([0.0, 1.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-4, 1, 2)),
                                      np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
                                      np.array([1.0, 1.0, 0.0]))
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))
        # escalón continuo
        d = cs.Step()
        np.testing.assert_array_equal(d.eval(np.arange(0, 2)),
                                      np.array([np.nan, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-1, 2)),
                                      np.array([0.0, np.nan, 1.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-4, 1, 2)),
                                      np.array([0.0, 0.0, np.nan]))
        np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
                                      np.array([1.0, 1.0, 0.0]))

    def test_getitem_scalar(self):
        ''' Step (discrete/continuous): eval[scalar] '''
        # escalón discreto
        d = ds.Step()
        self.assertEqual(d[0], 1.0)
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d[-1], 0.0)
        with self.assertRaises(TypeError):
            d[0.5]
        # escalón discreto
        d = cs.Step()
        self.assertTrue(np.isnan(d[0]))
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d[-1], 0.0)

    def test_getitem_slice(self):
        ''' Step (discrete/continuous): eval[:] '''
        # escalón discreto
        d = ds.Step()
        np.testing.assert_array_equal(d[0:2], np.array([1.0, 1.0]))
        np.testing.assert_array_equal(d[-1:2], np.array([0.0, 1.0, 1.0]))
        np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([1.0, 1.0, 0.0]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]
        # escalón continuo
        d = cs.Step()
        np.testing.assert_array_equal(d[0:2], np.array([np.nan, 1.0]))
        np.testing.assert_array_equal(d[-1:2], np.array([0.0, np.nan, 1.0]))
        np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, np.nan]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([1.0, 1.0, 0.0]))

    def test_dtype(self):
        ''' Step (discrete/continuous): dtype '''
        # escalón discreto
        d = ds.Step()
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_
        # escalón continuo
        d = cs.Step()
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_

    def test_name(self):
        ''' Step (discrete/continuous): name '''
        # escalón discreto
        d = ds.Step()
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        # escalón continuo
        d = ds.Step()
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar(self):
        ''' Step (discrete/continuous): free variable '''
        # escalón discreto
        d = ds.Step() >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'u[n - 3]')
        d.xvar = sp.symbols('m', integer=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'u[m - 3]')
        # escalón continuo
        d = cs.Step() >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'u(t - 3)')
        d.xvar = sp.symbols('x', real=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'u(x - 3)')

    def test_latex(self):
        ''' Step (discrete/continuous): latex '''
        # escalón discreto
        d = ds.Step()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$u\left[n\right]$')
        d = ds.Step(3)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$u\left[n - 3\right]$')
        # escalón continuo
        d = cs.Step()
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$u\left(t\right)$')
        d = cs.Step(1.5)
        self.assertEqual(pt.latex(d, mode='inline'),
                         r'$u\left(t - 1.5\right)$')

if __name__ == "__main__":
    unittest.main()
