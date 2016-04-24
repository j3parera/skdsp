from skdsp.signal.discrete import Step, DiscreteFunctionSignal, DiscreteMixin
from skdsp.signal.signal import Signal, FunctionSignal
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex

import numpy as np
import sympy as sp
import unittest


class StepTest(unittest.TestCase):

    def test_constructor(self):
        d = Step()
        self.assertIsInstance(d, Signal)
        self.assertIsInstance(d, FunctionSignal)
        self.assertIsInstance(d, DiscreteMixin)
        self.assertIsInstance(d, DiscreteFunctionSignal)
        self.assertTrue(is_discrete(d))
        self.assertFalse(is_continuous(d))

    def test_eval_sample(self):
        d = Step()
        self.assertEqual(d.eval(0), 1.0)
        self.assertEqual(d.eval(1), 1.0)
        self.assertEqual(d.eval(-1), 0.0)
        with self.assertRaises(TypeError):
            d.eval(0.5)

    def test_eval_range(self):
        d = Step()
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

    def test_getitem_scalar(self):
        d = Step()
        self.assertEqual(d[0], 1.0)
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d[-1], 0.0)
        with self.assertRaises(TypeError):
            d[0.5]

    def test_getitem_slice(self):
        d = Step()
        np.testing.assert_array_equal(d[0:2], np.array([1.0, 1.0]))
        np.testing.assert_array_equal(d[-1:2], np.array([0.0, 1.0, 1.0]))
        np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([1.0, 1.0, 0.0]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]

    def test_dtype(self):
        d = Step()
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_

    def test_name(self):
        d = Step()
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar(self):
        d = Step() >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'u[n - 3]')
        d.xvar = sp.symbols('m', integer=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'u[m - 3]')

if __name__ == "__main__":
    unittest.main()
