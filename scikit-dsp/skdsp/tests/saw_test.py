from skdsp.signal.discrete import SawTooth, DiscreteFunctionSignal, \
    DiscreteMixin
from skdsp.signal.signal import Signal, FunctionSignal
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex

import numpy as np
import sympy as sp
import unittest


class SawToothTest(unittest.TestCase):

    def test_constructor(self):
        s = SawTooth()
        self.assertIsInstance(s, Signal)
        self.assertIsInstance(s, FunctionSignal)
        self.assertIsInstance(s, DiscreteMixin)
        self.assertIsInstance(s, DiscreteFunctionSignal)
        self.assertTrue(is_discrete(s))
        self.assertFalse(is_continuous(s))
        s = SawTooth(8)
        self.assertEqual(str(s), 'saw[n, 8]')

    def test_eval_sample(self):
        N = 12
        d = SawTooth(N)
        self.assertEqual(d.eval(0), 0.0)
        self.assertEqual(d.eval(1), 1.0)
        self.assertEqual(d.eval(2), 2.0)
        self.assertEqual(d.eval(-1), -1 % N)
        self.assertEqual(d.eval(-2), -2 % N)
        with self.assertRaises(TypeError):
            d.eval(0.5)

    def test_eval_range(self):
        N = 12
        d = SawTooth(N)
        n = np.arange(0, 3)
        np.testing.assert_array_equal(d.eval(n), np.mod(n, N))
        n = np.arange(-1, 3)
        np.testing.assert_array_equal(d.eval(n), np.mod(n, N))
        n = np.arange(-4, 3, 2)
        np.testing.assert_array_equal(d.eval(n), np.mod(n, N))
        n = np.arange(3, -2, -2)
        np.testing.assert_array_equal(d.eval(n), np.mod(n, N))
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        N = 12
        d = SawTooth(N)
        self.assertEqual(d[0], 0.0)
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d[2], 2.0)
        self.assertEqual(d[-1], -1 % N)
        self.assertEqual(d[-2], -2 % N)
        with self.assertRaises(TypeError):
            d[0.5]

    def test_getitem_slice(self):
        N = 12
        d = SawTooth(N)
        np.testing.assert_array_equal(d[0:3], np.mod(np.arange(0, 3), N))
        np.testing.assert_array_equal(d[-1:3], np.mod(np.arange(-1, 3), N))
        np.testing.assert_array_equal(d[-4:2:2],
                                      np.mod(np.arange(-4, 2, 2), N))
        np.testing.assert_array_equal(d[3:-2:-2],
                                      np.mod(np.arange(3, -2, -2), N))
        with self.assertRaises(TypeError):
            d[0:2:0.5]

    def test_dtype(self):
        N = 12
        d = SawTooth(N)
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_

    def test_name(self):
        N = 12
        d = SawTooth(N)
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar(self):
        N = 12
        d = SawTooth(N) >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'saw[n - 3, {0}]'.format(N))
        d.xvar = sp.symbols('m', integer=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'saw[m - 3, {0}]'.format(N))

    def test_period(self):
        N = 12
        d = SawTooth(N) >> 3
        self.assertTrue(d.is_periodic())
        self.assertEqual(d.period, N)

if __name__ == "__main__":
    unittest.main()
