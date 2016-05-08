from skdsp.signal.discrete import Square, DiscreteFunctionSignal, \
    DiscreteMixin
from skdsp.signal.signal import Signal, FunctionSignal
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex

import numpy as np
import sympy as sp
import unittest


class SquareTest(unittest.TestCase):

    def square(self, N, w, n0):
        n0 = n0 % N
        if n0 < w:
            return 1
        else:
            return -1

    def test_constructor(self):
        s = Square()
        self.assertIsInstance(s, Signal)
        self.assertIsInstance(s, FunctionSignal)
        self.assertIsInstance(s, DiscreteMixin)
        self.assertIsInstance(s, DiscreteFunctionSignal)
        self.assertTrue(is_discrete(s))
        self.assertFalse(is_continuous(s))
        N = 12
        width = 4
        s = Square(N, width)
        self.assertEqual(str(s), 'square[n, {0}, {1}]'.format(N, width))

    def test_eval_sample(self):
        N = 12
        w = 4
        d = Square(N, w)
        self.assertEqual(d.eval(0), self.square(N, w, 0))
        self.assertEqual(d.eval(1), self.square(N, w, 1))
        self.assertEqual(d.eval(2), self.square(N, w, 2))
        self.assertEqual(d.eval(-1), self.square(N, w, -1))
        self.assertEqual(d.eval(-2), self.square(N, w, -2))
        with self.assertRaises(TypeError):
            d.eval(0.5)

    def test_eval_range(self):
        N = 12
        w = 4
        d = Square(N, w)
        n = np.arange(0, 3)
        expected = np.zeros((len(n)))
        for i, n0 in enumerate(n):
            expected[i] = self.square(N, w, n0)
        np.testing.assert_array_equal(d.eval(n), expected)
        n = np.arange(-1, 3)
        expected = np.zeros((len(n)))
        for i, n0 in enumerate(n):
            expected[i] = self.square(N, w, n0)
        np.testing.assert_array_equal(d.eval(n), expected)
        n = np.arange(-4, 3, 2)
        expected = np.zeros((len(n)))
        for i, n0 in enumerate(n):
            expected[i] = self.square(N, w, n0)
        np.testing.assert_array_equal(d.eval(n), expected)
        n = np.arange(3, -2, -2)
        expected = np.zeros((len(n)))
        for i, n0 in enumerate(n):
            expected[i] = self.square(N, w, n0)
        np.testing.assert_array_equal(d.eval(n), expected)
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        N = 12
        w = 4
        d = Square(N, w)
        self.assertEqual(d[0], self.square(N, w, 0))
        self.assertEqual(d[1], self.square(N, w, 1))
        self.assertEqual(d[2], self.square(N, w, 2))
        self.assertEqual(d[-1], self.square(N, w, -1))
        self.assertEqual(d[-2], self.square(N, w, -2))
        with self.assertRaises(TypeError):
            d[0.5]

    def test_getitem_slice(self):
        N = 12
        w = 4
        d = Square(N, w)
        n = np.arange(0, 3)
        expected = np.zeros((len(n)))
        for i, n0 in enumerate(n):
            expected[i] = self.square(N, w, n0)
        np.testing.assert_array_equal(d[0:3], expected)
        n = np.arange(-1, 3)
        expected = np.zeros((len(n)))
        for i, n0 in enumerate(n):
            expected[i] = self.square(N, w, n0)
        np.testing.assert_array_equal(d[-1:3], expected)
        n = np.arange(-4, 2, 2)
        expected = np.zeros((len(n)))
        for i, n0 in enumerate(n):
            expected[i] = self.square(N, w, n0)
        np.testing.assert_array_equal(d[-4:2:2], expected)
        n = np.arange(3, -2, -2)
        expected = np.zeros((len(n)))
        for i, n0 in enumerate(n):
            expected[i] = self.square(N, w, n0)
        np.testing.assert_array_equal(d[3:-2:-2], expected)
        with self.assertRaises(TypeError):
            d[0:2:0.5]

    def test_dtype(self):
        N = 12
        w = 4
        d = Square(N, w)
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_

    def test_name(self):
        N = 12
        w = 4
        d = Square(N, w)
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar(self):
        N = 12
        w = 4
        shift = 3
        d = Square(N, w) >> shift
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d),
                         'square[n - {0}, {1}, {2}]'.format(shift, N, w))
        d.xvar = sp.symbols('m', integer=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d),
                         'square[m - {0}, {1}, {2}]'.format(shift, N, w))

    def test_period(self):
        N = 12
        w = 4
        d = Square(N, w)
        self.assertTrue(d.is_periodic())
        self.assertEqual(d.period, N)

    def test_dfs(self):
        N = 12
        w = 4
        d = Square(N, w)
        X = d.dfs()
        print(X)
        X = d.dfs(symbolic=True)
        print(X)

if __name__ == "__main__":
    unittest.main()
