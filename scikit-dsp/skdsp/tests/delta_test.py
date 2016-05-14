import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.signal as sg
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest


class DeltaTest(unittest.TestCase):

    def test_constructor(self):
        ''' Delta (discrete/continuous): constructors '''
        # delta discreta
        d = ds.Delta()
        self.assertIsInstance(d, sg.Signal)
        self.assertIsInstance(d, sg.FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        self.assertTrue(is_discrete(d))
        self.assertFalse(is_continuous(d))
        # delta continua
        d = cs.Delta()
        self.assertIsInstance(d, sg.Signal)
        self.assertIsInstance(d, sg.FunctionSignal)
        self.assertIsInstance(d, cs.ContinuousFunctionSignal)
        self.assertFalse(is_discrete(d))
        self.assertTrue(is_continuous(d))

    def test_eval_sample(self):
        ''' Delta (discrete/continuous): eval(scalar) '''
        # delta discreta
        d = ds.Delta()
        self.assertEqual(d.eval(0), 1.0)
        self.assertEqual(d.eval(1), 0.0)
        self.assertEqual(d.eval(-1), 0.0)
        with self.assertRaises(TypeError):
            d.eval(0.5)
        # delta continua
        d = cs.Delta()
        self.assertTrue(np.isnan(d.eval(0)))
        self.assertEqual(d.eval(1), 0.0)
        self.assertEqual(d.eval(-1), 0.0)
        self.assertEqual(d.eval(0.5), 0.0)

    def test_eval_range(self):
        ''' Delta (discrete/continuous): eval(array) '''
        # delta discreta
        d = ds.Delta()
        expected = np.array([1.0, 0.0])
        actual = d.eval(np.arange(0, 2))
        np.testing.assert_array_equal(expected, actual)
        expected = np.array([0.0, 1.0, 0.0])
        actual = d.eval(np.arange(-1, 2))
        np.testing.assert_array_equal(expected, actual)
        expected = np.array([0.0, 0.0, 1.0])
        actual = d.eval(np.arange(-4, 1, 2))
        np.testing.assert_array_equal(expected, actual)
        expected = np.array([0.0, 0.0, 0.0])
        actual = d.eval(np.arange(3, -2, -2))
        np.testing.assert_array_equal(expected, actual)
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))
        # delta continua
        d = cs.Delta()
        expected = np.array([np.nan, 0.0])
        actual = d.eval(np.arange(0, 2))
        np.testing.assert_array_equal(expected, actual)
        expected = np.array([0.0, np.nan, 0.0])
        actual = d.eval(np.arange(-1, 2))
        np.testing.assert_array_equal(expected, actual)
        expected = np.array([0.0, 0.0, np.nan])
        actual = d.eval(np.arange(-4, 1, 2))
        np.testing.assert_array_equal(expected, actual)
        expected = np.array([0.0, 0.0, 0.0])
        actual = d.eval(np.arange(3, -2, -2))
        np.testing.assert_array_equal(expected, actual)

    def test_getitem_scalar(self):
        ''' Delta (discrete/continuous): eval[scalar] '''
        # delta discreta
        d = ds.Delta()
        self.assertEqual(d[0], 1.0)
        self.assertEqual(d[1], 0.0)
        self.assertEqual(d[-1], 0.0)
        with self.assertRaises(TypeError):
            d[0.5]
        # delta continua
        d = cs.Delta()
        self.assertTrue(np.isnan(d[0]))
        self.assertEqual(d[1], 0.0)
        self.assertEqual(d[-1], 0.0)

    def test_getitem_slice(self):
        ''' Delta (discrete/continuous): eval[:] '''
        # delta discreta
        d = ds.Delta()
        np.testing.assert_array_equal(d[0:2], np.array([1.0, 0.0]))
        np.testing.assert_array_equal(d[-1:2], np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([0.0, 0.0, 0.0]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]
        # delta continua
        d = cs.Delta()
        np.testing.assert_array_equal(d[0:2], np.array([np.nan, 0.0]))
        np.testing.assert_array_equal(d[-1:2], np.array([0.0, np.nan, 0.0]))
        np.testing.assert_array_equal(d[-4:1:2], np.array([0.0, 0.0, np.nan]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([0.0, 0.0, 0.0]))

    def test_dtype(self):
        ''' Delta (discrete/continuous): dtype '''
        # delta discreta
        d = ds.Delta()
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_
        # delta continua
        d = cs.Delta()
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_

    def test_name(self):
        ''' Delta (discrete/continuous): name '''
        # delta discreta
        d = ds.Delta()
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        # delta continua
        d = cs.Delta()
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar(self):
        ''' Delta (discrete/continuous): free variable '''
        # delta discreta
        d = ds.Delta() >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'd[n - 3]')
        d.xvar = sp.symbols('m', integer=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'd[m - 3]')
        # delta continua
        d = cs.Delta() >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'd(t - 3)')
        d.xvar = sp.symbols('u', real=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'd(u - 3)')

if __name__ == "__main__":
    unittest.main()
