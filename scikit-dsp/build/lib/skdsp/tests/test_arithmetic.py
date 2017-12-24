import unittest

import numpy as np
import skdsp.signal.discrete as ds


class ArithmeticTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        np.seterr(all='ignore')

    def test_add1(self):
        s = ds.Delta(-1) + ds.Step()
        np.testing.assert_array_equal(s[-5:6], np.r_[[0]*4, [1]*7])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Step)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)

    def test_add2(self):
        s = ds.Constant(0)
        for shift in range(0, 6):
            s += (ds.Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, [1]*6])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)

    def test_add3(self):
        s = ds.Delta() + 3
        np.testing.assert_equal(s[-5:6], np.r_[[3]*5, 4, [3]*5])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)
        s = ds.Delta() + 0
        self.assertIsInstance(s, ds.Delta)
        self.assertEqual(s, ds.Delta())
        s = 0 + ds.Delta()
        self.assertIsInstance(s, ds.Delta)
        self.assertEqual(s, ds.Delta())
        s = ds.Delta() + ds.Constant(0)
        self.assertIsInstance(s, ds.Delta)
        self.assertEqual(s, ds.Delta())
        s = ds.Constant(0) + ds.Delta()
        self.assertIsInstance(s, ds.Delta)
        s = ds.Delta()
        self.assertEqual(s, ds.Delta())
        self.assertEqual(s, s + 0)
        self.assertEqual(s, 0 + s)
        self.assertEqual(s, s + ds.Constant(0))
        self.assertEqual(s, ds.Constant(0) + s)

    def test_sub1(self):
        s = ds.Delta(-1) - ds.Step()
        np.testing.assert_array_equal(s[-5:6], np.r_[[0]*4, 1, [-1]*6])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Step)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)

    def test_sub2(self):
        s = ds.Constant(0)
        for shift in range(0, 6):
            s -= (ds.Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, [-1]*6])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)

    def test_sub3(self):
        s = ds.Delta() - 3
        np.testing.assert_equal(s[-5:6], np.r_[[-3]*5, -2, [-3]*5])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)
        s = 3 - ds.Delta()
        np.testing.assert_equal(s[-5:6], np.r_[[3]*5, 2, [3]*5])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)
        s = ds.Delta() - 0
        self.assertIsInstance(s, ds.Delta)
        s = 0 - ds.Delta()
        self.assertNotIsInstance(s, ds.Delta)
        s = ds.Delta() - ds.Constant(0)
        self.assertIsInstance(s, ds.Delta)
        s = ds.Constant(0) - ds.Delta()
        self.assertNotIsInstance(s, ds.Delta)
        s = ds.Delta()
        self.assertEqual(s, ds.Delta())
        self.assertEqual(s, s - 0)
        self.assertEqual(-s, 0 - s)
        self.assertEqual(s, s - ds.Constant(0))
        self.assertEqual(-s, ds.Constant(0) - s)

    def test_mul1(self):
        s = ds.Delta(1) * ds.Step()
        np.testing.assert_array_equal(s[-5:6], np.r_[[0]*6, 1, [0]*4])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Step)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)
        s = ds.Delta()
        self.assertEqual(ds.Constant(0), s*0)
        self.assertEqual(ds.Constant(0), 0*s)
        self.assertEqual(ds.Constant(0), s*ds.Constant(0))
        self.assertEqual(ds.Constant(0), ds.Constant(0)*s)
        self.assertEqual(s, s*1)
        self.assertEqual(s, 1*s)
        self.assertEqual(s, s*ds.Constant(1))
        self.assertEqual(s, ds.Constant(1)*s)

    def test_mul2(self):
        s = ds.Constant(1)
        for shift in range(0, 6):
            s *= (ds.Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*11])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)

    def test_mul3(self):
        s = ds.Delta() * 3
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, 3, [0]*5])
        self.assertNotIsInstance(s, ds.Delta)
        s = 3 * ds.Delta()
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, 3, [0]*5])
        self.assertNotIsInstance(s, ds.Delta)
        s = ds.Delta() * (2+2j)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, (2+2j), [0]*5])
        self.assertNotIsInstance(s, ds.Delta)
        s = (2+2j) * ds.Delta()
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, (2+2j), [0]*5])
        self.assertNotIsInstance(s, ds.Delta)
        s = ds.Delta() * 1
        self.assertIsInstance(s, ds.Delta)
        s = 1 * ds.Delta()
        self.assertIsInstance(s, ds.Delta)

    def test_div1(self):
        s = ds.Delta(1) / ds.Step()
        np.testing.assert_array_equal(s[-5:6], np.r_[[np.NaN]*5, 0, 1, [0]*4])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Step)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)

    def test_div2(self):
        s = ds.Constant(1)
        for shift in range(0, 6):
            s /= (ds.Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[np.Inf]*11])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)

    def test_div3(self):
        s = ds.Delta() / 3
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, 1/3, [0]*5])
        self.assertNotIsInstance(s, ds.Delta)
        s = 3 / ds.Delta()
        np.testing.assert_equal(s[-5:6], np.r_[[np.Inf]*5, 3, [np.Inf]*5])
        self.assertNotIsInstance(s, ds.Delta)
        s = ds.Delta() / (2+2j)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, 1.0/(2+2j), [0]*5])
        self.assertNotIsInstance(s, ds.Delta)
        s = (2+2j) / ds.Delta()
        infc = np.complex(np.Inf, np.Inf)
        np.testing.assert_equal(s[-5:6], np.r_[[infc]*5, (2+2j), [infc]*5])
        self.assertNotIsInstance(s, ds.Delta)
        s = ds.Delta() / 1
        self.assertIsInstance(s, ds.Delta)

    def test_neg1(self):
        s = -ds.Delta()
        np.testing.assert_array_equal(s[-5:6], np.r_[[0.]*5, -1, [0.]*5])
        self.assertNotIsInstance(s, ds.Delta)

    def test_neg2(self):
        s = ds.Constant(0)
        for shift in range(0, 6):
            s += -(ds.Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, [-1]*6])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)

    def test_neg3(self):
        s = -ds.Delta() + 3
        np.testing.assert_equal(s[-5:6], np.r_[[3]*5, 2, [3]*5])
        self.assertNotIsInstance(s, ds.Delta)
        self.assertNotIsInstance(s, ds.Constant)
        self.assertIsInstance(s, ds.DiscreteFunctionSignal)


if __name__ == "__main__":
    unittest.main()
