import unittest

import numpy as np
from skdsp.signal.discrete import Delta, Step, Constant, DiscreteFunctionSignal


class ArithmeticTest(unittest.TestCase):

    def test_add1(self):
        s = Delta(-1) + Step()
        np.testing.assert_array_equal(s[-5:6], np.r_[[0]*4, [1]*7])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Step)
        self.assertIsInstance(s, DiscreteFunctionSignal)

    def test_add2(self):
        s = Constant(0)
        for shift in range(0, 6):
            s += (Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, [1]*6])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)

    def test_add3(self):
        s = Delta() + 3
        np.testing.assert_equal(s[-5:6], np.r_[[3]*5, 4, [3]*5])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)
        s = Delta() + 0
        self.assertIsInstance(s, Delta)
        self.assertEqual(s, Delta())
        s = 0 + Delta()
        self.assertIsInstance(s, Delta)
        self.assertEqual(s, Delta())
        s = Delta() + Constant(0)
        self.assertIsInstance(s, Delta)
        self.assertEqual(s, Delta())
        s = Constant(0) + Delta()
        self.assertIsInstance(s, Delta)
        s = Delta()
        self.assertEqual(s, Delta())
        self.assertEqual(s, s + 0)
        self.assertEqual(s, 0 + s)
        self.assertEqual(s, s + Constant(0))
        self.assertEqual(s, Constant(0) + s)

    def test_sub1(self):
        s = Delta(-1) - Step()
        np.testing.assert_array_equal(s[-5:6], np.r_[[0]*4, 1, [-1]*6])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Step)
        self.assertIsInstance(s, DiscreteFunctionSignal)

    def test_sub2(self):
        s = Constant(0)
        for shift in range(0, 6):
            s -= (Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, [-1]*6])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)

    def test_sub3(self):
        s = Delta() - 3
        np.testing.assert_equal(s[-5:6], np.r_[[-3]*5, -2, [-3]*5])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)
        s = 3 - Delta()
        np.testing.assert_equal(s[-5:6], np.r_[[3]*5, 2, [3]*5])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)
        s = Delta() - 0
        self.assertIsInstance(s, Delta)
        s = 0 - Delta()
        self.assertIsInstance(s, Delta)
        s = Delta() - Constant(0)
        self.assertIsInstance(s, Delta)
        s = Constant(0) - Delta()
        self.assertIsInstance(s, Delta)
        s = Delta()
        self.assertEqual(s, Delta())
        self.assertEqual(s, s - 0)
        self.assertEqual(-s, 0 - s)
        self.assertEqual(s, s - Constant(0))
        self.assertEqual(-s, Constant(0) - s)

    def test_mul1(self):
        s = Delta(1) * Step()
        np.testing.assert_array_equal(s[-5:6], np.r_[[0]*6, 1, [0]*4])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Step)
        self.assertIsInstance(s, DiscreteFunctionSignal)
        s = Delta()
        self.assertEqual(Constant(0), s*0)
        self.assertEqual(Constant(0), 0*s)
        self.assertEqual(Constant(0), s*Constant(0))
        self.assertEqual(Constant(0), Constant(0)*s)
        self.assertEqual(s, s*1)
        self.assertEqual(s, 1*s)
        self.assertEqual(s, s*Constant(1))
        self.assertEqual(s, Constant(1)*s)

    def test_mul2(self):
        s = Constant(1)
        for shift in range(0, 6):
            s *= (Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*11])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)

    def test_mul3(self):
        s = Delta() * 3
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, 3, [0]*5])
        self.assertIsInstance(s, Delta)
        s = 3 * Delta()
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, 3, [0]*5])
        self.assertIsInstance(s, Delta)
        s = Delta() * (2+2j)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, (2+2j), [0]*5])
        self.assertIsInstance(s, Delta)
        s = (2+2j) * Delta()
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, (2+2j), [0]*5])
        self.assertIsInstance(s, Delta)
        s = Delta() * 1
        self.assertIsInstance(s, Delta)
        s = 1 * Delta()
        self.assertIsInstance(s, Delta)

    def test_div1(self):
        s = Delta(1) / Step()
        np.testing.assert_array_equal(s[-5:6], np.r_[[np.NaN]*5, 0, 1, [0]*4])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Step)
        self.assertIsInstance(s, DiscreteFunctionSignal)

    def test_div2(self):
        s = Constant(1)
        for shift in range(0, 6):
            s /= (Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[np.Inf]*11])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)

    def test_div3(self):
        s = Delta() / 3
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, 1/3, [0]*5])
        self.assertIsInstance(s, Delta)
        s = 3 / Delta()
        np.testing.assert_equal(s[-5:6], np.r_[[np.Inf]*5, 3, [np.Inf]*5])
        self.assertNotIsInstance(s, Delta)
        s = Delta() / (2+2j)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, 1.0/(2+2j), [0]*5])
        self.assertIsInstance(s, Delta)
        s = (2+2j) / Delta()
        infc = np.complex(np.Inf, np.Inf)
        np.testing.assert_equal(s[-5:6], np.r_[[infc]*5, (2+2j), [infc]*5])
        self.assertNotIsInstance(s, Delta)
        s = Delta() / 1
        self.assertIsInstance(s, Delta)

    def test_neg1(self):
        s = -Delta()
        np.testing.assert_array_equal(s[-5:6], np.r_[[0.]*5, -1, [0.]*5])
        self.assertIsInstance(s, Delta)

    def test_neg2(self):
        s = Constant(0)
        for shift in range(0, 6):
            s += -(Delta() >> shift)
        np.testing.assert_equal(s[-5:6], np.r_[[0]*5, [-1]*6])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)

    def test_neg3(self):
        s = -Delta() + 3
        np.testing.assert_equal(s[-5:6], np.r_[[3]*5, 2, [3]*5])
        self.assertNotIsInstance(s, Delta)
        self.assertNotIsInstance(s, Constant)
        self.assertIsInstance(s, DiscreteFunctionSignal)


if __name__ == "__main__":
    unittest.main()
