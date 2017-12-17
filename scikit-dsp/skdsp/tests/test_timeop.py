import numpy as np
import skdsp.signal.discrete as ds
import unittest


class TimeOperatorsTest(unittest.TestCase):
    """ Test of time operators (shift, flip, ...). This test assume that the
    basic signals Delta() and Step() work correctly.
    """

    def test_shift_magic(self):
        ''' Shift tests using magic __lshift__ and __rshift__ '''
        # --- delta -----------------------------------------------------------
        expected_0 = np.r_[[0.]*5, 1., [0.]*5]
        r = np.arange(-5, 6)
        N = r.size
        s = ds.Delta()
        # shift to the right (negative and positive values)
        for k in r:
            expected = np.roll(expected_0, k)
            actual = (s >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            expected = np.roll(expected_0, -k)
            actual = (s << k).eval(r)
            np.testing.assert_equal(actual, expected)
        # --- step ------------------------------------------------------------
        s = ds.Step()
        # shift to the right (negative and positive values)
        for k in r:
            n0s = (k - r[0])
            expected = np.r_[[0.0]*n0s, [1.0]*(N-n0s)]
            actual = (s >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            n1s = (r[-1] + 1 + k)
            expected = np.r_[[0.0]*(N-n1s), [1.0]*n1s]
            actual = (s << k).eval(r)
            np.testing.assert_equal(actual, expected)

    def test_shift_method(self):
        ''' Shift tests using method s.shift() '''
        # --- delta -----------------------------------------------------------
        expected_0 = np.r_[[0.]*5, 1., [0.]*5]
        r = np.arange(-5, 6)
        N = r.size
        s = ds.Delta()
        # shift to the right (negative and positive values)
        for k in r:
            expected = np.roll(expected_0, k)
            actual = s.shift(k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            expected = np.roll(expected_0, -k)
            actual = s.shift(-k).eval(r)
            np.testing.assert_equal(actual, expected)
        # --- step ------------------------------------------------------------
        s = ds.Step()
        # shift to the right (negative and positive values)
        for k in r:
            n0s = (k - r[0])
            expected = np.r_[[0.0]*n0s, [1.0]*(N-n0s)]
            actual = s.shift(k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            n1s = (r[-1] + 1 + k)
            expected = np.r_[[0.0]*(N-n1s), [1.0]*n1s]
            actual = s.shift(-k).eval(r)
            np.testing.assert_equal(actual, expected)

    def test_shift_func(self):
        ''' Shift tests using method shift(s, k) '''
        # --- delta -----------------------------------------------------------
        expected_0 = np.r_[[0.]*5, 1., [0.]*5]
        r = np.arange(-5, 6)
        N = r.size
        s = ds.Delta()
        # shift to the right (negative and positive values)
        for k in r:
            expected = np.roll(expected_0, k)
            actual = s.shift(k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            expected = np.roll(expected_0, -k)
            actual = s.shift(-k).eval(r)
            np.testing.assert_equal(actual, expected)
        # --- step ------------------------------------------------------------
        s = ds.Step()
        # shift to the right (negative and positive values)
        for k in r:
            n0s = (k - r[0])
            expected = np.r_[[0.0]*n0s, [1.0]*(N-n0s)]
            actual = s.shift(k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            n1s = (r[-1] + 1 + k)
            expected = np.r_[[0.0]*(N-n1s), [1.0]*n1s]
            actual = s.shift(-k).eval(r)
            np.testing.assert_equal(actual, expected)

    def test_shift_inplace(self):
        ''' Shift tests using magic __ilshift__ and __irshift__ '''
        # --- delta -----------------------------------------------------------
        expected_0 = np.r_[[0.]*5, 1., [0.]*5]
        r = np.arange(-5, 6)
        N = r.size
        # shift to the right (negative and positive values)
        for k in r:
            expected = np.roll(expected_0, k)
            s = ds.Delta()
            s >>= k
            actual = s.eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            expected = np.roll(expected_0, -k)
            s = ds.Delta()
            s <<= k
            actual = s.eval(r)
            np.testing.assert_equal(actual, expected)
        # --- step ------------------------------------------------------------
        # shift to the right (negative and positive values)
        for k in r:
            n0s = (k - r[0])
            expected = np.r_[[0.0]*n0s, [1.0]*(N-n0s)]
            s = ds.Step()
            s >>= k
            actual = s.eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        r = np.arange(-5, 6)
        for k in r:
            n1s = (r[-1] + 1 + k)
            expected = np.r_[[0.0]*(N-n1s), [1.0]*n1s]
            s = ds.Step()
            s <<= k
            actual = s.eval(r)
            np.testing.assert_equal(actual, expected)

    def test_flip_magic(self):
        ''' Flip tests using magic __reverse__'''
        # --- delta -----------------------------------------------------------
        expected_0 = np.r_[[0.]*5, 1., [0.]*5]
        r = np.arange(-5, 6)
        N = r.size
        s = ds.Delta()
        # shift to the right and flip
        for k in r:
            expected = np.roll(expected_0, k)[::-1]
            actual = reversed(s >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left and flip
        for k in r:
            expected = np.roll(expected_0, -k)[::-1]
            actual = reversed(s << k).eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the right
        for k in r:
            expected = np.roll(expected_0[::-1], k)
            actual = (reversed(s) >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the left
        for k in r:
            expected = np.roll(expected_0[::-1], -k)
            actual = (reversed(s) << k).eval(r)
            np.testing.assert_equal(actual, expected)
        # --- step ------------------------------------------------------------
        s = ds.Step()
        # shift to the right and flip
        for k in r:
            n0s = (k - r[0])
            expected = np.r_[[0.0]*n0s, [1.0]*(N-n0s)][::-1]
            actual = reversed(s >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            n1s = (r[-1] + 1 + k)
            expected = np.r_[[0.0]*(N-n1s), [1.0]*n1s][::-1]
            actual = reversed(s << k).eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the right
        for k in r:
            n1s = (k - r[0] + 1)
            expected = np.r_[[1.0]*n1s, [0.0]*(N-n1s)]
            actual = (reversed(s) >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            n0s = (r[-1] + k)
            expected = np.r_[[1.0]*(N-n0s), [0.0]*n0s]
            actual = (reversed(s) << k).eval(r)
            np.testing.assert_equal(actual, expected)

    def test_flip_method(self):
        ''' Flip tests using method s.flip() '''
        # --- delta -----------------------------------------------------------
        expected_0 = np.r_[[0.]*5, 1., [0.]*5]
        r = np.arange(-5, 6)
        N = r.size
        s = ds.Delta()
        # shift to the right and flip
        for k in r:
            expected = np.roll(expected_0, k)[::-1]
            actual = (s >> k).flip().eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left and flip
        for k in r:
            expected = np.roll(expected_0, -k)[::-1]
            actual = (s << k).flip().eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the right
        for k in r:
            expected = np.roll(expected_0[::-1], k)
            actual = (s.flip() >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the left
        for k in r:
            expected = np.roll(expected_0[::-1], -k)
            actual = (s.flip() << k).eval(r)
            np.testing.assert_equal(actual, expected)
        # --- step ------------------------------------------------------------
        s = ds.Step()
        # shift to the right and flip
        for k in r:
            n0s = (k - r[0])
            expected = np.r_[[0.0]*n0s, [1.0]*(N-n0s)][::-1]
            actual = (s >> k).flip().eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            n1s = (r[-1] + 1 + k)
            expected = np.r_[[0.0]*(N-n1s), [1.0]*n1s][::-1]
            actual = (s << k).flip().eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the right
        for k in r:
            n1s = (k - r[0] + 1)
            expected = np.r_[[1.0]*n1s, [0.0]*(N-n1s)]
            actual = (s.flip() >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            n0s = (r[-1] + k)
            expected = np.r_[[1.0]*(N-n0s), [0.0]*n0s]
            actual = (s.flip() << k).eval(r)
            np.testing.assert_equal(actual, expected)

    def test_flip_func(self):
        ''' Shift tests using function sg.flip(s) '''
        # --- delta -----------------------------------------------------------
        expected_0 = np.r_[[0.]*5, 1., [0.]*5]
        r = np.arange(-5, 6)
        N = r.size
        s = ds.Delta()
        # shift to the right and flip
        for k in r:
            expected = np.roll(expected_0, k)[::-1]
            s = s >> k
            s = s.flip()
            actual = s.eval(r)
            # actual = (s >> k).flip().eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left and flip
        for k in r:
            expected = np.roll(expected_0, -k)[::-1]
            actual = (s << k).flip().eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the right
        for k in r:
            expected = np.roll(expected_0[::-1], k)
            actual = (s.flip() >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the left
        for k in r:
            expected = np.roll(expected_0[::-1], -k)
            actual = (s.flip() << k).eval(r)
            np.testing.assert_equal(actual, expected)
        # --- step ------------------------------------------------------------
        s = ds.Step()
        # shift to the right and flip
        for k in r:
            n0s = (k - r[0])
            expected = np.r_[[0.0]*n0s, [1.0]*(N-n0s)][::-1]
            actual = (s >> k).flip().eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left and flip
        for k in r:
            n1s = (r[-1] + 1 + k)
            expected = np.r_[[0.0]*(N-n1s), [1.0]*n1s][::-1]
            actual = (s << k).flip().eval(r)
            np.testing.assert_equal(actual, expected)
        # flip and shift to the right
        for k in r:
            n1s = (k - r[0] + 1)
            expected = np.r_[[1.0]*n1s, [0.0]*(N-n1s)]
            actual = (s.flip() >> k).eval(r)
            np.testing.assert_equal(actual, expected)
        # shift to the left (negative and positive values)
        for k in r:
            n0s = (r[-1] + k)
            expected = np.r_[[1.0]*(N-n0s), [0.0]*n0s]
            actual = (s.flip() << k).eval(r)
            np.testing.assert_equal(actual, expected)

    def test_flip_flip(self):
        d = ds.Delta() >> 1
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d.flip()[1], 0.0)
        self.assertEqual(d.flip().flip()[1], 1.0)
        self.assertEqual(d.flip().flip().flip()[1], 0.0)

    def test_delay_constructors(self):
        # --- delta -----------------------------------------------------------
        d1 = ds.Delta() >> 1
        d2 = ds.Delta(1)
        self.assertEqual(d1, d2)
        d1 = ds.Delta() << 1
        d2 = ds.Delta(-1)
        self.assertEqual(d1, d2)
        with self.assertRaises(ValueError):
            ds.Delta(0.5)
        # --- step ------------------------------------------------------------
        s1 = ds.Step() >> 1
        s2 = ds.Step(1)
        self.assertEqual(s1, s2)
        s1 = ds.Step() << 1
        s2 = ds.Step(-1)
        self.assertEqual(s1, s2)
        with self.assertRaises(ValueError):
            ds.Step(0.5)

    def test_scale(self):
        # TODO
        # El escalado menor que 1 funciona con las limitaciones del
        # punto flotante (1/3, 1/6 y 1/7, no existen !!)
        pass


if __name__ == "__main__":
    unittest.main()
