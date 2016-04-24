from skdsp.signal.discrete import Sinusoid, Constant, Sine, Cosine
import numpy as np
import sympy as sp
import unittest


class SinusoidTest(unittest.TestCase):

    def test_00as_euler(self):
        s = Sinusoid(1, sp.S.Pi/4, sp.S.Pi/3).as_euler()
        print(s)
        print(s.yexpr.as_real_imag(True)[0])

    def test_peak_amplitude(self):
        s = Sinusoid(3, sp.S.Pi/4, sp.S.Pi/12)
        self.assertEqual(3, s.peak_amplitude)
        s.peak_amplitude = 4
        self.assertEqual(4, s.peak_amplitude)
        s.peak_amplitude = 2+2j
        self.assertEqual(2+2j, s.peak_amplitude)

    def test_magnitude(self):
        s = Sinusoid(1, sp.S.Pi, 0).magnitude()
        self.assertNotIsInstance(s, Sinusoid)
        np.testing.assert_equal(np.r_[[1]*5], s[0:10:2])
        np.testing.assert_equal(np.r_[[1]*5], s[1:11:2])
        s = Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, Sinusoid)
        s = s.magnitude()
        np.testing.assert_equal(np.r_[[3]*4], s[-2:49:16])
        np.testing.assert_equal(np.r_[[3]*4], s[14:65:16])
        s = Sinusoid(1+1j, sp.S.Pi, 0).magnitude()
        np.testing.assert_almost_equal(np.r_[[np.sqrt(2)]*5], s[0:10:2])
        np.testing.assert_almost_equal(np.r_[[np.sqrt(2)]*5], s[1:11:2])
        s = Sinusoid(3+3j, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, Sinusoid)
        s = s.magnitude()
        np.testing.assert_almost_equal(np.r_[[np.sqrt(18)]*4], s[-2:49:16])
        np.testing.assert_almost_equal(np.r_[[np.sqrt(18)]*4], s[14:65:16])
        # en dBs
        s = Sinusoid(1, sp.S.Pi, 0).magnitude(dB=True)
        self.assertNotIsInstance(s, Sinusoid)
        np.testing.assert_equal(np.r_[[0]*5], s[0:10:2])
        np.testing.assert_equal(np.r_[[0]*5], s[1:11:2])
        s = Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, Sinusoid)
        s = s.magnitude(True)
        np.testing.assert_equal(20*np.log10(np.r_[[3]*4]), s[-2:49:16])
        np.testing.assert_equal(20*np.log10(np.r_[[3]*4]), s[14:65:16])
        s = Sinusoid(1+1j, sp.S.Pi, 0).magnitude(True)
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(2)]*5]),
                                       s[0:10:2])
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(2)]*5]),
                                       s[1:11:2])
        s = Sinusoid(3+3j, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, Sinusoid)
        s = s.magnitude(True)
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(18)]*4]),
                                       s[-2:49:16])
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(18)]*4]),
                                       s[14:65:16])

    def test_components(self):
        s = Sinusoid()
        s0 = s.in_phase()
        self.assertEqual(s0, Cosine())
        s1 = s.in_quadrature()
        self.assertEqual(s1, Constant(0))
        s = Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        s0 = s.in_phase()
        self.assertEqual(s0, Constant(3*sp.cos(sp.S.Pi/4)) *
                         Cosine(sp.S.Pi/8, 0))
        s1 = s.in_quadrature()
        self.assertEqual(s1, Constant(-3*sp.sin(sp.S.Pi/4)) *
                         Sine(sp.S.Pi/8, 0))

if __name__ == "__main__":
    unittest.main()
