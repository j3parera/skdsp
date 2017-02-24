import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.printer as pt
import numpy as np
import sympy as sp
import unittest


class SinusoidTest(unittest.TestCase):

    def test_as_euler(self):
        ''' Sinusoid (discrete/continuous): as_euler '''
        # sinusoide discreta
        s = ds.Sinusoid(0.5, sp.S.Pi/4, sp.S.Pi/3)
        self.assertEqual('0.25*exp(j*(-pi*n/4 - pi/3))' +
                         ' + 0.25*exp(j*(pi*n/4 + pi/3))', s.as_euler())
        # sinusoide continua
        s = cs.Sinusoid(0.5, sp.S.Pi/4, sp.S.Pi/3)
        self.assertEqual('0.25*exp(j*(-pi*t/4 - pi/3))' +
                         ' + 0.25*exp(j*(pi*t/4 + pi/3))', s.as_euler())

    def test_peak_amplitude(self):
        ''' Sinusoid (discrete/continuous): peak_amplitude '''
        # sinusoide discreta
        s = ds.Sinusoid(3, sp.S.Pi/4, sp.S.Pi/12)
        self.assertEqual(3, s.peak_amplitude)
        s.peak_amplitude = 4
        self.assertEqual(4, s.peak_amplitude)
        s.peak_amplitude = 2+2j
        self.assertEqual(2+2j, s.peak_amplitude)
        # sinusoide continua
        s = cs.Sinusoid(3, sp.S.Pi/4, sp.S.Pi/12)
        self.assertEqual(3, s.peak_amplitude)
        s.peak_amplitude = 4
        self.assertEqual(4, s.peak_amplitude)
        s.peak_amplitude = 2+2j
        self.assertEqual(2+2j, s.peak_amplitude)

    def test_magnitude(self):
        ''' Sinusoid (discrete/continuous): magnitude '''
        # sinusoide discreta
        s = ds.Sinusoid(1, sp.S.Pi, 0).magnitude()
        self.assertNotIsInstance(s, ds.Sinusoid)
        np.testing.assert_equal(np.r_[[1]*5], s[0:10:2])
        np.testing.assert_equal(np.r_[[1]*5], s[1:11:2])
        s = ds.Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, ds.Sinusoid)
        s = s.magnitude()
        np.testing.assert_equal(np.r_[[3]*4], s[-2:49:16])
        np.testing.assert_equal(np.r_[[3]*4], s[14:65:16])
        s = ds.Sinusoid(1+1j, sp.S.Pi, 0).magnitude()
        np.testing.assert_almost_equal(np.r_[[np.sqrt(2)]*5], s[0:10:2])
        np.testing.assert_almost_equal(np.r_[[np.sqrt(2)]*5], s[1:11:2])
        s = ds.Sinusoid(3+3j, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, ds.Sinusoid)
        s = s.magnitude()
        np.testing.assert_almost_equal(np.r_[[np.sqrt(18)]*4], s[-2:49:16])
        np.testing.assert_almost_equal(np.r_[[np.sqrt(18)]*4], s[14:65:16])
        # en dBs
        s = ds.Sinusoid(1, sp.S.Pi, 0).magnitude(dB=True)
        self.assertNotIsInstance(s, ds.Sinusoid)
        np.testing.assert_equal(np.r_[[0]*5], s[0:10:2])
        np.testing.assert_equal(np.r_[[0]*5], s[1:11:2])
        s = ds.Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, ds.Sinusoid)
        s = s.magnitude(True)
        np.testing.assert_equal(20*np.log10(np.r_[[3]*4]), s[-2:49:16])
        np.testing.assert_equal(20*np.log10(np.r_[[3]*4]), s[14:65:16])
        s = ds.Sinusoid(1+1j, sp.S.Pi, 0).magnitude(True)
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(2)]*5]),
                                       s[0:10:2])
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(2)]*5]),
                                       s[1:11:2])
        s = ds.Sinusoid(3+3j, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, ds.Sinusoid)
        s = s.magnitude(True)
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(18)]*4]),
                                       s[-2:49:16])
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(18)]*4]),
                                       s[14:65:16])
        # sinusoide continua
        s = cs.Sinusoid(1, sp.S.Pi, 0).magnitude()
        self.assertNotIsInstance(s, ds.Sinusoid)
        np.testing.assert_equal(np.r_[[1]*5], s[0:10:2])
        np.testing.assert_equal(np.r_[[1]*5], s[1:11:2])
        s = cs.Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, cs.Sinusoid)
        s = s.magnitude()
        np.testing.assert_equal(np.r_[[3]*4], s[-2:49:16])
        np.testing.assert_equal(np.r_[[3]*4], s[14:65:16])
        s = cs.Sinusoid(1+1j, sp.S.Pi, 0).magnitude()
        np.testing.assert_almost_equal(np.r_[[np.sqrt(2)]*5], s[0:10:2])
        np.testing.assert_almost_equal(np.r_[[np.sqrt(2)]*5], s[1:11:2])
        s = cs.Sinusoid(3+3j, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, cs.Sinusoid)
        s = s.magnitude()
        np.testing.assert_almost_equal(np.r_[[np.sqrt(18)]*4], s[-2:49:16])
        np.testing.assert_almost_equal(np.r_[[np.sqrt(18)]*4], s[14:65:16])
        # en dBs
        s = cs.Sinusoid(1, sp.S.Pi, 0).magnitude(dB=True)
        self.assertNotIsInstance(s, ds.Sinusoid)
        np.testing.assert_equal(np.r_[[0]*5], s[0:10:2])
        np.testing.assert_equal(np.r_[[0]*5], s[1:11:2])
        s = cs.Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, cs.Sinusoid)
        s = s.magnitude(True)
        np.testing.assert_equal(20*np.log10(np.r_[[3]*4]), s[-2:49:16])
        np.testing.assert_equal(20*np.log10(np.r_[[3]*4]), s[14:65:16])
        s = cs.Sinusoid(1+1j, sp.S.Pi, 0).magnitude(True)
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(2)]*5]),
                                       s[0:10:2])
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(2)]*5]),
                                       s[1:11:2])
        s = cs.Sinusoid(3+3j, sp.S.Pi/8, sp.S.Pi/4)
        self.assertIsInstance(s, cs.Sinusoid)
        s = s.magnitude(True)
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(18)]*4]),
                                       s[-2:49:16])
        np.testing.assert_almost_equal(20*np.log10(np.r_[[np.sqrt(18)]*4]),
                                       s[14:65:16])

    def test_components(self):
        ''' Sinusoid (discrete/continuous): components '''
        # sinusoide discreta
        s = ds.Sinusoid()
        s0 = s.in_phase
        self.assertEqual(s0, ds.Cosine())
        s0 = s.I
        self.assertEqual(s0, ds.Cosine())
        s1 = s.in_quadrature
        self.assertEqual(s1, ds.Constant(0))
        s1 = s.Q
        self.assertEqual(s1, ds.Constant(0))
        s = ds.Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        s0 = s.in_phase
        self.assertEqual(s0, ds.Constant(3*sp.cos(sp.S.Pi/4)) *
                         ds.Cosine(sp.S.Pi/8, 0))
        s0 = s.I
        self.assertEqual(s0, ds.Constant(3*sp.cos(sp.S.Pi/4)) *
                         ds.Cosine(sp.S.Pi/8, 0))
        s1 = s.in_quadrature
        self.assertEqual(s1, ds.Constant(-3*sp.sin(sp.S.Pi/4)) *
                         ds.Sine(sp.S.Pi/8, 0))
        s1 = s.Q
        self.assertEqual(s1, ds.Constant(-3*sp.sin(sp.S.Pi/4)) *
                         ds.Sine(sp.S.Pi/8, 0))
        # sinusoide continua
        s = cs.Sinusoid()
        s0 = s.in_phase
        self.assertEqual(s0, cs.Cosine())
        s0 = s.I
        self.assertEqual(s0, cs.Cosine())
        s1 = s.in_quadrature
        self.assertEqual(s1, cs.Constant(0))
        s1 = s.Q
        self.assertEqual(s1, cs.Constant(0))
        s = cs.Sinusoid(3, sp.S.Pi/8, sp.S.Pi/4)
        s0 = s.in_phase
        self.assertEqual(s0, cs.Constant(3*sp.cos(sp.S.Pi/4)) *
                         cs.Cosine(sp.S.Pi/8, 0))
        s0 = s.I
        self.assertEqual(s0, cs.Constant(3*sp.cos(sp.S.Pi/4)) *
                         cs.Cosine(sp.S.Pi/8, 0))
        s1 = s.in_quadrature
        self.assertEqual(s1, cs.Constant(-3*sp.sin(sp.S.Pi/4)) *
                         cs.Sine(sp.S.Pi/8, 0))
        s1 = s.Q
        self.assertEqual(s1, cs.Constant(-3*sp.sin(sp.S.Pi/4)) *
                         cs.Sine(sp.S.Pi/8, 0))

    def test_latex(self):
        s = ds.Sinusoid(3, sp.S.Pi/4, sp.S.Pi/12)
        print(pt.latex(s))

if __name__ == "__main__":
    unittest.main()
