import skdsp.signal.discrete as ds
import skdsp.signal.continuous as cs
import skdsp.signal.signal as sg
from skdsp.signal.util import is_discrete, is_continuous, is_real, is_complex
import numpy as np
import sympy as sp
import unittest


class RampTest(unittest.TestCase):

    def test_00(self):
        import skdsp.signal.printer as pt
        d = ds.Ramp().delay(3)
        print(pt.latex(d, mode='inline'))
        d = (ds.Ramp()*ds.Step()).delay(3)
        print(pt.latex(d, mode='inline'))
        print(d.eval(2))
        d = cs.Ramp().delay(1.5)
        print(pt.latex(d, mode='inline'))
        d = (cs.Ramp()*cs.Step()).delay(1.5)
        print(pt.latex(d, mode='inline'))
        print(d.eval(1.3456))
        d = cs.Ramp().shift(1.3).flip()
        print(pt.latex(d, mode='inline'))
        d = (cs.Ramp()*cs.Step()).flip().shift(1.3)
        print(pt.latex(d, mode='inline'))
        print(d.eval(1.3456))

    def test_constructor(self):
        ''' Ramp (discrete/continuous): constructors '''
        # rampa discreta
        d = ds.Ramp()
        self.assertIsInstance(d, sg.Signal)
        self.assertIsInstance(d, sg.FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteMixin)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        self.assertTrue(is_discrete(d))
        self.assertFalse(is_continuous(d))
        # rampa continua
        d = cs.Ramp()
        self.assertIsInstance(d, sg.Signal)
        self.assertIsInstance(d, sg.FunctionSignal)
        self.assertIsInstance(d, cs.ContinuousMixin)
        self.assertIsInstance(d, cs.ContinuousFunctionSignal)
        self.assertFalse(is_discrete(d))
        self.assertTrue(is_continuous(d))

    def test_eval_sample(self):
        ''' Ramp (discrete/continuous): aval(scalar) '''
        # rampa discreta
        d = ds.Ramp()
        self.assertEqual(d.eval(0), 0.0)
        self.assertEqual(d.eval(1), 1.0)
        self.assertEqual(d.eval(2), 2.0)
        self.assertEqual(d.eval(-1), -1.0)
        self.assertEqual(d.eval(-2), -2.0)
        with self.assertRaises(TypeError):
            d.eval(0.5)
        # rampa continua
        d = cs.Ramp()
        self.assertEqual(d.eval(0), 0.0)
        self.assertEqual(d.eval(1), 1.0)
        self.assertEqual(d.eval(2), 2.0)
        self.assertEqual(d.eval(-1), -1.0)
        self.assertEqual(d.eval(-2), -2.0)

    def test_eval_range(self):
        ''' Ramp (discrete/continuous): aval(array) '''
        # rampa discreta
        d = ds.Ramp()
        np.testing.assert_array_equal(d.eval(np.arange(0, 3)),
                                      np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-1, 3)),
                                      np.array([-1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-4, 3, 2)),
                                      np.array([-4.0, -2.0, 0.0, 2.0]))
        np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
                                      np.array([3.0, 1.0, -1.0]))
        with self.assertRaises(TypeError):
            d.eval(np.arange(0, 2, 0.5))
        # rampa continua
        d = cs.Ramp()
        np.testing.assert_array_equal(d.eval(np.arange(0, 3)),
                                      np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-1, 3)),
                                      np.array([-1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(d.eval(np.arange(-4, 3, 2)),
                                      np.array([-4.0, -2.0, 0.0, 2.0]))
        np.testing.assert_array_equal(d.eval(np.arange(3, -2, -2)),
                                      np.array([3.0, 1.0, -1.0]))

    def test_getitem_scalar(self):
        ''' Ramp (discrete/continuous): aval[scalar] '''
        # rampa discreta
        d = ds.Ramp()
        self.assertEqual(d[0], 0.0)
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d[2], 2.0)
        self.assertEqual(d[-1], -1.0)
        self.assertEqual(d[-2], -2.0)
        with self.assertRaises(TypeError):
            d[0.5]
        # rampa continua
        d = cs.Ramp()
        self.assertEqual(d[0], 0.0)
        self.assertEqual(d[1], 1.0)
        self.assertEqual(d[2], 2.0)
        self.assertEqual(d[-1], -1.0)
        self.assertEqual(d[-2], -2.0)

    def test_getitem_slice(self):
        ''' Ramp (discrete/continuous): aval[array] '''
        # rampa discreta
        d = ds.Ramp()
        np.testing.assert_array_equal(d[0:3], np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(d[-1:3], np.array([-1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(d[-4:2:2], np.array([-4.0, -2.0, 0.0]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([3.0, 1.0, -1.0]))
        with self.assertRaises(TypeError):
            d[0:2:0.5]
        # rampa continua
        d = cs.Ramp()
        np.testing.assert_array_equal(d[0:3], np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(d[-1:3], np.array([-1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(d[-4:2:2], np.array([-4.0, -2.0, 0.0]))
        np.testing.assert_array_equal(d[3:-2:-2], np.array([3.0, 1.0, -1.0]))

    def test_dtype(self):
        ''' Ramp (discrete/continuous): dtype '''
        # rampa discreta
        d = ds.Ramp()
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_
        # rampa continua
        d = cs.Ramp()
        self.assertTrue(is_real(d))
        self.assertEqual(d.dtype, np.float_)
        d.dtype = np.complex_
        self.assertTrue(is_complex(d))
        self.assertEqual(d.dtype, np.complex_)
        with self.assertRaises(ValueError):
            d.dtype = np.int_

    def test_name(self):
        ''' Ramp (discrete/continuous): name '''
        # rampa discreta
        d = ds.Ramp()
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        # rampa continua
        d = cs.Ramp()
        self.assertEqual(d.name, 'x')
        d.name = 'z'
        self.assertEqual(d.name, 'z')

    def test_xvar(self):
        ''' Ramp (discrete/continuous): free variable '''
        # rampa discreta
        d = ds.Ramp() >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'r[n - 3]')
        d.xvar = sp.symbols('m', integer=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'r[m - 3]')
        # rampa continua
        d = cs.Ramp() >> 3
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'r(t - 3)')
        d.xvar = sp.symbols('u', real=True)
        self.assertEqual(d.name, 'x')
        self.assertEqual(str(d), 'r(u - 3)')

if __name__ == "__main__":
    unittest.main()
