from skdsp.signal._signal import _Signal, _FunctionSignal
from skdsp.signal.printer import latex
import numpy as np
import skdsp.signal.discrete as ds
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
        ''' Square: constructors.
        '''
        # square discreta
        d = ds.Square()
        self.assertIsNotNone(d)
        # square discreta
        d = ds.Square(ds.n)
        self.assertIsNotNone(d)
        # square discreta
        d = ds.Square(ds.n-3)
        # jerarquía
        self.assertIsInstance(d, _Signal)
        self.assertIsInstance(d, _FunctionSignal)
        self.assertIsInstance(d, ds.DiscreteFunctionSignal)
        # retardo simbólico
        d = ds.Square(sp.Symbol('r', integer=True))
        self.assertIsNotNone(d)
        d = ds.Square(ds.m)
        self.assertIsNotNone(d)
        d = ds.Square(ds.n-ds.m)
        self.assertIsNotNone(d)
        # retardo no entero
        with self.assertRaises(ValueError):
            ds.Square(sp.Symbol('z', real=True))
        with self.assertRaises(ValueError):
            ds.Square(ds.n-1.5)
        with self.assertRaises(ValueError):
            ds.Square(ds.n/3)

    def test_name(self):
        ''' Square: name.
        '''
        # square discreta
        d = ds.Square(ds.n-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        d.name = 'z'
        self.assertEqual(d.name, 'z')
        self.assertEqual(d.latex_name, 'z')
        with self.assertRaises(ValueError):
            d.name = 'x0'
        with self.assertRaises(ValueError):
            d.name = 'y0'
        d = ds.Square(ds.n-3, name='y0')
        self.assertEqual(d.name, 'y0')
        self.assertEqual(d.latex_name, 'y_{0}')
        del d
        d = ds.Square(ds.n-3, name='yupi')
        self.assertEqual(d.name, 'yupi')
        self.assertEqual(d.latex_name, 'yupi')

    def test_xvar_xexpr(self):
        ''' Square: independent variable and expression.
        '''
        # square discreta
        d = ds.Square()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar)
        # shift
        shift = 5
        d = ds.Square().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, d.xvar - shift)
        d = ds.Square(ds.n-shift)
        self.assertEqual(d.xexpr, d.xvar - shift)
        # flip
        d = ds.Square().flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar)
        # shift and flip
        shift = 5
        d = ds.Square().shift(shift).flip()
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar - shift)
        d = ds.Square(ds.n-shift).flip()
        self.assertEqual(d.xexpr, -d.xvar - shift)
        # flip and shift
        shift = 5
        d = ds.Square().flip().shift(shift)
        self.assertTrue(d.is_discrete)
        self.assertFalse(d.is_continuous)
        self.assertEqual(d.xvar, d.default_xvar())
        self.assertEqual(d.xexpr, -d.xvar + shift)

    def test_yexpr_real_imag(self):
        ''' Square: function expression.
        '''
        # square discreta
        d = ds.Square(sp.Symbol('k', integer=True)-3)
        # expresión
        nm = sp.Mod(d.xexpr, d.period)
        self.assertEqual(d.yexpr, sp.Piecewise((1, nm < d.width),
                                               (-1, nm < d.period)))
        self.assertTrue(np.issubdtype(d.dtype, np.float))
        self.assertTrue(d.is_integer)
        self.assertTrue(d.is_real)
        self.assertTrue(d.is_complex)
        self.assertEqual(d, d.real)
        self.assertEqual(ds.Constant(0), d.imag)

    def test_period(self):
        ''' Square: period.
        '''
        # square discreta
        d = ds.Square(ds.n-3)
        # periodicidad
        self.assertTrue(d.is_periodic)
        self.assertEqual(d.period, 16)
        # square discreta
        d = ds.Square(ds.n, 27)
        # periodicidad
        self.assertTrue(d.is_periodic)
        self.assertEqual(d.period, 27)

    def test_repr_str_latex(self):
        ''' Square: repr, str and latex.
        '''
        # square discreta
        d = ds.Square(ds.n)
        # repr
        self.assertEqual(repr(d), 'Square(n, 16)')
        # str
        self.assertEqual(str(d), 'square[((n))16/8]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\mathrm{Square}_{8}\left[((n))_{16}\right]$')
        # square discreta
        d = ds.Square(ds.n, sp.Symbol('L', integer=True, positive=True))
        # repr
        self.assertEqual(repr(d), 'Square(n, L)')
        # str
        self.assertEqual(str(d), 'square[((n))L/8]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\mathrm{Square}_{8}\left[((n))_{L}\right]$')
        # square discreta
        d = ds.Square(ds.n+5, 25)
        # repr
        self.assertEqual(repr(d), 'Square(n + 5, 25)')
        # str
        self.assertEqual(str(d), 'square[((n + 5))25/8]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\mathrm{Square}_{8}\left[((n + 5))_{25}\right]$')

        # square discreta
        d = ds.Square(ds.n-5, 25)
        # repr
        self.assertEqual(repr(d), 'Square(n - 5, 25)')
        # str
        self.assertEqual(str(d), 'square[((n - 5))25/8]')
        # latex
        self.assertEqual(latex(d, mode='inline'),
                         r'$\mathrm{Square}_{8}\left[((n - 5))_{25}\right]$')

    def test_eval_sample(self):
        ''' Square: eval(scalar), eval(range)
        '''
        # scalar
        d = ds.Square()
        self.assertAlmostEqual(d.eval(0), 1)
        self.assertAlmostEqual(d.eval(1), 1)
        self.assertAlmostEqual(d.eval(-1), -1)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([-1, 1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([-1, -1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([1, 1, -1]))
        # scalar
        d = ds.Square(ds.n-1)
        self.assertAlmostEqual(d.eval(0), -1)
        self.assertAlmostEqual(d.eval(1), 1)
        self.assertAlmostEqual(d.eval(-1), -1)
        with self.assertRaises(ValueError):
            d.eval(0.5)
        # range
        np.testing.assert_array_almost_equal(d.eval(np.arange(0, 2)),
                                             np.array([-1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-1, 2)),
                                             np.array([-1, -1, 1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(-4, 1, 2)),
                                             np.array([-1, -1, -1]))
        np.testing.assert_array_almost_equal(d.eval(np.arange(3, -2, -2)),
                                             np.array([1, 1, -1]))
        with self.assertRaises(ValueError):
            d.eval(np.arange(0, 2, 0.5))

    def test_getitem_scalar(self):
        ''' Square (discrete): eval[scalar], eval[slice] '''
        # scalar
        d = ds.Square()
        self.assertAlmostEqual(d[0], 1)
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[-1], -1)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([-1, 1, 1]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([-1, -1, 1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([1, 1, -1]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]
        # scalar
        d = ds.Square(ds.n+1)
        self.assertAlmostEqual(d[0], 1)
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[-1], 1)
        with self.assertRaises(ValueError):
            d[0.5]
        # slice
        np.testing.assert_array_almost_equal(d[0:2],
                                             np.array([1, 1]))
        np.testing.assert_array_almost_equal(d[-1:2],
                                             np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(d[-4:1:2],
                                             np.array([-1, -1, 1]))
        np.testing.assert_array_almost_equal(d[3:-2:-2],
                                             np.array([1, 1, 1]))
        with self.assertRaises(ValueError):
            d[0:2:0.5]

    def test_generator(self):
        ''' Square (discrete): generate '''
        d = ds.Square()
        with self.assertRaises(ValueError):
            d.generate(0, step=0.1)
        dg = d.generate(s0=-3, size=5, overlap=4)
        np.testing.assert_array_equal(next(dg), np.array([-1, -1, -1, 1, 1]))
        np.testing.assert_array_equal(next(dg), np.array([-1, -1, 1, 1, 1]))
        np.testing.assert_array_equal(next(dg), np.array([-1, 1, 1, 1, 1]))

    def test_flip(self):
        ''' Square (discrete): flip '''
        d = ds.Square()
        np.testing.assert_array_equal(d.flip()[-3:3],
                                      np.array([1, 1, 1, 1, -1, -1]))
        d = ds.Square(ds.n-1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([1, 1, 1, -1, -1, -1]))
        d = ds.Square(ds.n+1).flip()
        np.testing.assert_array_equal(d[-3:3], np.array([1, 1, 1, 1, 1, -1]))

    def test_shift_delay(self):
        ''' Square (discrete): shift, delay '''
        d = ds.Square()
        with self.assertRaises(ValueError):
            d.shift(0.5)
        d = ds.Square().shift(2)
        np.testing.assert_array_equal(d[-3:3],
                                      np.array([-1, -1, -1, -1, -1, 1]))
        with self.assertRaises(ValueError):
            d.delay(0.5)
        d = ds.Square().delay(-2)
        np.testing.assert_array_equal(d[-3:3], np.array([-1, 1, 1, 1, 1, 1]))

    def test_scale(self):
        ''' Square (discrete): shift, delay '''
        d = ds.Square()
        with self.assertRaises(ValueError):
            d.scale(sp.pi)
        d = ds.Square().scale(3)
        np.testing.assert_array_equal(d[-3:3], np.array([-1, -1, -1, 1, 1, 1]))
        d = ds.Square().scale(0.75)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([1, -1, -1, 1, 1, -1]))
        d = ds.Square(ds.n-1).scale(1.5)
        np.testing.assert_array_equal(d[-12:12:4],
                                      np.array([1, 1, -1, -1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
