import unittest

from skdsp.signal.discrete import Delta, Step
from skdsp.transform.discrete import ZTransform, DTFT

import sympy as sp


class ZTransformTest(unittest.TestCase):

    def test_delta(self):
        d = Delta()
        # self.assertEqual(d.zt, ZTransform(ZTransform.default_var,
        #                                  sp.sympify(1),
        #                                  sp.Interval(0, sp.oo)))
        # self.assertEqual(d.dtft, DTFT(DTFT.default_var, sp.sympify(1)))
        print(d)

    def test_step(self):
        d = Step()
        # z = ZTransform.default_var
        # self.assertEqual(d.zt, ZTransform(z, sp.sympify(1/(1-z**(-1))),
        #                                  sp.Interval(1, sp.oo, True)))
        # omega = DTFT.default_var
        # self.assertEqual(d.dtft, DTFT(omega, sp.DiracDelta(omega) +
        #                              sp.sympify(1/(1-sp.exp(-sp.I*omega)))))
        print(d)

if __name__ == "__main__":
    unittest.main()
