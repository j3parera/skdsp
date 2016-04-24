import math

import numpy as np
from skdsp.operator.operator import ShiftOperator
from skdsp.signal.signal import FunctionSignal
import sympy as sp

# TODO: todo, todito

__all__ = ['ContinuousMixin', 'Sinusoid']


class ContinuousMixin(object):

    def __init__(self):
        self._time_var = sp.symbols('t', real=True)

    def _check_eval(self, x):
        pass


class ContinuousFunctionSignal(FunctionSignal, ContinuousMixin):

    def eval(self, x):
        ContinuousMixin._check_eval(self, x)
        if not isinstance(x, np.ndarray):
            x = np.array([x])
        return FunctionSignal.eval(self, x)


class Sinusoid(ContinuousFunctionSignal):

    @classmethod
    def _func(cls, *args):
        """ Signal function
        Parameters
        ----------
        cls : class
        args[0] : numpy array with time indexes
        args[1] : numpy data type of the result

        Returns
        -------
        y : numpy array with signal values evaluated at args[0] with type
        as arg[1]

        Notes
        -----
        .. versionadded:: 0.0.1
        """
        return np.cos(args[0]).astype(args[1])

    def __init__(self, Omega=1, Phi=0):
        ContinuousMixin.__init__(self)
        FunctionSignal.__init__(self, Sinusoid._func)
        self._phi_op = ShiftOperator(math.fmod(-Phi, 2*math.pi))
        self._time_operators.append(self._phi_op)
        # self._omega_op = TimeScaleOperator(Omega)
        self._time_operators.append(self._omega_op)
        self._texpr = None

    def __repr__(self, *args, **kwargs):
        self._compose_texpr()
        return 'cos(' + str(self._texpr) + ')'

    @property
    def phase_shift(self):
        return self._phi_op._k

    @phase_shift.setter
    def phase_shift(self, Phi):
        self._phi_op._k = math.fmod(Phi, 2*math.pi)
        self._texpr = None
