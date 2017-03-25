"""
This module defines an abstract base class for a `signal`.
This class implements all the common methods of signals.

"""

from ..operator.operator import AbsOperator, HermitianOperator
from ..operator.operator import ConjugateOperator, RealPartOperator
from ..operator.operator import ImaginaryPartOperator
from ..operator.operator import FlipOperator, ShiftOperator
from ..operator.operator import ScaleOperator
from ._util import _is_real_scalar
from abc import ABC, abstractproperty
from copy import deepcopy
from numbers import Number
import numpy as np
import sympy as sp


class _Signal(ABC):
    """
    Abstract base class for all kind of signals.

    Attributes:
        name (str): Name of the signal; defaults to `'x'`.

    """
    def __init__(self):
        """
        Common `__init__()` method for all signals.
        """
        self._dtype = np.float_
        self._period = None
        self._xvar = None
        self._xexpr = None
        self.name = 'x'

    # def __deepcopy__(self, memo):
    #    """ Este deepcopy no es, en principio, necesario, salvo que
    #    haya que hacer algo con las expresiones sympy por el bug #7672.
    #    No hace lo mismo que la función deepcopy, pero parece suficiente.
    #    """
    #    cls = self.__class__
    #    result = cls.__new__(cls)
    #    memo[id(self)] = result
    #    for k, v in self.__dict__.items():
    #        setattr(result, deepcopy(k, memo), deepcopy(v, memo))
    #    return result

    def _copy_to(self, other):
        other._dtype = self._dtype
        other._period = self._period
        other._xexpr = self._xexpr
        other._xvar = self._xvar
        other.name = self.name

    @property
    def dtype(self):
        """
        Type of the signal values.\n
        Note that if `dtype` changes from complex to float, the signal's
        `yexpr` is replaced by its real part and thus the imaginary part
        is destructively lost.

        Returns:
            The type of the signal values; one of Numpy `float_` or
            `complex_`.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if value not in (np.float_, np.complex_):
            raise ValueError('only signal types float or complex allowed')
        if self._dtype == np.complex_:
            self._yexpr = sp.re(self._yexpr)
        self._dtype = value

    @property
    def xexpr(self):
        """
        `Independent variable` (time, frequency...) expression.

        Returns:
            The Sympy expression for the signal's independent variable.
        """
        return self._xexpr

    @property
    def xvar(self):
        """
        `Independent variable` (time, frequency...) expression.

        Returns:
            The Sympy signal's independent variable.
        """
        return self._xvar

    @xvar.setter
    def xvar(self, newvar):
        vmap = {self._xvar: newvar}
        self._xexpr = self._xexpr.xreplace(vmap)
        self._yexpr = self._yexpr.xreplace(vmap)
        self._xvar = newvar

    @property
    def is_real(self):
        """ Tests weather the signal is real valued."""
        return self._dtype == np.float_

    @property
    def is_complex(self):
        """ Tests weather the signal is complex valued."""
        return self._dtype == np.complex_

    @abstractproperty
    def is_discrete(self):
        """ Tests weather the signal is discrete."""
        pass

    @abstractproperty
    def is_continuous(self):
        """ Tests weather the signal is continuous."""
        pass

    @property
    def is_periodic(self):
        """ Tests weather the signal is periodic."""
        return self._period is not None and self._period != sp.oo

    @property
    def period(self):
        """ Period of the signal. One of *value* (if signal is periodic),
        *inf* (if signal is not periodic) or *None* (if signal periodicity
        could not be determined)."""
        # Calcular el periodo de una señal genérica es difícil
        # Si se intenta hacer con sympy solve(expr(var)-expr(var+T), T)
        # se obtienen resultados raros
        return self._period

    @period.setter
    def period(self, value):
        v = sp.sympify(value)
        if _is_real_scalar(v):
            self._period = v
        else:
            raise TypeError('`period` must be a real scalar')

    def __str__(self):
        if hasattr(self.__class__, '_print'):
            return self._print()
        if self._yexpr.is_number:
            return sp.Basic.__str__(self._yexpr)
        return sp.Basic.__str__(self._yexpr.args[0])

    def __repr__(self):
        return "Generic '_Signal' object"

    def latex(self):
        """
        A
        :math:`\LaTeX`
        representation of the signal.
        """
        return self.__str__()

    def __eq__(self, other):
        equal = other._dtype == self._dtype and other._period == self._period
        if not equal:
            return False
        return sp.Eq(other._xexpr.xreplace({other._xvar: self._xvar}) -
                     self._xexpr, 0) == True

    # --- eval wrappers -------------------------------------------------------
    @abstractproperty
    def eval(self, r):
        """ Returns value(s) of signal evaluated at 'r'."""
        pass

    def __getitem__(self, key):
        """
        Evaluates signal at `key`; i.e
        :math:`y[key]`,
        :math:`y(key)`.

        Args:
            key: Range of index values of the signal; could be either a single
            idx or a slice.

        Returns:
            Signal values at `key`.

        """
        if isinstance(key, slice):
            return self.eval(np.arange(key.start, key.stop, key.step))
        else:
            return self.eval(key)

    def generate(self, s0=0, step=1, size=1, overlap=0):
        """
        Signal value generator. Evaluates signal value chunks of size `size`,
        every `step` units, starting at `s0`. Each chunk overlaps `overlap`
        values with the previous chunk.

        Args:
            s0: Starting index of chunk; defaults to 0.
            step: Distance between indexes; defaults to 1.
            size: Number of values in chunk; defaults to 1.
            overlap: Number of values overlapped between chunks; defaults to 0.
                idx or a slice.

        Yields:
            Numpy array: A chunk of signal values.

        Examples:
            1. `generate(s0=0, step=1, size=3, overlap=2)` returns
            [s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]], ...

            2. `generate(s0=-1, step=1, size=2, overlap=0)` returns
            [s[-1], s[0]], [s[1], s[2]], [s[3], s[4]], ...

            3. `generate(s0=0, step=0.1, size=3, overlap=0.1)` returns
            [s[0], s[0.1], s[0.2]], [s[0.1], s[0.2], s[0.3]],
            [s[0.2], s[0.3], s[0.4]], ...

        """
        s = s0
        while True:
            sl = np.linspace(s, s+(size*step), size, endpoint=False)
            yield self[sl]
            s += size*step - overlap

    # --- independent variable operations -------------------------------------
    def flip(self):
        """
        Inverts the independent variable; i.e.
        :math:`y[n] = x[-n]`,
        :math:`y(t) = x(-t)`.

        Returns:
            A signal copy with the independent variable inverted.

        """
        s = deepcopy(self)
        s._xexpr = FlipOperator.apply(s._xvar, s._xexpr)
        return s

    def shift(self, k):
        """
        Shifts the independent variable; i.e.
        :math:`y[n] = x[n-k]`,
        :math:`y(t) = x(t-k)`.

        Args:
            k: The amount of shift.

        Returns:
            A signal copy with the independent variable shifted.

        """
        s = deepcopy(self)
        s._xexpr = ShiftOperator.apply(s._xvar, s._xexpr, k)
        return s

    def delay(self, k):
        """
        Delays the signal by shifting the independent variable; i.e.
        :math:`y[n] = x[n-k]`,
        :math:`y(t) = x(t-k)`.

        Args:
            k: The amount of shift.

        Returns:
            A delayed signal copy.

        """
        return self.shift(k)

    def scale(self, v, mul=True):
        """
        Scales the the independent variable; i.e.
        :math:`y[n] = x[v*n]`,
        :math:`y(t) = x(v*t)`.

        Args:
            v: The amount of scaling.
            mul (bool): If True, the scale multiplies, else
                divides.

        Returns:
            A signal copy with the independent variable scaled.

        """
        s = deepcopy(self)
        s._xexpr = ScaleOperator.apply(s._xvar, s._xexpr, v, mul)
        return s

    # --- operations ----------------------------------------------------------
    @property
    def real(self):
        """
        Real part of the signal.

        Returns:
            A signal copy with the real part of the signal.

        """
        s = deepcopy(self)
        s._yexpr = RealPartOperator.apply(s._xvar, s._yexpr)
        s._dtype = np.float_
        return s

    @property
    def imag(self):
        """
        Imaginary part of the signal.

        Returns:
            A signal copy with the imaginary part of the signal.

        """
        s = deepcopy(self)
        s._yexpr = ImaginaryPartOperator.apply(s._xvar, s._yexpr)
        s._dtype = np.float_
        return s


class _FunctionSignal(_Signal):

    def __init__(self, expr):
        super().__init__()
        if not isinstance(expr, sp.Expr):
            raise TypeError("'expr' must be a sympy expression")
        if expr.is_number:
            # just in case is a symbol or constant
            self._yexpr = expr
            self._xvar = self._default_xvar()
            self._xexpr = sp.Expr(self._xvar)
        else:
            fs = expr.free_symbols
            if len(fs) != 1:
                raise TypeError("'expr' must contain a free symbol")
            self._yexpr = expr
            self._xvar = fs.pop()
            self._xexpr = self._xvar

    def _copy_to(self, other):
        other._yexpr = self._yexpr
        other._xexpr = self._xexpr
        other._xvar = self._xvar
        _Signal._copy_to(self, other)

    @property
    def yexpr(self):
        return self._yexpr

    def eval(self, x):
        # Hay que ver si hay 'Pow'
        to_real = False
        pows = []
        for arg in sp.preorder_traversal(self._yexpr):
            if isinstance(arg, sp.Pow):
                pows.append(arg)
        for p in pows:
            base = p.args[0]
            if isinstance(base, (Number, sp.Number)):
                if base <= 0:
                    # base negativa, los exponentes deben ser complejos
                    # por si acaso no son enteros
                    x = x.astype(np.complex_)
                    self.dtype = np.complex_
                    to_real = True
                    # break # ??
        try:
            ylambda = sp.lambdify(self._xvar, self._yexpr, 'numpy')
            y = ylambda(x)
            if not hasattr(y, "__len__"):
                # workaround para issue #5642 de sympy. Cuando yexpr es una
                # constante, se devuelve un escalar aunque la entrada sea un
                # array
                y = np.full(x.shape, y, self.dtype)
            if not to_real:
                y = y.astype(self.dtype)
        except (NameError, ValueError):
            # sympy no ha podido hacer una función lambda
            # (o hay algún problema de cálculo, p.e 2^(-1) enteros)
            # así que se procesan los valores uno a uno
            y = np.zeros_like(x, self.dtype)
            for k, x0 in enumerate(x):
                try:
                    y[k] = self._yexpr.xreplace({self._xvar: x0})
                except TypeError:
                    y[k] = np.nan
        if to_real:
            y = np.real_if_close(y)
        return y

    # -- operadores temporales ----------------------------------------------
    #    _Signal.xxxx hace la copia de señal y aplica el operador a xexpr

    def flip(self):
        s = _Signal.flip(self)
        s._yexpr = FlipOperator.apply(s._xvar, s._yexpr)
        return s

    __reversed__ = flip

    def shift(self, k):
        s = _Signal.shift(self, k)
        s._yexpr = ShiftOperator.apply(s._xvar, s._yexpr, k)
        return s

    def scale(self, v):
        s = _Signal.scale(self, v)
        s._yexpr = ScaleOperator.apply(s._xvar, s._yexpr, v)
        return s

    # -- operadores aritméticos ----------------------------------------------
    def __mul__(self, other):
        if other.dtype == np.complex_:
            self.dtype = np.complex_
        s = deepcopy(self)
        s._yexpr *= other._yexpr.xreplace({other._xvar: s._xvar})
        return s

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, other):
        if other.dtype == np.complex_:
            self.dtype = np.complex_
        s = deepcopy(self)
        s._yexpr /= other._yexpr.xreplace({other._xvar: s._xvar})
        return s

    __itruediv__ = __truediv__

    def __rtruediv__(self, other):
        if self.dtype == np.complex_:
            other.dtype = np.complex_
        s = deepcopy(other)
        s._yexpr /= self._yexpr.xreplace({self._xvar: s._xvar})
        return 1/other

    def __add__(self, other):
        if other.dtype == np.complex_:
            self.dtype = np.complex_
        s = deepcopy(self)
        s._yexpr += other._yexpr.xreplace({other._xvar: s._xvar})
        return s

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        if other.dtype == np.complex_:
            self.dtype = np.complex_
        s = deepcopy(self)
        s._yexpr -= other._yexpr.xreplace({other._xvar: s._xvar})
        return s

    def __rsub__(self, other):
        if self.dtype == np.complex_:
            other.dtype = np.complex_
        s = deepcopy(other)
        s._yexpr -= self._yexpr.xreplace({self._xvar: s._xvar})
        return -s

    __isub__ = __sub__

    def __neg__(self):
        s = deepcopy(self)
        s._yexpr = -self._yexpr
        return s

    def __abs__(self):
        s = deepcopy(self)
        s._yexpr = AbsOperator.apply(s._xvar, s._yexpr)
        return s

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return sp.Eq(other._yexpr.xreplace({other._xvar: self._xvar}) -
                     self._yexpr, 0) == True

#         if isinstance(other, _FunctionSignal):
#             return self._yexpr == other._yexpr
#         d = self._yexpr - other
#         if (sp.expand(d) == 0) or \
#            (sp.simplify(d) == 0) or \
#            (sp.trigsimp(d) == 0):
#             return True
#         else:
#             return False

    @property
    def even(self):
        s1 = deepcopy(self)
        s2 = deepcopy(self)
        s2._yexpr = HermitianOperator.apply(s2._xvar, s2._yexpr)
        return sp.Rational(1, 2)*(s1 + s2)

    @property
    def odd(self):
        s1 = deepcopy(self)
        s2 = deepcopy(self)
        s2._yexpr = HermitianOperator.apply(s2._xvar, s2._yexpr)
        return sp.Rational(1, 2)*(s1 - s2)

    @property
    def conjugate(self):
        s = deepcopy(self)
        s._yexpr = ConjugateOperator.apply(s._xvar, s._yexpr)
        return s
