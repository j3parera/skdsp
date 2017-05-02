"""
This module defines an abstract base class for a `signal`.
This class implements all the common methods of signals.

"""

from ..operator.operator import AbsOperator, HermitianOperator
from ..operator.operator import ConjugateOperator, RealPartOperator
from ..operator.operator import FlipOperator, ShiftOperator
from ..operator.operator import ImaginaryPartOperator
from ..operator.operator import ScaleOperator
from ._util import _is_real_scalar
from ._util import _latex_mode
from abc import ABC, abstractproperty
from copy import deepcopy
from numbers import Number
import numpy as np
import re
import sympy as sp
from operator import __truediv__


class _Signal(ABC):
    """
    Abstract base class for all kind of signals.

    Attributes:
        name (str): Name of the signal; defaults to
            :math:`xk`
            :math:`k`
            an autoincremented integer.
    """

    """
    Signal registry. Maintains a dictionary of created signals names to avoid
    using the same name for more than one signal. Registering is performed
    automatically at class creation. In order to de-register, the signal
    must be 'deleted' using `del s`
    """
    _registry = {}

    """
    Last name index used for automatic signal name assigment.
    """
    _last_name_idx = -1

    @classmethod
    def _register(cls, name):
        """
        Registers a signal name.

        Args:
            name (str): The name to be registered
        """
        cls._registry[name] = True

    @classmethod
    def _deregister(cls, name):
        """
        Deregisters a signal name.

        Args:
            name (str): The name to be deregistered
        """
        try:
            del cls._registry[name]
        except:
            pass

    @classmethod
    def _check_name(cls, name):
        """
        Checks a signal name for duplicate or reserved.

        Args:
            name (str): The name to be checked.
        """
        if name in cls._registry:
            raise ValueError("Duplicate signal name")
        if re.match('x\d+', name):
            raise ValueError("Signal names 'xk' are reserved.")
        return name

    def __new__(cls, *args):
        """
        Signal common allocation.

        Args:
            args (str): Whatever list of arguments to be internally held.
        """
        obj = object.__new__(cls)
        obj._args = args
        return obj

    def __del__(self):
        """
        Signal deallocation.
        """
        self._deregister(self._name)

    def __init__(self, **kwargs):
        """
        Common initialization for all signals.

        Args:
            kwargs['name'] (str): Signal's name. If not provided the signal is
            assigned a unique reserved name.
            kwargs['cmplx'] (bool): If True, signal is complex; defaults to
            False.
        """
        self._period = None
        self._xvar = None
        self._xexpr = None
        if 'cmplx' in kwargs and kwargs['cmplx'] == True:
            self._dtype = np.complex_
        else:
            self._dtype = np.float_
        if 'name' in kwargs:
            self._name = kwargs['name']
        else:
            self._name = 'x' + str(_Signal._last_name_idx + 1)
            _Signal._last_name_idx += 1
        self._register(self._name)

    @property
    def args(self):
        """
        Internally held arguments.
        """
        return self._args

    @property
    def name(self):
        """
        The signal's name.
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        Assigns a name to the signal.
        """
        if self._name == value:
            return
        self._name = self._check_name(value)

    def latex_name(self, mode=None):
        """
        The signal's name in
        :math:`\LaTeX`
        .

        Args:
            mode (str): if mode='inline' the signal's name is surrounded by
            $ signs.
        """
        m = re.match(r'(\D+)(\d+)', self._name)
        if m:
            s = m.group(1) + '_{{0}}'.format(m.group(2))
        else:
            s = self._name
        return _latex_mode(s, mode)

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
        """ Tests whether the signal is real valued."""
        return self._dtype == np.float_

    @property
    def is_complex(self):
        """ Tests whether the signal is complex valued."""
        return self._dtype == np.complex_

    @abstractproperty
    def is_discrete(self):
        """ Tests whether the signal is discrete."""
        pass

    @abstractproperty
    def is_continuous(self):
        """ Tests whether the signal is continuous."""
        pass

    @property
    def is_periodic(self):
        """ Tests whether the signal is periodic."""
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

    def __repr__(self):
        """ Signal's `repr()`esentation. """
        return "Generic '_Signal' object"

    def __eq__(self, other):
        """ Tests equality with other signal. """
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

    # --- math wrappers -------------------------------------------------------
    # math functions must not arrive here, they must be previously catched
    def __add__(self, other):
        """ Signal addition:
        :math:`z[n] = x[n] + y[n]`,
        :math:`z(t) = x(t) + y(t)`.
        """
        raise NotImplementedError('({0}).__add__'.format(self))

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        """ Signal substraction:
        :math:`z[n] = x[n] - y[n]`,
        :math:`z(t) = x(t) - y(t)`.
        """
        raise NotImplementedError('({0}).__sub__'.format(self))

    __rsub__ = __sub__
    __isub__ = __sub__

    def __neg__(self):
        """ Signal sign inversion:
        :math:`y[n] = -x[n]`,
        :math:`y(t) = -x(t)`.
        """

    def __mul__(self, other):
        """ Signal multiplication:
        :math:`z[n] = x[n] * y[n]`,
        :math:`z(t) = x(t) * y(t)`.
        """
        raise NotImplementedError('({0}).__mul__'.format(self))

    __rmul__ = __mul__
    __imul__ = __mul__

    def __pow__(self, other):
        """ Signal exponentiation:
        :math:`z[n] = x[n]^{y[n]}`,
        :math:`z(t) = x(t)^{y(t)}`.
        """
        raise NotImplementedError('({0}).__pow__'.format(self))

    def __truediv__(self, other):
        """ Signal division:
        :math:`z[n] = x[n] / y[n]`,
        :math:`z(t) = x(t) / y(t)`.
        """
        raise NotImplementedError('({0}).__truediv__'.format(self))

    __rtruediv__ = __truediv__
    __itruediv__ = __truediv__

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
        if self.is_complex:
            s._yexpr = RealPartOperator.apply(s._xvar, s._yexpr)
            s._dtype = np.float_
        return s

#    @property
    def imag(self):
        """
        Imaginary part of the signal.

        Returns:
            A signal copy with the imaginary part of the signal if it's
            complex, else None.
        """
        if self.is_complex:
            s = deepcopy(self)
            s._yexpr = ImaginaryPartOperator.apply(s._xvar, s._yexpr)
            s._dtype = np.float_
            return s
        return None


class _SignalOp(_Signal):
    """
    Tag for a class holding an operation (add, mul or pow) on signals.
    """
    pass


class _FunctionSignal(_Signal):
    """
    Generic class for a signal defined by a mathematical expression.
    """
    def __init__(self, expr, **kwargs):
        """
        Common initialization for all functional signals.

        Args:
            expr: The sympy mathematical expression that defines the signal.
        """
        super().__init__(**kwargs)
        if not isinstance(expr, sp.Expr):
            raise TypeError("'expr' must be a sympy expression")
        if expr.is_number:
            # just in case is a symbol or constant
            self._yexpr = expr
            self._xvar = self._default_xvar()
            self._xexpr = self._xvar
        else:
            fs = expr.free_symbols
            if len(fs) != 1:
                raise TypeError("'expr' must contain a free symbol")
            self._yexpr = expr
            self._xvar = fs.pop()
            self._xexpr = self._xvar

    @property
    def yexpr(self):
        return self._yexpr

    def latex_yexpr(self):
        """
        A
        :math:`\LaTeX`
        representation of the signal expression.
        """
        return self.__str__()

    def __str__(self):
        if hasattr(self.__class__, '_print'):
            return self._print()
        if self._yexpr.is_number:
            return sp.Basic.__str__(self._yexpr)
        return sp.Basic.__str__(self._yexpr.args[0])

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

    def __abs__(self):
        s = deepcopy(self)
        s._yexpr = AbsOperator.apply(s._xvar, s._yexpr)
        return s

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return sp.Eq(other._yexpr.xreplace({other._xvar: self._xvar}) -
                     self._yexpr, 0) == True

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
