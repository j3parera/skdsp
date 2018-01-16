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
from .printer import latex
from abc import abstractproperty
from numbers import Number
import numpy as np
import re
import sympy as sp


class _Signal(sp.Basic):
    """
    Abstract base class for all kind of signals.

    Attributes:
        name (str): Name of the signal; defaults to
            :math:`xk`
            :math:`k`
            an autoincremented integer.
    """

    # To be overridden with True in the appropriate subclasses
    is_Signal = True
    is_FunctionSignal = False
    is_Continuous = False
    is_Discrete = False
    is_DataSignal = False

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
        except BaseException:
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

    def _clone(self):
        s = self.func(*self.args)
        s._xexpr = self._xexpr
        s._yexpr = self._yexpr
        return s

    def __del__(self):
        """
        Signal deallocation.
        """
        if hasattr(self, '_name'):
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
        if 'cmplx' in kwargs and kwargs['cmplx'] is True:
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

    @property
    def latex_name(self):
        """
        The signal's name in
        :math:`\LaTeX`
        .
        """
        m = re.match(r'(\D+)(\d+)', self._name)
        if m:
            s = m.group(1) + '_{{0}}'.format(m.group(2))
        else:
            s = self._name
        return s

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

    @classmethod
    def _sympify_xexpr(cls, expr):
        expr = sp.sympify(expr)
        if len(expr.free_symbols) == 0:
            raise ValueError('xexpr must be have at least a variable')
        return expr

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
    def dtype_is_real(self):
        """ Tests whether the signal is real valued."""
        return self._dtype == np.float_

    @property
    def dtype_is_complex(self):
        """ Tests whether the signal is complex valued."""
        return self._dtype == np.complex_

    @property
    def is_continuous(self):
        return self.is_Continuous

    @property
    def is_discrete(self):
        return self.is_Discrete

    @property
    def is_data(self):
        return self.is_DataSignal

    @property
    def is_function(self):
        return self.is_FunctionSignal

    @property
    def is_signal(self):
        return self.is_Signal

    @property
    def is_periodic(self):
        """ Tests whether the signal is periodic."""
        return self.period is not None and self.period != sp.oo

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
            raise ValueError('`period` must be a real scalar')

    def __repr__(self):
        """ Signal's `repr()`esentation. """
        return "Generic '_Signal' object"

    def _latex(self, *_args, **_kwargs):
        return latex(self)

    __str__ = __repr__
    _sympystr = __repr__
    _sympyrepr = __repr__

    def __eq__(self, other):
        """ Tests equality with other signal. """
        equal = other._dtype == self._dtype and other._period == self._period
        if not equal:
            return False
        return sp.Eq(other._xexpr.xreplace({other._xvar: self._xvar}) -
                     self._xexpr, 0) == sp.S.true

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
        s = self._clone()
        s._xexpr = FlipOperator.apply(s._xvar, s._xexpr)
        return s

    __reversed__ = flip

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
        s = self._clone()
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
        s = self._clone()
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
        s = self._clone()
        if self.dtype_is_complex:
            s._yexpr = RealPartOperator.apply(s._xvar, s._yexpr)
            s._dtype = np.float_
        return s

    @property
    def imag(self):
        """
        Imaginary part of the signal.

        Returns:
            A signal copy with the imaginary part of the signal if it's
            complex, else None.
        """
        if self.dtype_is_complex:
            s = self._clone()
            s._yexpr = ImaginaryPartOperator.apply(s._xvar, s._yexpr)
            s._dtype = np.float_
            return s
        return None

    # --- utilities -----------------------------------------------------------
    @property
    def latex_xexpr(self):
        if self.xexpr == self.xvar:
            sn = sp.latex(self._xexpr)
        else:
            # just in case it appears '- n' with space
            s0 = sp.latex(self.xexpr)
            if s0.startswith('- '):
                s0 = s0.replace(' ', '', 1)
            sn = r'\left(' + s0 + r'\right)'
        return sn

    @property
    def str_xexpr(self):
        if self.xexpr == self.xvar:
            sn = sp.latex(self._xexpr)
        else:
            sn = '(' + str(self.xexpr) + ')'
        return sn


class _SignalOp(_Signal):
    """
    Tag for a class holding an operation (add or mul) on signals.
    """
    class OperationNotDefined(Exception):
        def __init__(self):
            Exception.__init__('Operation not defined for signals')


class _FunctionSignal(_Signal):
    """
    Generic class for a signal defined by a mathematical expression.
    """

    is_FunctionSignal = True

    def __init__(self, xexpr, yexpr, **kwargs):
        """
        Common initialization for all functional signals.

        Args:
            expr: The sympy mathematical expression that defines the signal.
        """
        super().__init__(**kwargs)
        self._xexpr = xexpr
        self._yexpr = yexpr
        fs = xexpr.free_symbols
        # Si hay más de un símbolo, p.e. (k-n), (n-m), se selecciona el primero
        self._xvar = self.default_xvar() if len(fs) == 0 else next(iter(fs))
        self._ylambda = None

    @property
    def yexpr(self):
        return self._yexpr

    def __str__(self):
        if hasattr(self.__class__, '_print'):
            return self._print()
        return sp.Basic.__str__(self._yexpr)

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
            if self._ylambda is None:
                self._ylambda = sp.lambdify(self._xvar, self._yexpr, 'numpy')
            y = self._ylambda(x)
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
        s = self._clone()
        s._yexpr = AbsOperator.apply(s._xvar, s._yexpr)
        return s

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return sp.Eq(other._yexpr.xreplace({other._xvar: self._xvar}) -
                     self._yexpr, 0) == sp.S.true

    @property
    def is_even(self):
        return sp.Eq(self._yexpr,
                     self._yexpr.xreplace({self._xvar: -self._xvar}))

    @property
    def is_odd(self):
        return sp.Eq(self._yexpr,
                     -self._yexpr.xreplace({self._xvar: -self._xvar}))

    @property
    def even(self):
        if self.is_even:
            return self
        if self.is_odd:
            return sp.S.Zero
        s1 = self._clone()
        s2 = self._clone()
        s2._xexpr = FlipOperator.apply(s2._xvar, s2._xexpr)
        s2._yexpr = HermitianOperator.apply(s2._xvar, s2._yexpr)
        return sp.S.Half*(s1 + s2)

    @property
    def odd(self):
        if self.is_odd:
            return self
        if self.is_even:
            return sp.S.Zero
        s1 = self._clone()
        s2 = self._clone()
        s2._xexpr = FlipOperator.apply(s2._xvar, s2._xexpr)
        s2._yexpr = HermitianOperator.apply(s2._xvar, s2._yexpr)
        return sp.S.Half*(s1 - s2)

    @property
    def conjugate(self):
        s = self._clone()
        s._yexpr = ConjugateOperator.apply(s._xvar, s._yexpr)
        return s

    def magnitude(self, dB=False):
        m = abs(self)
        if dB:
            m._yexpr = 20*sp.log(m._yexpr, 10)
        return m
