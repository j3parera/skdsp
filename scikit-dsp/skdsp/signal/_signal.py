"""
This module defines an abstract base class for a `signal`.
This class implements all the common methods of signals.

"""

from ..operator.operator import AbsOperator, HermitianOperator
from ..operator.operator import ConjugateOperator
from ..operator.operator import FlipOperator, ShiftOperator
from ..operator.operator import ScaleOperator, GainOperator
from ._util import _is_real_scalar
from abc import ABC, abstractproperty
from copy import deepcopy
from numbers import Number
import numpy as np
import sympy as sp


class _Signal(ABC):
    ''' _Signal is the abstract base class
    '''
    def __init__(self):
        """Common *__init__* method for all signals.
        """
        self._dtype = np.float_
        self._period = None
        self._xexpr = None
        self._xvar = None
        self.name = 'x'

    def copy(self):
        return deepcopy(self)

    def _copy_to(self, other):
        other._dtype = self._dtype
        other._period = self._period
        other._xexpr = self._xexpr
        other._xvar
        other.name = self.name

    @property
    def dtype(self):
        """ Type of the signal values. One of *np.float_* or *np.complex_*."""
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if value not in (np.float_, np.complex_):
            raise ValueError('only signal types float or complex allowed')
        self._dtype = value

    @property
    def xexpr(self):
        """ Symbolic *time* expression."""
        return self._xexpr

    @property
    def xvar(self):
        """ Symbolic *time* variable used in the symbolic expression
        `xexpr`."""
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
        return self.dtype == np.float_

    @property
    def is_complex(self):
        """ Tests weather the signal is complex valued."""
        return self.dtype == np.complex_

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

    def eval(self, r):
        pass

    def __str__(self):
        if hasattr(self.__class__, '_print'):
            return self._print()
        return sp.Basic.__str__(self._yexpr)

#     def __repr__(self):
#         return self.__str__()

    # --- eval wrappers -------------------------------------------------------
    def __getitem__(self, idx):
        """
        getitemmmm
        """
        if isinstance(idx, slice):
            return self.eval(np.arange(idx.start, idx.stop, idx.step))
        elif isinstance(idx, np.ndarray):
            return self.eval(idx)
        else:
            return self.eval(idx)

    # --- operadores temporales -----------------------------------------------
    def flip(self):
        s = self.__class__._factory(self)
        s._xexpr = FlipOperator.apply(s._xvar, s._xexpr)
        return s

    __reversed__ = flip

    def shift(self, k):
        s = self.__class__._factory(self)
        s._xexpr = ShiftOperator.apply(s._xvar, s._xexpr, k)
        return s

    def delay(self, k):
        s = self.__class__._factory(self)
        s._xexpr = ShiftOperator.apply(s._xvar, s._xexpr, k)
        return s

    def scale(self, v):
        s = self.__class__._factory(self)
        s._xexpr = ScaleOperator.apply(s._xvar, s._xexpr, v)
        return s

    def generate(self, s0=0, step=1, size=1, overlap=0):
        ''' Generador de señal
        devuelve trozos de señal de tamaño 'size', muestreados cada 'step'
        unidades, empezando desde s0; cada trozo solapa 'overlap' muestras
        con el anterior.
        Ejemplos:
        1) (s0=0, step=1, size=3, overlap=2) devuelve
        (s[0], s[1], s[2]), (s[1], s[2], s[3]), (s[2], s[3], s[4]), ...
        2) (s0=-1, step=1, size=2, overlap=0) devuelve
        (s[-1], s[0]), (s[1], s[2]), (s[3], s[4]), ...
        3) (s0=0, step=0.1, size=3, overlap=0.1) devuelve
        (s[0], s[0.1], s[0.2]), (s[0.1], s[0.2], s[0.3)),
        (s[0.2], s[0.3], s[0.4), ...
        '''
        s = s0
        while True:
            sl = np.linspace(s, s+(size*step), size, endpoint=False)
            yield self[sl]
            s += size*step - overlap


class _FunctionSignal(_Signal):

    def __init__(self, expr):
        _Signal.__init__(self)
        if not isinstance(expr, sp.Expr):
            raise TypeError("'expr' must be a sympy expression")
        if expr.is_number:
            # just in case is a symbol or constant
            self._yexpr = sp.Expr(expr)
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
        self._yexpr *= other._yexpr.xreplace({other._xvar: self._xvar})
        return self

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, other):
        if other.dtype == np.complex_:
            self.dtype = np.complex_
        self._yexpr /= other._yexpr.xreplace({other._xvar: self._xvar})
        return self

    __itruediv__ = __truediv__

    def __rtruediv__(self, other):
        if self.dtype == np.complex_:
            other.dtype = np.complex_
        other._yexpr /= self._yexpr.xreplace({self._xvar: other._xvar})
        return 1/other

    def __add__(self, other):
        if other.dtype == np.complex_:
            self.dtype = np.complex_
        self._yexpr += other._yexpr.xreplace({other._xvar: self._xvar})
        return self

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        if other.dtype == np.complex_:
            self.dtype = np.complex_
        self._yexpr -= other._yexpr.xreplace({other._xvar: self._xvar})
        return self

    def __rsub__(self, other):
        if self.dtype == np.complex_:
            other.dtype = np.complex_
        other._yexpr -= self._yexpr.xreplace({self._xvar: other._xvar})
        return -other

    __isub__ = __sub__

    def __neg__(self):
        self._yexpr = GainOperator.apply(self._xvar, self._yexpr, -1)
        return self

    def __abs__(self):
        self._yexpr = AbsOperator.apply(self._xvar, self._yexpr)
        return self

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
#         # TODO: ¿es correcto? NO si las variables no son iguales
#         return str(self).__eq__(str(other))
#         if isinstance(other, _FunctionSignal):
#             return self._yexpr == other._yexpr
#         d = self._yexpr - other
#         if (sp.expand(d) == 0) or \
#            (sp.simplify(d) == 0) or \
#            (sp.trigsimp(d) == 0):
#             return True
#         else:
#             return False

    def range(self, dB=False):
        dr = self.max() - self.min()
        if dB:
            return 20*sp.log(dr, 10)
        return dr

    @property
    def even(self):
        s1 = self.__class__._factory(self)
        s2 = self.__class__._factory(self)
        s2._yexpr = HermitianOperator.apply(s2._xvar, s2._yexpr)
        return sp.Rational(1, 2)*(s1 + s2)

    @property
    def odd(self):
        s1 = self.__class__._factory(self)
        s2 = self.__class__._factory(self)
        s2._yexpr = HermitianOperator.apply(s2._xvar, s2._yexpr)
        return sp.Rational(1, 2)*(s1 - s2)

    @property
    def conjugate(self):
        s = self.__class__._factory(self)
        s._yexpr = ConjugateOperator.apply(s._xvar, s._yexpr)
        return s
