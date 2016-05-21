from abc import ABC
from numbers import Integral, Number, Real
from skdsp.operator.operator import *
from sympy.functions.elementary.trigonometric import _pi_coeff
import numpy as np
import sympy as sp


__all__ = ['Signal', 'FunctionSignal', 'ConstantSignal']


class Signal(ABC):
    ''' Signal is the abstract base class
    '''
    @staticmethod
    def _check_is_real(x):
        ok = True
        if isinstance(x, sp.Expr):
            x = x.evalf()
            if not isinstance(x, sp.Float):
                ok = False
        elif not isinstance(x, Real):
            ok = False
        return ok

    @staticmethod
    def _check_is_integer(x):
        ok = True
        if isinstance(x, sp.Expr):
            x = x.evalf()
            if not isinstance(x, sp.Integer):
                ok = False
        elif not isinstance(x, Integral):
            ok = False
        return ok

    @staticmethod
    def _extract_omega(x):
        px = sp.arg(x)
        pc = _pi_coeff(px)
        if pc is not None:
            return sp.S.Pi*pc
        # última posibilidad para algunos caso raros
        # siempre y cuando la fase quede como (pi +) atan(algo) y
        # se haya pasado x como a*exp(sp.I*omega0)
        pisub = False
        if px.func == sp.Add:
            pisub = True
            if px.args[0].is_constant():
                pisubarg = px.args[0]
                px -= px.args[0]  # +- pi, supuestamente
        if px.func == sp.atan:
            if isinstance(x, sp.Expr):
                exponent = None
                if x.func == sp.exp:
                    exponent = x.args[0]
                elif x.func == sp.Mul and x.args[1].func == sp.exp:
                    exponent = x.args[1].args[0]
                if exponent is not None:
                    expoverj = exponent/sp.I
                    pc = _pi_coeff(expoverj)
                    if pc is not None:
                        return sp.S.Pi*pc
        if pisub:
            px += pisubarg
        return px.evalf()

    def __init__(self):
        self._dtype = np.float_
        self.name = 'x'
        self._period = None

    def _copy_to(self, other):
        other._dtype = self._dtype

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype not in (np.float_, np.complex_):
            raise ValueError('signal types float or complex allowed')
        self._dtype = dtype

    @property
    def xexpr(self):
        return self._xexpr

    @property
    def xvar(self):
        return self._xvar

    @xvar.setter
    def xvar(self, newvar):
        vmap = {self._xvar: newvar}
        self._xexpr = self._xexpr.xreplace(vmap)
        self._yexpr = self._yexpr.xreplace(vmap)
        self._xvar = newvar

    def is_real(self):
        return self._dtype == np.float_

    def is_complex(self):
        return self._dtype == np.complex_

    def is_periodic(self):
        return self.period != np.Inf and self._period is not None

    @property
    def period(self):
        # Calcular el periodo de una señal genérica es difícil
        # Si se intenta hacer con sympy solve(expr(var)-expr(var+T), T)
        # se obtienen resultados raros
        return self._period

    def eval(self, r):
        pass

    def __str__(self):
        if hasattr(self.__class__, '_print'):
            return self._print()
        return sp.Basic.__str__(self._yexpr)

    def __repr__(self):
        return self.__str__()

    # --- eval wrappers -------------------------------------------------------
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.eval(np.arange(idx.start, idx.stop, idx.step))
        elif isinstance(idx, np.ndarray):
            return self.eval(idx)
        else:
            return self.eval(idx)

    # -- operadores temporales ------------------------------------------------
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


class FunctionSignal(Signal):

    def __init__(self, expr):
        Signal.__init__(self)
        if not isinstance(expr, sp.Expr):
            raise TypeError("'expr' must be a sympy expression")
        if expr.is_number:
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

    def _copy_to(self, other):
        other._yexpr = self._yexpr
        other._xexpr = self._xexpr
        other._xvar = self._xvar
        Signal._copy_to(self, other)

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
                    self._dtype = np.complex_
                    to_real = True
                    break
        try:
            ylambda = sp.lambdify(self._xvar, self._yexpr, 'numpy')
            y = ylambda(x)
            if not hasattr(y, "__len__"):
                # workaround para issue #5642 de sympy. Cuando yexpr es una
                # constante, se devuelve un escalar aunque la entrada sea un
                # array
                y = np.full(x.shape, y, self._dtype)
            if not to_real:
                y = y.astype(self._dtype)
        except NameError:
            # sympy no ha podido hacer una función lambda
            # así que se procesan los valores uno a uno
            y = np.zeros_like(x, self._dtype)
            for k, x0 in enumerate(x):
                try:
                    y[k] = self._yexpr.xreplace({self._xvar: x0})
                except TypeError:
                    y[k] = np.nan
        if to_real:
            y = np.real_if_close(y)
        return y

    # -- operadores temporales ----------------------------------------------
    #    Signal.xxxx hace la copia de señal y aplica el operador a xexpr

    def flip(self):
        s = Signal.flip(self)
        s._yexpr = FlipOperator.apply(s._xvar, s._yexpr)
        return s

    __reversed__ = flip

    def shift(self, k):
        s = Signal.shift(self, k)
        s._yexpr = ShiftOperator.apply(s._xvar, s._yexpr, k)
        return s

    def scale(self, v):
        s = Signal.scale(self, v)
        s._yexpr = ScaleOperator.apply(s._xvar, s._yexpr, v)
        return s

    # -- operadores aritméticos ----------------------------------------------
    def __mul__(self, other):
        if other._dtype == np.complex_:
            self._dtype = np.complex_
        self._yexpr *= other._yexpr.xreplace({other._xvar: self._xvar})
        return self

    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, other):
        if other._dtype == np.complex_:
            self._dtype = np.complex_
        self._yexpr /= other._yexpr.xreplace({other._xvar: self._xvar})
        return self

    __itruediv__ = __truediv__

    def __rtruediv__(self, other):
        if self._dtype == np.complex_:
            other._dtype = np.complex_
        other._yexpr /= self._yexpr.xreplace({self._xvar: other._xvar})
        return 1/other

    def __add__(self, other):
        if other._dtype == np.complex_:
            self._dtype = np.complex_
        self._yexpr += other._yexpr.xreplace({other._xvar: self._xvar})
        return self

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        if other._dtype == np.complex_:
            self._dtype = np.complex_
        self._yexpr -= other._yexpr.xreplace({other._xvar: self._xvar})
        return self

    def __rsub__(self, other):
        if self._dtype == np.complex_:
            other._dtype = np.complex_
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
        # TODO: ¿es correcto? NO si las variables no son iguales
        return str(self).__eq__(str(other))
#         if isinstance(other, FunctionSignal):
#             return self._yexpr == other._yexpr
#         d = self._yexpr - other
#         if (sp.expand(d) == 0) or \
#            (sp.simplify(d) == 0) or \
#            (sp.trigsimp(d) == 0):
#             return True
#         else:
#             return False

    def dynamic_range(self, dB=False):
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


class ConstantSignal(FunctionSignal):

    def __init__(self, const, var):
        super().__init__(sp.sympify(const))
        if isinstance(const, complex):
            self._dtype = np.complex_
        self._xvar = var
        self._xexpr = var
        self._yexpr = sp.sympify(const)
