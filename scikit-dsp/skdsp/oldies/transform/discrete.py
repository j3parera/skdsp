from abc import ABC
import sympy as sp


class DiscreteTimeTransform(ABC):

    def __init__(self, var, expr):
        self._xvar = var
        self._xexpr = var
        self._yexpr = expr

    @property
    def xvar(self):
        return self._xvar

    @xvar.setter
    def xvar(self, newvar):
        self._xvar = newvar
        vmap = {self._xvar: newvar}
        self._xexpr.xreplace(vmap)
        self._yexpr.xreplace(vmap)

    @property
    def xexpr(self):
        return self._xexpr

    @property
    def yexpr(self):
        return self._yexpr

    def __eq__(self, other):
        oldvar = other._xvar
        other.xvar(self._xvar)
        equal = sp.simplify(self._yexpr - other._yexpr) == 0
        other.xvar(oldvar)
        return equal

    def apply(self, op):
        op.apply(self._xvar, self._xexpr)
        op.apply(self._xvar, self._yexpr)


class DTFT(DiscreteTimeTransform):

    default_var = sp.symbols('omega', real=True)

    def __init__(self, var, expr):
        super().__init__(var, expr)

    def __repr__(self):
        return self._yexpr.__repr__()


class ZTransform(DiscreteTimeTransform):

    default_var = sp.symbols('z', complex=True)

    def __init__(self, var, expr, roc):
        super().__init__(var, expr)
        self._roc = roc

    @property
    def roc(self):
        return self._roc

    def __eq__(self, other):
        return (super().__eq__(other)) & (self.roc == other.roc)

    def __repr__(self):
        return self._yexpr.__repr__() + ', ' + self._roc.__repr__()
