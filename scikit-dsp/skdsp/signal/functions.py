import inspect

import numpy as np
import sympy as sp
from sympy.core.logic import fuzzy_not

__all__ = [s for s in dir() if not s.startswith("_")]

# functions
# note: plot looks for a callable expression, it should be repr but uses str
class UnitDelta(sp.KroneckerDelta):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        # pylint: disable-msg=no-member
        if arg.is_zero:
            # pylint: enable-msg=no-member
            return sp.S.One
        if arg.is_negative or arg.is_positive:
            return sp.S.Zero
        if arg.is_Number and not arg.is_zero:
            return sp.S.Zero

    def __new__(cls, iv):
        obj = super().__new__(sp.KroneckerDelta, sp.S.Zero, iv)
        if isinstance(obj, sp.KroneckerDelta):
            obj.__class__ = UnitDelta
        return obj

    def _eval_rewrite_as_UnitStep(self, *args, **kwargs):
        return UnitStep(args[1]) - UnitStep(args[1] - 1)

    @property
    def func(self):
        stack = inspect.stack()
        for frame in stack:
            if frame.function == 'lambdify':
                return UnitDelta
        return sp.KroneckerDelta

    @staticmethod
    # zero is the first arg of the KroneckerDelta but not used
    def _imp_(_zero, n):
        return np.equal(n, 0).astype(np.float_)

    def __str__(self):
        return "UnitDelta({0})".format(self.args[1])

    def __repr__(self):
        return "\u03b4[{0}]".format(self.args[1])

    def _sympystr(self, printer=None):
        return r"UnitDelta({0})".format(printer.doprint(self.args[1]))

    def _latex(self, printer=None):
        return r"\delta\left[{0}\right]".format(printer.doprint(self.args[1]))


class UnitStep(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if arg.is_negative:
            return sp.S.Zero
        elif arg.is_nonnegative:
            return sp.S.One

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        return sp.Piecewise((1, args[0] >= 0), (0, True))

    def _eval_rewrite_as_UnitDelta(self, *args, **kwargs):
        k = sp.Dummy(integer=True)
        if "form" in kwargs and kwargs["form"] == "accum":
            return sp.Sum(UnitDelta(k), (k, sp.S.NegativeInfinity, args[0]))
        else:
            return sp.Sum(UnitDelta(args[0] - k), (k, 0, sp.S.Infinity))

    def _eval_rewrite_as_UnitRamp(self, *args, **kwargs):
        return UnitRamp(args[0] + 1) - UnitRamp(args[0])

    def _eval_difference_delta(self, var, step):
        # f[n+step] - f[n]
        # TODO será útil para differencias hacia atrás
        if var in self.free_symbols:
            if step == 0:
                return sp.S.Zero
            elif step == 1:
                return UnitDelta(self.args[0])
            elif step == -1:
                return -UnitDelta(self.args[0])
        return None

    @staticmethod
    def _imp_(n):
        return np.greater_equal(n, 0).astype(np.float_)

    def __str__(self):
        return "UnitStep({0})".format(self.args[0])

    def __repr__(self):
        return "u[{0}]".format(self.args[0])

    def _sympystr(self, printer=None):
        return r"UnitStep({0})".format(printer.doprint(self.args[0]))

    def _latex(self, printer=None):
        return r"u\left[{0}\right]".format(printer.doprint(self.args[0]))


class UnitRamp(sp.Function):

    nargs = 1
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if arg.is_negative:
            return sp.S.Zero
        elif arg.is_nonnegative:
            return arg

    def _eval_rewrite_as_Piecewise(self, arg):
        return sp.Piecewise((arg, arg >= 0), (0, True))

    def _eval_rewrite_as_UnitStep(self, *args, **kwargs):
        k = sp.Dummy(integer=True)
        if "form" in kwargs and kwargs["form"] == "accum":
            return sp.Sum(UnitStep(k - 1), (k, sp.S.NegativeInfinity, args[0]))
        else:
            return args[0] * UnitStep(args[0])

    def _eval_rewrite_as_Max(self, *args, **kwargs):
        return sp.Max(0, args[0])

    def _eval_rewrite_as_Abs(self, *args, **kwargs):
        return sp.S.Half * (args[0] + sp.Abs(args[0]))

    @staticmethod
    def _imp_(n):
        return n * np.greater_equal(n, 0).astype(np.float_)

    def __str__(self):
        return "UnitRamp({0})".format(self.args[0])

    def __repr__(self):
        return "r[{0}]".format(self.args[0])

    def _sympystr(self, printer=None):
        return r"UnitRamp({0})".format(printer.doprint(self.args[0]))

    def _latex(self, printer=None):
        return r"r\left[{0}\right]".format(printer.doprint(self.args[0]))


class UnitDeltaTrain(sp.Function):

    nargs = 2
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg, N):
        exp = sp.Mod(sp.sympify(arg), sp.sympify(N))
        # pylint: disable-msg=no-member
        if exp.is_zero:
            # pylint: enable-msg=no-member
            return sp.S.One
        if arg.is_negative or arg.is_positive:
            return sp.S.Zero
        if exp.is_Number and not exp.is_zero:
            return sp.S.Zero

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        nm = sp.Eq(sp.Mod(args[0], args[1]), 0)
        return sp.Piecewise((1, nm), (0, True))

    @staticmethod
    def _imp_(n, N):
        return np.equal(np.mod(n, N), 0).astype(np.float_)

    def __str__(self):
        return "UnitDeltaTrain({0}, {1})".format(self.args[0], self.args[1])

    def __repr__(self):
        return "\u0428[(({0})){1}]".format(self.args[0], self.args[1])

    def _sympystr(self, printer=None):
        return r"UnitDeltaTrain({0}, {1})".format(
            printer.doprint(self.args[0]), printer.doprint(self.args[1])
        )

    def _latex(self, printer=None):
        return r"{{\rotatebox[origin=c]{{180}}{{$\Pi\kern-0.361em\Pi$}}\left[(({0}))_{{{1}}}\right]".format(
            printer.doprint(self.args[0]), printer.doprint(self.args[1])
        )
