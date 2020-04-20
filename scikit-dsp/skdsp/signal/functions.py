import inspect

import numpy as np
import sympy as sp

__all__ = [s for s in dir() if not s.startswith("_")]

# functions
# note: plot looks for a callable expression, it should be repr but uses str
class UnitDelta(sp.Function):

    nargs = 1
    is_finite = True
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, arg):
        arg = sp.sympify(arg)
        if arg.is_zero:
            return sp.S.One
        if arg.is_negative or arg.is_positive:
            return sp.S.Zero
        if arg.is_Number and not arg.is_zero:
            return sp.S.Zero

    def _eval_Abs(self):
        return self

    def _eval_power(self, other):
        # other is not Nan, 0 or 1
        return self

    def _eval_subs(self, old, new):
        if self.func_arg.has(old):
            return UnitDelta(self.func_arg.xreplace({old: new}))

    def _eval_rewrite_as_UnitStep(self, *_args, **_kwargs):
        return UnitStep(self.func_arg) - UnitStep(self.func_arg - 1)

    def _eval_rewrite_as_Piecewise(self, *_args, **_kwargs):
        return sp.Piecewise((1, sp.Eq(self.func_arg, 0)), (0, True))

    @property
    @sp.cacheit
    def func_arg(self):
        return self.args[0]

    @staticmethod
    def _imp_(n):
        return np.equal(n, 0).astype(np.float_)

    def doit(self, **kwargs):
        # callers = ["Sum"]
        # frame = inspect.currentframe().f_back
        # try:
        #     caller = frame.f_locals['self']
        # except KeyError:
        #     # not a class but a function
        #     return self
        # if caller.__class__.__name__ in callers:
        sum = kwargs.pop("sum", None)
        if sum:
            # disguise as sympy KroneckerDelta for summations
            return sp.KroneckerDelta(sp.S.Zero, self.func_arg)
        return self

    def __str__(self):
        return "UnitDelta({0})".format(self.func_arg)

    def __repr__(self):
        return "\u03b4[{0}]".format(self.func_arg)

    def _sympystr(self, printer=None):
        return r"UnitDelta({0})".format(printer.doprint(self.func_arg))

    def _latex(self, printer=None):
        return r"\delta\left[{0}\right]".format(printer.doprint(self.func_arg))

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

    def _eval_Abs(self):
        return self

    def _eval_power(self, other):
        # other is not Nan, 0 or 1
        return self

    @property
    def func_arg(self):
        return self.args[0]

    def _eval_rewrite_as_Piecewise(self, *_args, **_kwargs):
        return sp.Piecewise((1, self.func_arg >= 0), (0, True))

    def _eval_rewrite_as_UnitDelta(self, *_args, **kwargs):
        k = sp.Dummy(integer=True)
        if kwargs and kwargs.get("form", "None") == "accum":
            return sp.Sum(UnitDelta(k), (k, sp.S.NegativeInfinity, self.func_arg))
        return sp.Sum(UnitDelta(self.func_arg - k), (k, 0, sp.S.Infinity))

    def _eval_rewrite_as_UnitRamp(self, *_args, **_kwargs):
        return UnitRamp(self.func_arg + 1) - UnitRamp(self.func_arg)

    def _eval_difference_delta(self, var, step):
        # f[n+step] - f[n]
        # TODO será útil para differencias hacia atrás
        if var in self.free_symbols:
            if step == 0:
                return sp.S.Zero
            elif step == 1:
                return UnitDelta(self.func_arg)
            elif step == -1:
                return -UnitDelta(self.func_arg)
        return None

    @staticmethod
    def _imp_(n):
        return np.greater_equal(n, 0).astype(np.float_)

    def __str__(self):
        return "UnitStep({0})".format(self.func_arg)

    def __repr__(self):
        return "u[{0}]".format(self.func_arg)

    def _sympystr(self, printer=None):
        return r"UnitStep({0})".format(printer.doprint(self.func_arg))

    def _latex(self, printer=None):
        return r"u\left[{0}\right]".format(printer.doprint(self.func_arg))

    def doit(self, *args, **kwargs):
        if kwargs and kwargs.get("piecewise", False):
            return self._eval_rewrite_as_Piecewise()
        return self


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

    def _eval_Abs(self):
        return self

    def _eval_power(self, other):
        # other is not Nan, 0 or 1
        return self.func_arg ** other * UnitStep(self.func_arg)

    @property
    def func_arg(self):
        return self.args[0]

    def _eval_rewrite_as_Piecewise(self, *_args, **_kwargs):
        # return sp.Piecewise((self.func_arg, self.func_arg >= 0), (0, True))
        return sp.Piecewise((self.func_arg, self.func_arg > 0), (0, True))

    def _eval_rewrite_as_UnitStep(self, *_args, **kwargs):
        k = sp.Dummy(integer=True)
        if "form" in kwargs and kwargs["form"] == "accum":
            return sp.Sum(UnitStep(k - 1), (k, sp.S.NegativeInfinity, self.func_arg))
        else:
            return self.func_arg * UnitStep(self.func_arg)

    def _eval_rewrite_as_Max(self, *_args, **_kwargs):
        return sp.Max(0, self.func_arg)

    def _eval_rewrite_as_Abs(self, *_args, **_kwargs):
        return sp.S.Half * (self.func_arg + sp.Abs(self.func_arg))

    @staticmethod
    def _imp_(n):
        return n * np.greater_equal(n, 0).astype(np.float_)

    def __str__(self):
        return "UnitRamp({0})".format(self.func_arg)

    def __repr__(self):
        return "r[{0}]".format(self.func_arg)

    def _sympystr(self, printer=None):
        return r"UnitRamp({0})".format(printer.doprint(self.func_arg))

    def _latex(self, printer=None):
        return r"r\left[{0}\right]".format(printer.doprint(self.func_arg))


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

    @property
    def func_arg(self):
        return self.args[0]

    @property
    def N(self):
        return self.args[1]

    def _eval_rewrite_as_Piecewise(self, *_args, **_kwargs):
        nm = sp.Eq(sp.Mod(self.func_arg, self.N), 0)
        return sp.Piecewise((1, nm), (0, True))

    @staticmethod
    def _imp_(n, N):
        return np.equal(np.mod(n, N), 0).astype(np.float_)

    def __str__(self):
        return "UnitDeltaTrain({0}, {1})".format(self.func_arg, self.N)

    def __repr__(self):
        return "\u0428[(({0})){1}]".format(self.func_arg, self.N)

    def _sympystr(self, printer=None):
        return r"UnitDeltaTrain({0}, {1})".format(
            printer.doprint(self.func_arg), printer.doprint(self.N)
        )

    def _latex(self, printer=None):
        return r"{{\rotatebox[origin=c]{{180}}{{$\Pi\kern-0.361em\Pi$}}\left[(({0}))_{{{1}}}\right]".format(
            printer.doprint(self.func_arg), printer.doprint(self.N)
        )


def stepsimp(expr):
    if isinstance(expr, sp.Mul):
        cnt = len(expr.atoms(UnitStep))
        if cnt >= 2:
            expr = sp.simplify(expr.rewrite(sp.Piecewise))
            if isinstance(expr, sp.Piecewise):
                if expr.args[1] == (0, True):
                    A = sp.Wild("A")
                    s = sp.Wild("s")
                    k = sp.Wild("k")
                    l = sp.Wild("l")
                    m = expr.args[0].match((A, s >= k))
                    if m:
                        return m[A] * UnitStep(m[s] - m[k])
                    m = expr.args[0].match((A, s <= k))
                    if m:
                        return m[A] * UnitStep(-m[s] + m[k])
                    m = expr.args[0].match((A, (s >= k) & (s <= l)))
                    if m:
                        e = m[A] * (UnitStep(m[s] - m[k]) - UnitStep(m[s] - (m[l] + 1)))
                        return sp.Piecewise((e, m[l] >= 0), (0, True))
    else:
        e = sp.Wild("e")
        iv = sp.Wild("iv")
        k = sp.Wild("k")
        m = expr.match(sp.Piecewise((e, iv >= k), (0, True)))
        if m:
            expr = m[e] * UnitStep(m[iv] - m[k])
            return expr
        m = expr.match(sp.Piecewise((0, iv < k), (e, True)))
        if m:
            expr = m[e] * UnitStep(m[iv] - m[k])
            return expr
    return expr


def deltasimp(expr, iv):
    # prefer d[n-1] to d[-n+1]
    deltas = expr.atoms(UnitDelta)
    for d in deltas:
        arg = d.func_arg
        if not arg.has(sp.Mod):
            k = list(sp.solveset(arg, iv, sp.S.Integers))[0]
            expr = expr.replace(d, UnitDelta(iv - k))
    kdeltas = expr.atoms(sp.KroneckerDelta)
    for d in kdeltas:
        arg = d.args[1]
        if not arg.has(sp.Mod):
            k = list(sp.solveset(arg, iv, sp.S.Integers))[0]
            expr = expr.replace(d, UnitDelta(iv - k))
    return expr
