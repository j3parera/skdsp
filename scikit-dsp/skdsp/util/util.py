import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.functions.elementary.trigonometric import _pi_coeff
from sympy.plotting.plot import (
    Line2DBaseSeries,
    Plot,
    check_arguments,
    vectorized_lambdify,
)

__all__ = [s for s in dir() if not s.startswith("_")]


class StemOver1DRangeSeries(Line2DBaseSeries):
    def __init__(self, expr, var_start_end, **kwargs):
        super().__init__()
        self.expr = sp.sympify(expr)
        self.label = str(self.expr)
        self.var = sp.sympify(var_start_end[0])
        self.start = float(var_start_end[1])
        self.end = float(var_start_end[2])
        self.nb_of_points = self.end - self.start + 1
        self.adaptive = False
        self.depth = kwargs.get("depth", 12)
        self.line_color = kwargs.get("line_color", None)
        self.xscale = kwargs.get("xscale", "linear")

    def __str__(self):
        return "stem lines: %s for %s over %s" % (
            str(self.expr),
            str(self.var),
            str((self.start, self.end)),
        )

    def get_segments(self):
        x, y = self.get_points()
        stemlines = [((xi, 0), (xi, yi)) for xi, yi in zip(x, y)]
        return stemlines

    def get_points(self):
        if self.xscale == "log":
            list_x = np.logspace(
                int(self.start), int(self.end), num=int(self.end) - int(self.start) + 1
            )
        else:
            list_x = np.linspace(
                int(self.start), int(self.end), num=int(self.end) - int(self.start) + 1
            )
        f = vectorized_lambdify([self.var], self.expr)
        list_y = f(list_x)
        return (list_x, list_y)


def stem(*args, **kwargs):
    args = list(map(sp.sympify, args))
    free = set()
    for a in args:
        if isinstance(a, sp.Expr):
            free |= a.free_symbols
            if len(free) > 1:
                raise ValueError(
                    "The same variable should be used in all "
                    "univariate expressions being plotted."
                )
    x = free.pop() if free else sp.Symbol("x")
    # pylint: disable-msg=no-member
    kwargs.setdefault("xlabel", x.name)
    kwargs.setdefault("ylabel", "f(%s)" % x.name)
    # TODO continuar o no
    kwargs.setdefault("markers", [{"args": "o"}])
    # pylint: enable-msg=no-member
    show = kwargs.pop("show", True)
    series = []
    plot_expr = check_arguments(args, 1, 1)
    series = [StemOver1DRangeSeries(*arg, **kwargs) for arg in plot_expr]

    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots


def ipystem(
    nx, x, xlabel=None, title=None, axis=None, color="C0", marker="o", markersize=8, **kwargs
):
    # TODO
    # comprobar que no existan símbolos libres sin asignar
    offx = 0.5
    lines = plt.stem(
        nx,
        x,
        linefmt=color + "-",
        markerfmt=color + marker,
        basefmt=color + "-",
        use_line_collection=True,
    )
    lines[0].set_markersize(markersize)
    lines[2].set_xdata([nx[0] - offx, nx[-1] + offx])
    ax = plt.gca()
    if axis is None:
        mmin = np.min(x)
        mmax = np.max(x)
        dy = 0.1 * (mmax - mmin)
        ymin = min(-dy, mmin - dy) 
        ymax = max(dy, mmax + dy)
        if ymin == ymax:
            ymin = -0.5
            ymax = 0.5
        plt.axis(
            [np.min(nx) - offx, np.max(nx) + offx, ymin, ymax]
        )
    else:
        plt.axis(axis)
    if title is not None:
        plt.title(title, size=20, pad=8, loc="left")
    if xlabel is not None:
        plt.xlabel(xlabel, size=20, labelpad=0, ha="left")
        ax.xaxis.set_label_coords(1.02, 0.06)
    plt.grid(True)
    plt.xticks(range(nx[0], nx[-1]+1, max(1, (nx[-1]-nx[0])//10)))
    # plt.tight_layout()
    return ax


def as_coeff_polar(z):
    if z.is_Add:
        return (sp.Abs(z), sp.arg(z))
    re, im = z.as_real_imag()
    if (re != sp.S.Zero and im == sp.S.Zero) or (re == sp.S.Zero and im != sp.S.Zero):
        # pure real or pure imag
        return (sp.Abs(z), sp.arg(z))
    # (+-)r*exp(sp.I*phi)
    r = sp.Wild("r")
    om = sp.Wild("om")
    phi = sp.Wild("phi")
    d = z.match(r * sp.exp(sp.I * om + phi))
    if d is not None:
        if d[r] == sp.S.Zero:
            return (sp.S.Zero, None)
        else:
            return (sp.Abs(d[r]) * sp.exp(d[phi]), sp.arg(d[r]) + d[om])
    return (sp.Abs(z), sp.arg(z))


class Constraint(object):

    def __init__(self, symbol, constraint):
        self._symbol = sp.S(symbol)
        if not isinstance(self._symbol, sp.Symbol):
            raise TypeError("The parameter symbol must be a symbol.")
        self._constraint = sp.S(constraint)
        if isinstance(self._constraint, sp.Interval):
            d = sp.Dummy('d', real=True, positive=True)
            new = (self._constraint.start * d + self._constraint.measure) / (1 + d) 
            self._replace_expr = new
            self._replace_symbol = d
            self._replace_undo = (symbol - self._constraint.end) / (self._constraint.start - symbol)
        else:
            raise NotImplementedError
        
    @property
    def symbol(self):
        return self._symbol
    
    @property
    def constraint(self):
        return self._constraint
        
    @property
    def replace_expr(self):
        return self._replace_expr

    @property
    def replace_symbol(self):
        return self._replace_symbol

    @property
    def replace_undo_expr(self):
        return self._replace_undo

    def apply(self, expr):
        if isinstance(expr, sp.Piecewise):
            for pair in expr.args:
                sol = sp.solveset(pair.cond)
                if self.constraint.intersect(sol) == sp.EmptySet:
                    expr = expr.xreplace({pair.cond: False})
        if expr.has(self.symbol):
            expr = expr.xreplace({self.symbol: self.replace_expr})
        return expr

    def revert(self, expr):
        if expr.has(self.replace_symbol):
            expr = expr.xreplace({self.replace_symbol: self.replace_undo_expr})
        return expr

