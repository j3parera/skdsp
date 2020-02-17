import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from sympy.plotting.plot import Line2DBaseSeries, Plot, check_arguments, vectorized_lambdify

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
        self.depth = kwargs.get('depth', 12)
        self.line_color = kwargs.get('line_color', None)
        self.xscale = kwargs.get('xscale', 'linear')

    def __str__(self):
        return 'stem lines: %s for %s over %s' % (
            str(self.expr), str(self.var), str((self.start, self.end)))

    def get_segments(self):
        x, y = self.get_points()
        stemlines = [((xi, 0), (xi, yi)) for xi, yi in zip(x, y)]
        return stemlines

    def get_points(self):
        if self.xscale == 'log':
            list_x = np.logspace(int(self.start), int(self.end), num=int(self.end) - int(self.start) + 1)
        else:
            list_x = np.linspace(int(self.start), int(self.end), num=int(self.end) - int(self.start) + 1)
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
                    'The same variable should be used in all '
                    'univariate expressions being plotted.')
    x = free.pop() if free else sp.Symbol('x')
    # pylint: disable-msg=no-member
    kwargs.setdefault('xlabel', x.name)
    kwargs.setdefault('ylabel', 'f(%s)' % x.name)
    # TODO continuar o no
    kwargs.setdefault('markers', [{'args': 'o'}])
    # pylint: enable-msg=no-member
    show = kwargs.pop('show', True)
    series = []
    plot_expr = check_arguments(args, 1, 1)
    series = [StemOver1DRangeSeries(*arg, **kwargs) for arg in plot_expr]

    plots = Plot(*series, **kwargs)
    if show:
        plots.show()
    return plots

def ipystem(
    nx, x, xlabel=None, title=None, axis=None, color="k", marker="o", markersize=8
):
    offx = 0.5
    offy = 0.5
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
        plt.axis(
            [np.min(nx) - offx, np.max(nx) + offx, np.min(x) - offy, np.max(x) + offy]
        )
    else:
        plt.axis(axis)
    if title is not None:
        plt.title(title, size=20, pad=8, loc="left")
    if xlabel is not None:
        plt.xlabel(xlabel, size=20, labelpad=0, ha="left")
        ax.xaxis.set_label_coords(1.02, 0.06)
    plt.grid(True)
    # plt.tight_layout()
    return ax

