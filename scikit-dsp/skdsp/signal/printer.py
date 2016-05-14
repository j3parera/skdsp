from sympy.printing.str import StrPrinter
from sympy.printing.latex import LatexPrinter
from skdsp.signal.continuous import ContinuousFunctionSignal
import sympy as sp
from skdsp.signal.discrete import DiscreteFunctionSignal


class CustomStrPrinter(StrPrinter):

    def _print_ImaginaryUnit(self, expr):
        return 'j'

    def _print_Abs(self, e):
        return '|' + self._print(e.args[0]) + '|'


class CustomLatexPrinter(LatexPrinter):

    def _print_ImaginaryUnit(self, expr):
        return 'j'

    def _print__DiscreteDelta(self, e):
        if isinstance(e, sp.Expr):
            return r'\delta[{0}]'.format(self._print(e.args[0]))
        elif isinstance(e, ContinuousFunctionSignal):
            return r'\delta[{0}]'.format(self._print(e._xexpr))
        return r'\delta[?]'

    def _print__DiscreteStep(self, e):
        if isinstance(e, sp.Expr):
            return r'u[{0}]'.format(self._print(e.args[0]))
        elif isinstance(e, ContinuousFunctionSignal):
            return r'u[{0}]'.format(self._print(e._xexpr))
        return r'u[?]'

    def _print__DiscreteRamp(self, e):
        if isinstance(e, sp.Expr):
            return r'r[{0}]'.format(self._print(e.args[0]))
        elif isinstance(e, ContinuousFunctionSignal):
            return r'r[{0}]'.format(self._print(e._xexpr))
        return r'u[?]'

    def _print__ContinuousRamp(self, e):
        if isinstance(e, sp.Expr):
            return r'r\left({0}\right)'.format(self._print(e.args[0]))
        elif isinstance(e, ContinuousFunctionSignal):
            return r'r\left({0}\right)'.format(self._print(e._xexpr))
        return r'r\left(?\right)'

    def _print_Heaviside(self, e):
        if isinstance(e, sp.Expr):
            return r'u\left({0}\right)'.format(self._print(e.args[0]))
        elif isinstance(e, ContinuousFunctionSignal):
            return r'u\left({0}\right)'.format(self._print(e._xexpr))
        return r'u(?)'

    def _make_var(self, expr, simple=False):
        if simple or isinstance(expr, sp.Symbol):
            s1 = self._print(expr)
        else:
            s1 = '(' + self._print(expr) + ')'
        return s1

    def _make_s(self, pre, var, e, simple=False):
        if isinstance(e, DiscreteFunctionSignal):
            s = pre + r'\left[{}\right]'
        elif isinstance(e, ContinuousFunctionSignal):
            s = pre + r'\left({}\right)'
        else:
            s = r'?{}'
        return s.format(var)

    def _print_Rect(self, e):
        sv = r'{}/{}'.format(self._make_var(e._xexpr),
                             self._print(e._width))
        return self._make_s(r'\Pi', sv, e)

    def _print_Triang(self, e):
        sv = r'{}/{}'.format(self._make_var(e._xexpr),
                             self._print(e._width))
        return self._make_s(r'\Delta', sv, e)

    def _print_Cosine(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr, True))
        return self._make_s(r'\cos', sv, e)

    def _print_Sine(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr, True))
        return self._make_s(r'\sin', sv, e)


def latex(signal, **settings):
    return CustomLatexPrinter(settings).doprint(signal)
