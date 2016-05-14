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

    def _print_Rect(self, e):
        s = r'\Pi\left{0}{1}\right{2}'
        s1 = r'{0}/{1}'
        if isinstance(e, DiscreteFunctionSignal):
            s0 = '['
            s2 = ']'
        elif isinstance(e, ContinuousFunctionSignal):
            s0 = '('
            s2 = ')'
        else:
            return '?'
        if isinstance(e._xexpr, sp.Symbol):
            s1 = s1.format(self._print(e._xexpr), self._print(e._width))
        else:
            s1 = s1.format('(' + self._print(e._xexpr) + ')',
                           self._print(e._width))
        return s.format(s0, s1, s2)

    def _print_Triang(self, e):
        s = r'\Delta\left{0}{1}\right{2}'
        s1 = r'{0}/{1}'
        if isinstance(e, DiscreteFunctionSignal):
            s0 = '['
            s2 = ']'
        elif isinstance(e, ContinuousFunctionSignal):
            s0 = '('
            s2 = ')'
        else:
            return '?'
        if isinstance(e._xexpr, sp.Symbol):
            s1 = s1.format(self._print(e._xexpr), self._print(e._width))
        else:
            s1 = s1.format('(' + self._print(e._xexpr) + ')',
                           self._print(e._width))
        return s.format(s0, s1, s2)


def latex(signal, **settings):
    return CustomLatexPrinter(settings).doprint(signal)
