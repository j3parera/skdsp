from sympy.printing.str import StrPrinter
from sympy.printing.latex import LatexPrinter


class CustomStrPrinter(StrPrinter):

    def _print_ImaginaryUnit(self, expr):
        return 'j'

    def _print_Abs(self, e):
        return '|' + self._print(e.args[0]) + '|'

    def _print__DiscreteDelta(self, e):
        return 'd[' + self._print(e.args[0]) + ']'

    def _print__DiscreteStep(self, e):
        return 'u[' + self._print(e.args[0]) + ']'

    def _print__DiscreteRamp(self, e):
        return 'r[' + self._print(e.args[0]) + ']'


class CustomLatexPrinter(LatexPrinter):

    def _print_ImaginaryUnit(self, expr):
        return 'j'

    def _print_Delta(self, e):
        return r'\delta[' + self._print(e._xexpr) + ']'


def latex(expr, **settings):
    return CustomLatexPrinter(settings).doprint(expr)
