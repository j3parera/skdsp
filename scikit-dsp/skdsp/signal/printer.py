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
            return r'\delta\left[{0}\right]'.format(self._print(e.args[0]))
        elif isinstance(e, ContinuousFunctionSignal):
            return r'\delta\left[{0}\right]'.format(self._print(e._xexpr))
        return r'\delta[?\right]'

    def _print__DiscreteStep(self, e):
        if isinstance(e, sp.Expr):
            return r'u\left[{0}\right]'.format(self._print(e.args[0]))
        elif isinstance(e, ContinuousFunctionSignal):
            return r'u\left[{0}\right]'.format(self._print(e._xexpr))
        return r'u\left[?\right]'

    def _print__DiscreteRamp(self, e):
        if isinstance(e, sp.Expr):
            return r'r\left[{0}\right]'.format(self._print(e.args[0]))
        elif isinstance(e, DiscreteFunctionSignal):
            return r'r\left[{0}\right]'.format(self._print(e._xexpr))
        return r'r\left[?\right]'

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

    def _print_Delta(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr, True))
        return self._make_s(r'\delta', sv, e)

    def _print_Step(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr, True))
        return self._make_s(r'u', sv, e)

    def _print_Ramp(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr, True))
        return self._make_s(r'r', sv, e)

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

    def _print_complex_exponent(self, expr, sv):
        sb = r'{{\rm{{e}}}}^{{{0}}}'
        exponent = expr.args[0]
        if exponent.is_complex:
            exponent = sp.im(exponent)
            if exponent.is_negative:
                se = '-'
                exponent = -exponent
            else:
                se = r'\,'
            se += r'{{\rm{{j}}}}'
            s1 = self._print(exponent)
            if 'frac' in s1:
                se += s1
            else:
                se += r'(' + s1 + ')'
            se += sv
        s = sb.format(se)
        return s

    def _print_Exponential(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr))
        if isinstance(e._base, sp.exp):
            s = self._print_complex_exponent(e._base, sv)
        elif isinstance(e._base, sp.Mul):
            s = self._print(e._base.args[0])
            s += self._print_complex_exponent(e._base.args[1], sv)
        else:
            s = r'{0}^{1}'.format(e._base, sv)
        return s


def latex(signal, **settings):
    toprint = signal
    sc = signal.__class__
    if sc == DiscreteFunctionSignal or sc == ContinuousFunctionSignal:
        toprint = signal._yexpr
    return CustomLatexPrinter(settings).doprint(toprint)
