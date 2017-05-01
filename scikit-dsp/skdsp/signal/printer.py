from ._signal import _FunctionSignal
from .continuous import ContinuousFunctionSignal
from .discrete import DiscreteFunctionSignal, Exponential
from sympy.printing.latex import LatexPrinter
from sympy.printing.str import StrPrinter
import sympy as sp


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

    def _print__DiscreteDeltaTrain(self, e):
        return self._print__DiscreteDelta(e)

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

    def _print__DiscreteCosine(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr, True))
        return self._make_s(r'\cos', sv, e, True)

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
        if simple or isinstance(e, ContinuousFunctionSignal):
            s = pre + r'\left({}\right)'
        elif isinstance(e, DiscreteFunctionSignal):
            s = pre + r'\left[{}\right]'
        else:
            s = r'?{}'
        return s.format(var)

    def _print_Mod(self, e):
        return '(({0}))_{{{1}}}'.format(e.args[0], e.args[1])

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

    def _print_cos(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr, True))
        return self._make_s(r'\cos', sv, e, True)

    def _print_Sine(self, e):
        sv = r'{}'.format(self._make_var(e._xexpr, True))
        return self._make_s(r'\sin', sv, e, True)

    def _print_exp(self, e):
        sb = r'{{\rm{{e}}}}^{{{0}}}'
        se = ''
        exponent = sp.im(e.args[0])
        if exponent.is_negative:
            se = '-'
            exponent = -exponent
        else:
            se = r'\,'
        se += r'{\rm{j}}'
        s1 = self._print(exponent)
        if 'frac' in s1:
            se += s1
        else:
            se += r'(' + s1 + ')'
        s = sb.format(se)
        return s

    def _print_complex_exponent(self, expr, xexpr):
        sb = r'{{\rm{{e}}}}^{{{0}}}'
        se = ''
        exponent = sp.im(expr.args[0])
        if isinstance(xexpr, sp.Mul) and (xexpr.args[0] == -1):
            se = '-'
            xexpr = -xexpr
        else:
            se = r'\,'
        se += r'{\rm{j}}'
        s1 = self._print(exponent)
        if 'frac' in s1:
            se += s1
        else:
            se += r'(' + s1 + ')'
        se += self._print(xexpr)
        s = sb.format(se)
        return s

    def _print_Exponential(self, e):
        if isinstance(e._base, sp.exp):
            s = self._print_complex_exponent(e._base, e._xexpr)
        elif isinstance(e._base, sp.Mul):
            s = self._print(e._base.args[0])
            s += self._print_complex_exponent(e._base.args[1], e._xexpr)
        else:
            sv = r'{0}'.format(self._make_var(e._xexpr))
            if len(sv) != 1:
                sv = '{' + sv + '}'
            sb = self._print(sp.nsimplify(e._base))
            if e._base < 0 or 'frac' in sb:
                s = r'\left({0}\right)^{1}'.format(sb, sv)
            else:
                s = r'{0}^{1}'.format(sb, sv)
        return s

    def _print_Pow(self, e):
        exp = Exponential(e.args[0])
        exp._xexpr = e.args[1]
        exp._yexpr = e
        exp._xvar = e.free_symbols.pop()
        return self._print_Exponential(exp)

    def _print_ComplexSinusoid(self, e):
        s1 = self._print(e.phasor)
        s2 = self._print(e.carrier)
        if s1 == '1':
            s = s2
        else:
            s = s1 + s2
        return s

#     def _print_Add(self, expr, order=None):
#         # TODO or not
#         unordered_terms = list(expr.args)
#         terms = []
#         for term in unordered_terms:
#             terms.append(term)
#
#         tex = ""
#         for i, term in enumerate(terms):
#             if i == 0:
#                 pass
#             elif self._coeff_isneg(term):
#                 tex += " - "
#                 term = -term
#             else:
#                 tex += " + "
#             term_tex = self._print(term)
#             if self._needs_add_brackets(term):
#                 term_tex = r"\left(%s\right)" % term_tex
#             tex += term_tex
#
#         return tex


def latex(signal, **settings):
    toprint = r'{0}\left[{1}\right]'.format(signal.name, signal._xexpr)
    if isinstance(signal, _FunctionSignal):
        toprint = signal._yexpr
    clp = CustomLatexPrinter(settings)
    return clp.doprint(toprint)
