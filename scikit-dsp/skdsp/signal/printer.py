import re
import inspect
from sympy.printing.latex import LatexPrinter
import skdsp.signal.signal


class SignalLatexPrinter(LatexPrinter):
    def print_signal(self, sg):
        ltx = super().doprint(sg.amplitude)
        return self.apply_rules(ltx, sg.iv)

    def print_expr(self, expr, iv):
        ltx = super().doprint(expr)
        return self.apply_rules(ltx, iv)

    def apply_rules(self, ltx, iv):
        rules = inspect.getmembers(self, inspect.ismethod)
        for rule in rules:
            if rule[0].startswith("rule"):
                ltx = rule[1](ltx, iv)
        return ltx

    def rule_iv_in_frac(self, ltx, iv):
        # extract iv from frac
        pat = r"(.*?)?(\\frac{)(.*)(" + "{0}".format(iv.name) + ")(}{.+?})(.*)?"
        fr = re.compile(pat)
        m = fr.match(ltx)
        if m is not None:
            s = (
                "".join(m.group(1, 2))
                + ("1" if m.group(3) == "" else m.group(3))
                + "".join(m.group(5, 4, 6))
            )
            return s
        return ltx

    def rule_j_in_frac(self, ltx, iv):
        pat = r"(.*?)?(\\frac{)(\\mathrm{j}*)(.*)(}{.+?})(.*)?"
        fr = re.compile(pat)
        m = fr.match(ltx)
        if m is not None:
            return "".join(m.group(1, 3, 2, 4, 5, 6))
        return ltx

    def rule_exp(self, ltx, iv):
        pat = r"(.*?)?(e\^{)(.*)"
        fr = re.compile(pat)
        m = fr.match(ltx)
        if m is not None:
            s = "".join(m.group(1)) + "\mathrm{e}^{\," + "".join(m.group(3))
            return s
        return ltx


def latex(expr, iv):
    printer = SignalLatexPrinter(
        settings={
            'imaginary_unit': 'rj',
            'fold_frac_powers': True,
            'order': 'lex',
            'gothic_re_im': True,
        }
    )
    return printer.print_expr(expr, iv)
