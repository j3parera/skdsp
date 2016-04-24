from sympy import Basic
from skdsp.signal.printer import CustomStrPrinter

Basic.__str__ = lambda self: CustomStrPrinter().doprint(self)
