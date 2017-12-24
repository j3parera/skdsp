from sympy import Basic
from .printer import CustomStrPrinter

Basic.__str__ = lambda self: CustomStrPrinter().doprint(self)
