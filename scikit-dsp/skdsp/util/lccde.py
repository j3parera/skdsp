from collections import defaultdict

import sympy as sp
from sympy.solvers.ode import get_numbered_constants, iter_numbered_constants

from skdsp.signal.functions import UnitDelta
from skdsp.signal.discrete import DiscreteSignal


class LCCDE(sp.Basic):
    @classmethod
    def from_expression(cls, f, x, y):
        if isinstance(f, sp.Equality):
            f = f.lhs - f.rhs
        n = y.args[0]
        k = sp.Wild("k", exclude=(n,))
        h_part = defaultdict(lambda: sp.S.Zero)
        i_part = defaultdict(lambda: sp.S.Zero)
        for g in sp.Add.make_args(f):
            coeff = sp.S.One
            kspec = None
            for h in sp.Mul.make_args(g):
                if h.is_Function:
                    if h.func == y.func:
                        result = h.args[0].match(n + k)
                        if result is not None:
                            kspec = int(result[k])
                            if kspec is not None:
                                h_part[kspec] += coeff
                        else:
                            raise ValueError(
                                "'%s(%s + k)' expected, got '%s'" % (y.func, n, h)
                            )
                    elif h.func == x.func:
                        result = h.args[0].match(n + k)
                        if result is not None:
                            kspec = int(result[k])
                            if kspec is not None:
                                i_part[kspec] += -coeff
                        else:
                            raise ValueError(
                                "'%s(%s + k)' expected, got '%s'" % (x.func, n, h)
                            )
                    else:
                        if not h.is_constant(n):
                            raise ValueError(
                                "Non constant coefficient found: {0}".format(h)
                            )
                else:
                    if not h.is_constant(n):
                        raise ValueError(
                            "Non constant coefficient found: {0}".format(h)
                        )
                    coeff *= h
        if h_part:
            if max(h_part.keys()) > 0:
                raise ValueError("Equation is not recursive.")
            K_min = min(h_part.keys())
            A = [sp.simplify(h_part[i]) for i in range(0, K_min - 1, -1)]
        else:
            A = [sp.S.One]
        if i_part:
            if max(i_part.keys()) > 0:
                raise ValueError("Equation is not recursive.")
            K_min = min(i_part.keys())
            B = [sp.simplify(i_part[i]) for i in range(0, K_min - 1, -1)]
        else:
            B = [sp.S.One]
        obj = LCCDE.__new__(cls, B, A)
        return obj

    def __new__(cls, B=[], A=[], x=None, y=None):
        n = sp.Symbol("n", integer=True)
        if x is None:
            x = sp.Function("x")(n)
        if y is None:
            y = sp.Function("y")(n)
        if x.args[0] != y.args[0]:
            raise ValueError("Independent variable of x[n] and y[n] must be the same.") 
        B = [sp.S(b) / sp.S(A[0]) for b in B]
        A = [sp.S(a) / sp.S(A[0]) for a in A]
        obj = sp.Basic.__new__(cls, B, A, x, y)
        return obj

    @property
    def x(self):
        return self.args[2]

    @property
    def y(self):
        return self.args[3]

    @property
    def iv(self):
        return self.y.args[0]

    @property
    def B(self):
        return self.args[0]

    @property
    def A(self):
        return self.args[1]

    @property
    def order(self):
        return len(self.A) - 1

    @property
    def M(self):
        return len(self.B) - 1

    @property
    def N(self):
        return self.order

    def x_part(self, offset=0):
        n = self.iv
        x = self.x
        adds = [c * x.subs(n, n - k + offset) for k, c in enumerate(self.B)]
        return sp.Add(*adds)

    def y_recursive_part(self, forward=True, offset=0):
        n = self.iv
        y = self.y
        if forward:
            adds = [
                a * y.subs(n, n - (k + 1) + offset) for k, a in enumerate(self.A[1:])
            ]
        else:
            adds = [a * y.subs(n, n - k + offset) for k, a in enumerate(self.A[:-1])]
        return sp.Add(*adds)

    def y_part(self, offset=0):
        return self.y + self.y_recursive_part(offset=offset)

    @property
    def as_expression(self):
        return sp.Eq(self.y_part(), self.x_part())

    def apply_input(self, fin=None, offset=0):
        x_part = self.x_part()
        if fin is not None:
            n = self.iv
            for k in range(self.M + 1):
                old = self.x.subs(n, n - k + offset)
                new = fin.subs(n, n - k + offset)
                x_part = x_part.subs(old, new)
        return x_part

    def as_forward_recursion(self, fin=None):
        return -self.y_recursive_part() + self.apply_input(fin)

    def as_backward_recursion(self, fin=None):
        offset = self.N
        return (
            -self.y_recursive_part(forward=False, offset=offset)
            + self.apply_input(fin, offset=offset)
        ) / self.A[-1]

    def solve_homogeneous(self):
        n = self.iv
        # Characteristic polynomial
        chareq, symbol = sp.S.Zero, sp.Dummy("lambda")
        for k, a in enumerate(reversed(self.A)):
            chareq += a * symbol ** k
        chareq = sp.Poly(chareq, symbol)
        chareqroots = sp.roots(chareq, symbol)
        if len(chareqroots) != self.order:
            chareqroots = [sp.rootof(chareq, k) for k in range(chareq.degree())]
        chareq_is_complex = not all([i.is_real for i in chareq.all_coeffs()])
        # Create a dict root: multiplicity or charroots
        charroots = defaultdict(int)
        for root in chareqroots:
            charroots[root] += 1
        # generate solutions
        gensols = []
        conjugate_roots = []  # used to prevent double-use of conjugate roots
        # Loop over roots in the order provided by roots/rootof...
        for root in chareqroots:
            # but don't repeat multiple roots.
            if root not in charroots:
                continue
            multiplicity = charroots.pop(root)
            for i in range(multiplicity):
                if chareq_is_complex:
                    gensols.append(n ** i * root ** n)
                    continue
                absroot = sp.Abs(root)
                angleroot = sp.arg(root)
                if angleroot.has(sp.atan2):
                    # Remove this condition when arg stop returning circular atan2 usages.
                    gensols.append(n ** i * root ** n)
                else:
                    if root in conjugate_roots:
                        continue
                    if angleroot == 0:
                        gensols.append(n ** i * absroot ** n)
                        continue
                    conjugate_roots.append(sp.conjugate(root))
                    gensols.append(n ** i * absroot ** n * sp.cos(angleroot * n))
                    gensols.append(n ** i * absroot ** n * sp.sin(angleroot * n))
        # Constants
        const_gen = iter_numbered_constants(chareq, start=1, prefix="C")
        consts = [next(const_gen) for i in range(len(gensols))]
        yh = sp.Add(*[c * g for c, g in zip(consts, gensols)])
        return yh, consts[: self.order]

    def solve_forced(self, fin, ac):
        yh, consts = self.solve_homogeneous()
        # solution parts: extract solutions up to N < M
        partial_sols = []
        n = self.iv
        y = self.y
        x = self.x
        N = self.N
        lccde = self.copy()
        while lccde.M >= N:
            B = lccde.B
            xM = x.subs(n, n - lccde.M + 1)
            yk = B[-1] / self.A[-1] * xM
            partial_sols.append(yk.subs(xM, fin.subs(n, n - lccde.M + 1)))
            del B[-1]
            new_x_part = sp.Add(*[bk * x.subs(n, n - k) for k, bk in enumerate(B)]) - yk
            lccde = LCCDE.from_expression(
                sp.Eq(self.y_part(), new_x_part), self.x, self.y
            )
        #
        if isinstance(ac, dict):
            if len(ac.keys()) != self.order:
                raise ValueError(
                    "The number of auxiliary conditions is not equal to the LCDDE order."
                )
        elif isinstance(ac, str) and ac == "initial_rest":
            ac = dict(zip(range(-1, -lccde.order - 1, -1), [0] * lccde.order))
            yexpr = lccde.as_forward_recursion(fin)
        elif isinstance(ac, str) and ac == "final_rest":
            ac = dict(zip(range(0, lccde.order), [0] * lccde.order))
            yexpr = lccde.as_backward_recursion(fin)
        else:
            # TODO
            raise ValueError("Not valid auxiliary conditions.")
        # equations over non vanishing inputs
        xsup = DiscreteSignal.from_formula(lccde.apply_input(fin), iv=n, codomain=None).support
        inf = xsup.inf
        if inf != sp.S.NegativeInfinity:
            nvals = list(range(inf, inf + len(consts)))
        else:
            sup = xsup.sup
            if sup == sp.S.Infinity:
                nvals = range(len(consts))
            else:
                nvals = list(range(sup, sup - len(consts), -1))
        #
        lhss = [yh.subs(n, k) for k in nvals]
        rhss = [yexpr.subs(n, nvals[0])]
        for k in nvals[1:]:
            yk = yexpr.subs(n, k)
            yk = yk.subs(y.subs(n, k - 1), rhss[k - 1])
            rhss.append(yk)
        # apply auxiliary conditions
        for k1, rhs in enumerate(rhss):
            for k2, v in ac.items():
                rhs = rhs.subs(y.subs(n, k2), v)
            rhss[k1] = rhs
        eqs = [sp.Eq(lhs, rhs) for lhs, rhs in zip(lhss, rhss)]
        sol = sp.solve(eqs, consts)
        y = yh.subs(sol)
        y += sp.Add(*partial_sols)
        return y
