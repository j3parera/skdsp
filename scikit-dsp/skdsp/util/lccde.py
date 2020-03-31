from collections import defaultdict
from itertools import product
from numbers import Number

import sympy as sp
from sympy.solvers.ode import get_numbered_constants, iter_numbered_constants

from skdsp.signal.discrete import DiscreteSignal
from skdsp.signal.functions import UnitDelta, UnitRamp, UnitStep, stepsimp


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
                        # is part of the coeff
                        coeff *= h
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

    def x(self, value=None):
        if value is None:
            return self.args[2]
        return self.args[2].subs(self.iv, value)

    def y(self, value=None):
        if value is None:
            return self.args[3]
        return self.args[3].subs(self.iv, value)

    @property
    def iv(self):
        return self.y().args[0]

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
        adds = [c * self.x(n - k + offset) for k, c in enumerate(self.B)]
        return sp.Add(*adds)

    def y_recursive_part(self, forward=True, offset=0):
        n = self.iv
        if forward:
            adds = [a * self.y(n - (k + 1) + offset) for k, a in enumerate(self.A[1:])]
        else:
            adds = [a * self.y(n - k + offset) for k, a in enumerate(self.A[:-1])]
        return sp.Add(*adds)

    def y_part(self, offset=0):
        return self.y() + self.y_recursive_part(offset=offset)

    @property
    def as_expression(self):
        return sp.Eq(self.y_part(), self.x_part())

    def check_solution(self, fin, fout):
        # Caution. not 100% reliable
        x_sol = self.apply_input(fin).expand()
        y_sol = self.apply_output(fout).expand()
        dif = stepsimp(y_sol - x_sol)
        if sp.simplify(dif) == sp.S.Zero:
            return True
        # try some points
        n = self.iv
        for n0 in range(-100, 101):
            if sp.simplify(x_sol.subs(n, n0) - y_sol.subs(n, n0)) != sp.S.Zero:
                return False
        return True

    def apply_input(self, fin=None, offset=0):
        x_part = self.x_part()
        if offset != 0 or fin is not None:
            n = self.iv
            for k in range(self.M + 1):
                old = self.x(n - k)
                if fin is not None:
                    new = fin.subs(n, n - k + offset)
                else:
                    new = self.x(n - k + offset)
                x_part = x_part.subs(old, new)
        return x_part

    def apply_output(self, fout=None, offset=0):
        y_part = self.y_part()
        if offset != 0 or fout is not None:
            n = self.iv
            for k in range(self.N + 1):
                old = self.y(n - k)
                if fout is not None:
                    new = fout.subs(n, n - k + offset)
                else:
                    new = self.y(n - k + offset)
                y_part = y_part.subs(old, new)
        return y_part

    def as_forward_recursion(self, fin=None):
        return -self.y_recursive_part() + self.apply_input(fin)

    def as_backward_recursion(self, fin=None):
        offset = self.N
        return (
            -self.y_recursive_part(forward=False, offset=offset)
            + self.apply_input(fin, offset=offset)
        ) / self.A[-1]

    def _recur(self, eq, span, forward):
        offset = -1 if forward else 1
        result = dict()
        for k in range(span.start + offset * self.order, span.start, -offset):
            result[k] = self.y(k)
        for k in span:
            yk = eq.subs(self.iv, k)
            for r in range(self.order):
                idx = k + (r + 1) * offset
                yk = yk.subs(self.y(idx), result[idx])
            result[k] = sp.expand_mul(yk)
        return result

    def forward_recur(self, span, fin=None):
        y = self.as_forward_recursion(fin)
        return self._recur(y, span, True)

    def backward_recur(self, span, fin=None):
        y = self.as_backward_recursion(fin)
        return self._recur(y, span, False)

    def _process_aux_conditions(self, ac="initial_rest"):
        n = self.iv
        N = self.order
        if isinstance(ac, dict):
            if len(ac.keys()) != N:
                raise ValueError(
                    "The number of auxiliary conditions is not equal to the LCDDE order."
                )
            if all(isinstance(k, Number) for k in ac.keys()):
                if sorted(ac.keys()) != list(
                    range(min(ac.keys()), max(ac.keys()) + 1)
                ):
                    raise ValueError("Auxiliary conditions must be contiguous.")
        elif isinstance(ac, str):
            if str in ["initial_rest", "causal"]:
                ac = dict(zip(range(-1, -N - 1, -1), [0] * N))
            if str in ["final_rest", "anticausal"]:
                ac = dict(zip(range(0, self.order), [0] * self.order))
        else:
            raise ValueError("Invalid auxiliary conditions.")
        return ac

    def as_piecewise(self, fin=None, ac="initial_rest"):
        pac = self._process_aux_conditions(ac)
        n = self.iv
        N = self.order
        if ac in ["initial_rest", "causal"]:
            y = sp.Piecewise((self.as_forward_recursion(fin), n >= 0), (0, True))
        elif ac in ["final_rest", "anticausal"]:
            y = sp.Piecewise((self.as_backward_recursion(fin), n <= -1), (0, True))
        else:
            y = sp.Piecewise(
                (self.as_forward_recursion(fin), n > max(pac.keys())),
                *[(ac[n0], sp.Eq(n, n0)) for n0 in pac.keys()],
                (self.as_backward_recursion(fin), n < min(pac.keys()))
            )
        return y

    @property
    def x_roots(self):
        return self._roots(self.B)

    @property
    def y_roots(self):
        return self._roots(self.A)

    def _roots(self, vect):
        # try cosine
        if len(vect) == 3 and vect[1] <= 0 and vect[2] > 0:
            theta = sp.Wild("theta")
            a = sp.Wild("a")
            m = vect[1].match(a * sp.cos(theta))
            if m:
                r = sp.sqrt(vect[2])
                if sp.simplify(m[a] + 2 * r) == sp.S.Zero:
                    roots = {
                        r * sp.exp(sp.I * m[theta]): 1,
                        r * sp.exp(-sp.I * m[theta]): 1,
                    }
                    return roots
        # characteristic polynomial
        symbol = sp.Dummy("lambda")
        chareq = sp.Poly(vect, symbol)
        chareqroots = sp.roots(chareq, symbol)
        if len(chareqroots) != self.order:
            chareqroots = [sp.rootof(chareq, k) for k in range(chareq.degree())]
        # Create a dict root: multiplicity
        roots = defaultdict(int)
        for root in chareqroots:
            roots[root] += 1
        return roots

    def solve_homogeneous(self):
        n = self.iv
        roots = self.y_roots
        gensols = []
        done = False
        if len(roots) == 2:
            lroots = list(roots)
            r = sp.Wild("r")
            theta = sp.Wild("theta")
            m1 = lroots[0].match(r * sp.exp(sp.I * theta))
            m2 = lroots[1].match(r * sp.exp(sp.I * theta))
            if (m1 and m2) and (m1[r] == m2[r]) and (m1[theta] == -m2[theta]):
                gensols.append(m1[r] ** n * sp.cos(m1[theta] * n))
                gensols.append(m1[r] ** n * sp.sin(m1[theta] * n) * sp.I)
                done = True
        # Characteristic polynomial
        if not done:
            chareq_is_complex = not all([i.is_real for i in self.A])
            # generate solutions
            conjugate_roots = []  # used to prevent double-use of conjugate roots
            # Loop over roots in the order provided by roots/rootof...
            for root in list(roots):
                # but don't repeat multiple roots.
                if root not in roots:
                    continue
                multiplicity = roots.pop(root)
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
                        gensols.append(
                            n ** i * absroot ** n * sp.sin(angleroot * n) * sp.I
                        )
        # Constants
        const_gen = iter_numbered_constants(sp.Add(*self.A), start=1, prefix="C")
        consts = [next(const_gen) for i in range(len(gensols))]
        yh = sp.Add(*[c * g for c, g in zip(consts, gensols)])
        return yh, consts[: self.order]

    def solve_free(self, ac):
        # homogeneous solution
        yh, consts = self.solve_homogeneous()
        n = self.iv
        if isinstance(ac, dict):
            if len(ac.keys()) != self.order:
                raise ValueError(
                    "The number of auxiliary conditions is not equal to the LCDDE order."
                )
            nvals = ac.keys()
        elif ac == "initial_rest" or ac == "final_rest":
            return sp.S.Zero
        else:
            raise ValueError("Invalid auxiliary conditions.")
        # apply auxiliary conditions
        eqs = [sp.Eq(yh.subs(n, k), ac[k]) for k in nvals]
        sol = sp.linsolve(eqs, consts)
        if sol is None:
            raise ValueError("The equation cannot be solved.")
        yf = yh
        for c, s in zip(consts, list(sol)[0]):
            yf = yf.subs(c, s)
        return yf

    def solve_partial(self, fin):
        partial_sols = []
        if self.M >= self.N:
            n = self.iv
            lccde = self.copy()
            while lccde.M >= lccde.N:
                B = lccde.B
                xM = self.x(n - lccde.M + lccde.N)
                yk = B[-1] / self.A[-1] * xM
                partial_sols.append(yk.subs(xM, fin.subs(n, n - lccde.M + lccde.N)))
                del B[-1]
                new_x_part = (
                    sp.Add(*[bk * self.x(n - k) for k, bk in enumerate(B)]) - yk
                )
                lccde = LCCDE.from_expression(
                    sp.Eq(self.y_part(), new_x_part), self.x(), self.y()
                )
        else:
            lccde = self
        return partial_sols, lccde

    def solve_particular(self, fin):
        finc = sp.sympify(fin).expand()
        if finc.is_Add:
            raise ValueError("Cannot solve fon function addition.")
        for f in sp.Mul.make_args(fin):
            if f.is_Function:
                if not isinstance(
                    f, (UnitDelta, UnitStep, UnitRamp, sp.cos, sp.sin, sp.exp)
                ):
                    raise NotImplementedError
        n = self.iv
        nm = 0
        guess = sp.S.One
        cgen = iter_numbered_constants(fin, start=1, prefix="K")
        consts = []
        roots = self.y_roots
        terms, gens = fin.as_terms()
        for m, e in zip(gens, terms[0][1][1]):
            if m.is_Pow:
                b, exp = m.as_base_exp()
                if b in roots and exp == n:
                    multiplicity = roots[b] + 1
                    pguess = sp.S.Zero
                    for i in range(1, multiplicity):
                        consts.append(next(cgen))
                        pguess += consts[-1] * n ** i
                else:
                    pguess = sp.S.One
                guess *= pguess * m ** e
                guess = sp.simplify(guess)
                continue
            elif isinstance(m, (sp.cos, sp.sin)):
                nm = max(nm, sp.solve(m.args[0], n)[0])
                consts.append(next(cgen))
                consts.append(next(cgen))
                guess *= consts[0] * sp.cos(m.args[0]) + consts[1] * sp.sin(m.args[0])
                continue
            elif isinstance(m, sp.exp):
                nm = max(nm, sp.solve(m.args[0], n)[0])
                consts.append(next(cgen))
                consts.append(next(cgen))
                guess *= consts[0] * sp.exp(m.args[0]) + consts[1] * sp.exp(-m.args[0])
                continue
            if isinstance(m, UnitDelta):
                return sp.S.Zero
            elif isinstance(m, UnitStep):
                nm = max(nm, sp.solve(m.args[0], n)[0])
                if guess == sp.S.One or guess.is_Pow:
                    consts.append(next(cgen))
                    guess *= consts[0] * m
                else:
                    guess *= m
                break
            elif isinstance(m, UnitRamp):
                nm = max(nm, sp.solve(m.args[0], n)[0])
                if guess == sp.S.One or guess.is_Pow:
                    consts.append(next(cgen))
                    consts.append(next(cgen))
                    guess *= consts[0] + consts[1] * m
                else:
                    guess *= m
                break
        nmin = nm + self.N
        yg = self.apply_output(guess)
        xp = self.apply_input(fin)
        eqs = []
        for n0 in range(nmin, nmin + len(consts)):
            eqs.append(sp.Eq(yg.subs(n, n0), xp.subs(n, n0)))
        sol = sp.linsolve(eqs, consts)
        if sol == sp.S.EmptySet:
            raise ValueError("Cannot solve for constants.")
        yp = guess
        for c, s in zip(consts, list(sol)[0]):
            yp = yp.subs(c, s)
        return yp

    def solve_forced(self, fin, ac):
        if ac not in ["initial_rest", "causal", "final_rest", "anticausal"]:
            raise ValueError("Invalid auxiliary conditions.")
        return self.solve(fin, ac)

    def solve_transient_and_steady_state(self, fin):
        y = self.solve_forced(fin, ac="initial_rest")
        yss = sp.limit(y, self.iv, sp.S.Infinity)
        yt = sp.simplify(y - yss)
        return (yt, yss)

    def solve(self, fin, ac):
        # partial solutions: extract solutions up to N < M and reduced lccde
        partial_sols, lccde = self.solve_partial(fin)
        # homogeneous and particular
        yh, consts = self.solve_homogeneous()
        yp = lccde.solve_particular(fin)
        # auxiliary conditions
        n = self.iv
        if isinstance(ac, str):
            zone = None
            try:
                zone = int(ac)
            except:
                pass
            if zone == 0 or ac == "initial_rest" or ac == "causal":
                span = range(0, len(consts))
                yhs = [[yh * UnitStep(n)]]
            elif zone == -1 or ac == "final_rest" or ac == "anticausal":
                span = range(-1, -len(consts) - 1, -1)
                yhs = [[yh * UnitStep(-n - 1)]]
            else:
                pass
        else:
            m = min(ac.keys())
            M = max(ac.keys())
            span = list(range(M + 1, M + 1 + len(consts)))
            span += list(range(m - 1, m - 1 - len(consts), -1))
            # span = range(m, M + 1)
            roots = self.y_roots
            gens = []
            k = 0
            for r in roots:
                multiplicity = roots[r]
                rterm = sp.S.Zero
                for m in range(multiplicity):
                    rterm += consts[k] * (n ** m) * (r ** n)
                    k += 1
                gens.append([rterm * UnitStep(n), rterm * UnitStep(-n - 1)])
            yhs = product(*gens)
        # difference equation
        yeq = self.as_piecewise(fin, ac)
        # try to solve for any yh
        for yh in yhs:
            ysol = sp.Add(*yh) + yp
            eqs = [sp.Eq(ysol.subs(n, k), yeq.subs(n, k)) for k in span]
            eqs = list(filter(lambda e: e.has(*consts), eqs))
            sol = sp.linsolve(eqs, consts)
            if sol == sp.S.EmptySet:
                continue
            if any([c == 0 for c in list(sol)[0]]):
                continue
            sols = list(sol)[0]
            snull = False
            for s in sols:
                # auxiliary conditions
                while True:
                    a = list(s.atoms(self.y()))
                    if len(a) == 0:
                        break
                    s = s.subs(a[0], yeq.subs(n, a[0].args[0]))
                if s == 0:
                    snull = True
                    break
            if snull:
                continue
            # homogeneous constants
            y = ysol
            for c, s in zip(consts, list(sol)[0]):
                ysol = ysol.subs(c, s)
            # auxiliary conditions
            while True:
                a = list(ysol.atoms(self.y()))
                if len(a) == 0:
                    break
                ysol = ysol.subs(a[0], yeq.subs(n, a[0].args[0]))
            # add partial solutions
            ysol += sp.Add(*partial_sols)
            ysol = stepsimp(ysol)
            return ysol
        raise ValueError("Cannot solve for constants.")
