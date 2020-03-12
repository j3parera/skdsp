from collections import defaultdict

import sympy as sp
from sympy.solvers.ode import get_numbered_constants, iter_numbered_constants

class LCCDE(sp.Basic):
    
    @classmethod
    def from_expression(cls, f, x, y):
        if isinstance(f, sp.Equality):
            f = f.lhs - f.rhs
        n = y.args[0]
        k = sp.Wild('k', exclude=(n,))
        # Preprocess user input to allow things like
        # y(n) + a*(y(n + 1) + y(n - 1))/2
        f = f.expand().collect(y.func(sp.Wild('m', integer=True)))
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
                            raise ValueError("'%s(%s + k)' expected, got '%s'" % (y.func, n, h))
                    elif h.func == x.func:
                        result = h.args[0].match(n + k)
                        if result is not None:
                            kspec = int(result[k])
                            if kspec is not None:
                                i_part[kspec] += -coeff
                        else:
                            raise ValueError("'%s(%s + k)' expected, got '%s'" % (x.func, n, h))
                    else:
                        if not h.is_constant(n):
                            raise ValueError("Non constant coefficient found: {0}".format(h))
                else:
                    if not h.is_constant(n):
                        raise ValueError("Non constant coefficient found: {0}".format(h))
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
    
    def __new__(cls, B=[], A=[]):
        n = sp.Symbol('n', integer=True)
        x = sp.Function('x')(n)
        y = sp.Function('y')(n)
        B = [sp.S(b) / sp.S(A[0]) for b in B]
        A = [sp.S(a) / sp.S(A[0]) for a in A]
        obj = sp.Basic.__new__(cls, x, y, B, A)
        return obj

    @property
    def x(self):
        return self.args[0]

    @property
    def y(self):
        return self.args[1]

    @property
    def iv(self):
        return self.y.args[0]

    @property
    def B(self):
        return self.args[2]

    @property
    def A(self):
        return self.args[3]

    @property
    def order(self):
        return len(self.A) - 1

    @property
    def as_expression(self):
        ey = sp.S.Zero
        for k, c in enumerate(self.A):
            ey += c * self.y.subs(self.iv, (self.iv - k))
        ex = sp.S.Zero
        for k, c in enumerate(self.B):
            ex += c * self.x.subs(self.iv, (self.iv - k))
        return sp.Eq(ey, ex)        

    def solve_homogeneous(self, ac=None):
        n = self.iv
        # Characteristic equation
        chareq, symbol = sp.S.Zero, sp.Dummy('lambda')
        for k, a in enumerate(reversed(self.A)):
            chareq += a*symbol**k
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
        conjugate_roots = [] # used to prevent double-use of conjugate roots
        # Loop over roots in the order provided by roots/rootof...
        for root in chareqroots:
            # but don't repeat multiple roots.
            if root not in charroots:
                continue
            multiplicity = charroots.pop(root)
            for i in range(multiplicity):
                if chareq_is_complex:
                    gensols.append(n**i*root**n)
                    continue
                absroot = sp.Abs(root)
                angleroot = sp.arg(root)
                if angleroot.has(sp.atan2):
                    # Remove this condition when arg stop returning circular atan2 usages.
                    gensols.append(n**i*root**n)
                else:
                    if root in conjugate_roots:
                        continue
                    if angleroot == 0:
                        gensols.append(n**i*absroot**n)
                        continue
                    conjugate_roots.append(sp.conjugate(root))
                    gensols.append(n**i*absroot**n*sp.cos(angleroot*n))
                    gensols.append(n**i*absroot**n*sp.sin(angleroot*n))
        # Constants
        const_gen = iter_numbered_constants(chareq, start=1, prefix='C')
        consts = [next(const_gen) for i in range(len(gensols))]
        yh = sp.Add(*[c * g for c, g in zip(consts, gensols)])
        if ac is not None:
            if isinstance(ac, dict):
                if len(ac.keys()) != self.order:
                    raise ValueError("The number of auxiliary conditions is not equal to the LCDDE order.")
                eqs = []
                for k, v in ac.items():   
                    eqs.append(sp.Eq(yh.subs(n, k), v))
            elif isinstance(ac, str) and ac == 'initial_rest':
                eqs = []
                for k in range(-1, -self.order-1, -1):
                    eqs.append(sp.Eq(yh.subs(n, k), 0))
            else:
                raise ValueError("Not valid auxiliary conditions.")
            sol = sp.solve(eqs, consts)
            yh = yh.subs(sol)
        return yh

    def solve_homogeneous_bad(self, ac=None):
        n = self.iv
        # Characteristic equation
        chareq, symbol = sp.S.Zero, sp.Dummy('lambda')
        for k, a in enumerate(reversed(self.A)):
            chareq += a*symbol**k
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
        conjugate_roots = [] # used to prevent double-use of conjugate roots
        # Loop over roots in the order provided by roots/rootof...
        for root in chareqroots:
            # but don't repeat multiple roots.
            if root not in charroots:
                continue
            multiplicity = charroots.pop(root)
            for i in range(multiplicity):
                if chareq_is_complex:
                    gensols.append(n**i*root**n)
                    continue
                absroot = sp.Abs(root)
                angleroot = sp.arg(root)
                if angleroot.has(sp.atan2):
                    # Remove this condition when arg stop returning circular atan2 usages.
                    gensols.append(n**i*root**n)
                else:
                    if root in conjugate_roots:
                        continue
                    if angleroot == 0:
                        gensols.append(n**i*absroot**n)
                        continue
                    conjugate_roots.append(sp.conjugate(root))
                    # gensols.append(n**i*absroot**n*sp.sin(angleroot*n))
                    gensols.append(2*n**i*absroot**n*sp.cos(angleroot*n))
                    gensols.append(2*n**i*absroot**n*sp.sin(angleroot*n))
        # Constants
        constsm = get_numbered_constants(chareq, num=len(gensols)+1, prefix='A')
        solm = sp.Add(*[c * g for c, g in zip(constsm, gensols)])
        constsp = get_numbered_constants(chareq, num=len(gensols)+1, prefix='B')
        solp = sp.Add(*[c * g for c, g in zip(constsp, gensols)])
        yh = sp.Piecewise((solm, n < 0), (solp, n >= 0))
        # Input
        eqs = []
        for kx, bk in enumerate(self.B):
            if bk == sp.S.Zero:
                continue
            adds = [ak * yh.subs(n, kx-ky) for ky, ak in enumerate(self.A)]
            eqk = sp.Eq(sp.Add(*adds), bk)
            eqs.append(eqk)
        sol = sp.solve(eqs, constsm)
        if sol:
            yh = yh.subs(sol)
        else:
            for e in eqs:
                sol = sp.solve(e, constsm)
                if sol:
                    yh = yh.subs(dict(zip(constsm, sol[0])))
        if ac is not None:
            if isinstance(ac, dict):
                if len(ac.keys()) != self.order:
                    raise ValueError("The number of auxiliary conditions is not equal to the LCDDE order.")
                eqs = []
                for k, v in ac.items():   
                    eqs.append(sp.Eq(yh.subs(n, k), v))
            elif isinstance(ac, str) and ac == 'initial_rest':
                eqs = []
                for k in range(-1, -self.order-1, -1):
                    eqs.append(sp.Eq(yh.subs(n, k), 0))
            else:
                raise ValueError("Not valid auxiliary conditions.")
            sol = sp.solve(eqs, constsp)
            if sol:
                yh = yh.subs(sol)
            else:
                for e in eqs:
                    sol = sp.solve(e, constsp)
                    if sol:
                        yh = yh.subs(dict(zip(constsp, sol[0])))
        return yh
