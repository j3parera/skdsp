from collections import defaultdict

import sympy as sp
from sympy.solvers.ode import get_numbered_constants

class LCCDE(sp.Basic):
    
    def __new__(cls, B=[], A=[]):
        n = sp.Symbol('n', integer=True)
        x = sp.Function('x')(n)
        y = sp.Function('y')(n)
        mm = max(len(B), len(A))
        B = B + [0] * (mm - len(B))
        B = [sp.S(b) / sp.S(A[0]) for b in B]
        A = A + [0] * (mm - len(A))
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
        # Loop over roots in theorder provided by roots/rootof...
        for root in chareqroots:
            # but don't repoeat multiple roots.
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
                    gensols.append(n**i*absroot**n*sp.sin(angleroot*n))
                    gensols.append(n**i*absroot**n*sp.cos(angleroot*n))
        # Constants
        constsm = get_numbered_constants(chareq, num=len(gensols)+1, prefix='A')
        solm = sp.Add(*[c * g for c, g in zip(constsm, gensols)])
        constsp = get_numbered_constants(chareq, num=len(gensols)+1, prefix='B')
        solp = sp.Add(*[c * g for c, g in zip(constsp, gensols)])
        yh = sp.Piecewise((solm, self.iv < 0), (solp, self.iv >= 0))
        # Input
        eqs = []
        for kx, bk in enumerate(self.B):
            if bk == sp.S.Zero:
                continue
            eqk = sp.Eq(sp.Add(*[ak * yh.subs(self.iv, kx-ky) for ky, ak in enumerate(self.A)]), bk)
            eqs.append(eqk)
        yh = yh.subs(sp.solve(eqs, constsm))
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
            yh = yh.subs(sp.solve(eqs, constsp))
        return yh
