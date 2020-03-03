import sympy as sp
from skdsp.signal.signal import Signal

class System(sp.Basic):

    def __new__(cls, T, x, y, domain=sp.S.Integers, codomain=sp.S.Integers):
        if not x.is_Function or not y.is_Function:
            raise ValueError("Input and output must be functions.")
        if isinstance(T, sp.Equality):
            T = T.lhs - T.rhs
        sol = sp.solve(T, y)
        if not sol:
            raise ValueError("Mapping doesn't solve for output signal.")
        T = sol[0]
        if not T.has(x.func):
            raise ValueError("Mapping doesn't depends on input signal.")
        domain = sp.S.Integers if x.args[0].is_integer else sp.S.Reals
        codomain = sp.S.Integers if y.args[0].is_integer else sp.S.Reals
        obj = sp.Basic.__new__(cls, x, y, T, domain, codomain)
        return obj

    @property
    def input_(self):
        return self.args[0]

    @property
    def output_(self):
        return self.args[1]

    @property
    def mapping(self):
        return self.args[2]

    @property
    def domain(self):
        return self.args[3]

    @property
    def codomain(self):
        return self.args[4]

    def _traverse_compare(self, cmpfnc):
        fnci = self.input_
        argi = fnci.args[0]
        fnco = self.output_
        argo = fnco.args[0]
        insum = False
        inlim = False
        for expr in sp.preorder_traversal(self.mapping):
            if isinstance(expr, sp.Sum):
                fcn = expr.function
                if fcn.func == fnci.func:
                    arg = argi
                elif fcn.func == fnco.func:
                    arg = argo
                else:
                    continue
                limits = expr.limits[0]
                try:
                    term = fcn.subs({limits[0]: limits[1]})
                    cmp = cmpfnc(term.args[0], arg)
                    if cmp:
                        return False
                    term = fcn.subs({limits[0]: limits[2]})
                    cmp = cmpfnc(term.args[0], arg)
                    if cmp:
                        return False
                except:
                    return False
                insum = True
            elif not insum and isinstance(expr, sp.Function):
                if expr.func == fnci.func:
                    arg = argi
                elif expr.func == fnco.func:
                    arg = argo
                else:
                    continue
                try:
                    cmp = cmpfnc(expr.args[0], arg)
                    if cmp:
                        return False
                except:
                    return False
            elif insum and not inlim:
                if expr == limits:
                    inlim = True
            elif insum and inlim:
                if expr == limits[2]:
                    inlim = False
                    insum = False
        return True

    
    @property
    def is_memoryless(self):
        return self._traverse_compare(sp.Ne)
    
    is_static = is_memoryless

    @property
    def is_dynamic(self):
        return not self.is_memoryless

    @property
    def is_time_invariant(self):
        dummy = sp.Dummy(integer=True, nonnegative=True)
        x1 = Signal(sp.Function('x1')(self.input_.args[0]))
        y1 = self.apply(x1).shift(dummy)
        y2 = self.apply(x1.shift(dummy))
        d = sp.simplify((y1 - y2).amplitude)
        return d == sp.S.Zero

    is_shift_invariant = is_time_invariant

    @property
    def is_time_variant(self):
        return not self.is_time_invariant

    is_shift_variant = is_time_variant

    @property
    def is_linear(self):
        a, b = sp.symbols('a, b')
        x1 = Signal(sp.Function('x1')(self.input_.args[0]))
        x2 = Signal(sp.Function('x2')(self.input_.args[0]))
        y1 = self.apply(a * x1 + b * x2)
        y2 = a * self.apply(x1) + b * self.apply(x2)
        d = sp.simplify((y1 - y2).amplitude)
        return d == sp.S.Zero
    
    @property
    def is_lti(self):
        return self.is_linear and self.is_time_invariant

    is_lsi = is_lti

    @property
    def is_causal(self):
        return self._traverse_compare(sp.Gt)
    
    @property
    def is_anticausal(self):
        return self._traverse_compare(sp.Lt)
    
    @property
    def is_stable(self):
        raise NotImplementedError
    
    @property
    def is_discrete(self):
        return self.is_input_discrete and self.is_output_discrete

    @property
    def is_continuous(self):
        return self.is_input_continuous and self.is_output_continuous

    @property
    def is_hybrid(self):
        return self.domain != self.codomain

    @property
    def is_input_discrete(self):
        return self.domain == sp.S.Integers

    @property
    def is_output_discrete(self):
        return self.codomain == sp.S.Integers

    @property
    def is_input_continuous(self):
        return self.domain == sp.S.Reals

    @property
    def is_output_continuous(self):
        return self.codomain == sp.S.Reals

    def apply(self, ins, params={}):
        if not isinstance(params, dict):
            raise ValueError("Parameter values must be in a dictionary.")
        if not isinstance(ins, Signal):
            raise ValueError("Input must be a signal.")
        T = self.mapping
        if params:
            T = T.subs(params)

        def _apply(expr, ins):
            if isinstance(expr, sp.Add):
                result = sp.S.Zero
                for arg in expr.args:
                    result += _apply(arg, ins)
                return result
            elif isinstance(expr, sp.Mul):
                result = sp.S.One
                for arg in expr.args:
                    result *= _apply(arg, ins)
                return result
            elif isinstance(expr, sp.Pow):
                return _apply(expr.base, ins) ** expr.exp
            elif isinstance(expr, sp.Function):
                if expr.func == self.input_.func:
                    newarg = expr.args[0].subs({self.input_.args[0]: ins.iv})
                    e = ins.amplitude.subs({ins.iv: newarg})
                    return e
                return expr
            elif isinstance(expr, sp.Sum):
                return sp.Sum(_apply(expr.function, ins), expr.limits)
            else:
                return expr

        res = _apply(T, ins)
        result = ins.clone(None, res, period=None)
        return result       

    eval = apply

    def __call__(self, x, *args):
        p = dict()
        if len(args) != 0:
            for s, v in zip(self.free_symbols, args):
                p[s] = v
        return self.eval(x, p)
