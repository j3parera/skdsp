import sympy as sp
from skdsp.signal.signal import Signal


class System(sp.Basic):
    def __new__(cls, T, x, y):
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

    def _traverse_compare_iv(self, cmpfcn, fcn):
        insum = False
        inlim = False
        for expr in sp.preorder_traversal(self.mapping):
            if isinstance(expr, sp.Sum):
                efcn = expr.function
                if fcn.func == efcn.func:
                    arg = fcn.args[0]
                    limits = expr.limits[0]
                    try:
                        term = efcn.subs({limits[0]: limits[1]})
                        cmp = cmpfcn(term.args[0], arg)
                        if cmp:
                            return True
                        term = efcn.subs({limits[0]: limits[2]})
                        cmp = cmpfcn(term.args[0], arg)
                        if cmp:
                            return True
                    except:
                        return True
                    insum = True
            elif not insum and isinstance(expr, sp.Function):
                if fcn.func == expr.func:
                    arg = fcn.args[0]
                    try:
                        cmp = cmpfcn(expr.args[0], arg)
                        if cmp:
                            return True
                    except:
                        return True
            elif insum and not inlim:
                if expr == limits:
                    inlim = True
            elif insum and inlim:
                if expr == limits[2]:
                    inlim = False
                    insum = False
        return None

    @property
    def _depends_on_outputs(self):
        fcns = self.mapping.atoms(sp.Function)
        for fcn in fcns:
            if fcn.func == self.output_.func:
                return True
        return False
    
    @property
    def _depends_on_inputs(self):
        fcns = self.mapping.atoms(sp.Function)
        for fcn in fcns:
            if fcn.func == self.input_.func:
                return True
        return False
    
    @property
    def is_recursive(self):
        o = self._traverse_compare_iv(sp.Lt, self.output_)
        return o is not None and o

    @property
    def is_memoryless(self):
        i = self._traverse_compare_iv(sp.Ne, self.input_)
        o = self._traverse_compare_iv(sp.Ne, self.output_)
        return not i and not o

    is_static = is_memoryless

    @property
    def is_dynamic(self):
        return not self.is_memoryless

    @property
    def is_time_invariant(self):
        # if past values of output: can't tell
        if self._depends_on_outputs:
            return None
        dummy = sp.Dummy(integer=True, nonnegative=True)
        x1 = Signal(sp.Function("x1")(self.input_.args[0]))
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
        # if past values of output: can't tell
        if self._depends_on_outputs:
            return None
        a, b = sp.symbols("a, b")
        x1 = Signal(sp.Function("x1")(self.input_.args[0]))
        x2 = Signal(sp.Function("x2")(self.input_.args[0]))
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
        i = self._traverse_compare_iv(sp.Gt, self.input_)
        o = self._traverse_compare_iv(sp.Gt, self.output_)
        return not i and not o

    @property
    def is_anticausal(self):
        i = self._traverse_compare_iv(sp.Lt, self.input_)
        o = self._traverse_compare_iv(sp.Lt, self.output_)
        if i is None and o is None:
            return False
        return not i and not o

    @property
    def is_stable(self):
        if self._depends_on_outputs:
            return None
        M = sp.Symbol('M', finite=True)
        x1 = Signal(M, iv=self.input_.args[0])
        y1 = self.apply(x1)
        lim = sp.limit(sp.Abs(y1.amplitude), y1.iv, sp.S.Infinity)
        return lim.is_finite

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
            for s, v in zip(self.mapping.free_symbols, args):
                p[s] = v
        return self.eval(x, p)
