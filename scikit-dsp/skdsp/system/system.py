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
    def input(self):
        return self.args[0]

    @property
    def output(self):
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

    @property
    def is_memoryless(self):
        delay = sp.Wild('delay', integer=True)
        syms = [self.input.args[0]]
        if self.is_hybrid:
            syms.append(self.output.args[0])
        for arg in sp.preorder_traversal(self.mapping):
            for sym in syms:
                m = arg.match(sym - delay)
                if m is not None:
                    d = m.get(delay)
                    if d != sp.S.Zero and d.is_constant(sym):
                        return False
        return True
    
    is_static = is_memoryless

    @property
    def is_dynamic(self):
        return not self.is_memoryless

    @property
    def is_time_invariant(self):
        raise NotImplementedError

    is_shift_invariant = is_time_invariant

    @property
    def is_time_variant(self):
        raise NotImplementedError

    is_shift_variant = is_time_variant

    @property
    def is_linear(self):
        raise NotImplementedError
    
    @property
    def is_lti(self):
        return self.is_linear and self.is_time_invariant

    is_lsi = is_lti

    @property
    def is_causal(self):
        raise NotImplementedError
    
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

    def apply(self, input):
        raise NotImplementedError

    eval = apply