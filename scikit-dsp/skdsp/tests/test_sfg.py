import pytest
import sympy as sp

from skdsp.util.sfg import SignalFlowGraph, SignalFlowGraph_DirectFormII


class Test_SFG(object):
    def test_SFG_constructor(self):
        b0, b1, b2 = sp.symbols('b:3')
        _, a1, a2 = sp.symbols('a:3')
        z = sp.Symbol('z')
        delay = z**(-1)
        sources = ["X"]
        sinks = ["Y"]
        edges = [
            ("X", "A0"),
            ("A0", "W0"),
            ("W0", "B0", {"expr": b0}),
            ("B0", "Y"),
            ("A1", "A0"),
            ("W0", "W1", {"expr": delay}),
            ("W1", "A1", {"expr": a1}),
            ("W1", "B1", {"expr": b1}),
            ("B1", "B0"),
            ("A2", "A1"),
            ("W1", "W2", {"expr": delay}),
            ("W2", "A2", {"expr": a2}),
            ("W2", "B2", {"expr": b2}),
            ("B2", "B1"),
        ]
        biquad = SignalFlowGraph(sources, sinks, edges)
        gain = biquad.graph_gain()
        expected = (b0 + b1 * delay + b2 * delay ** 2) / (1 - (a1 * delay + a2 * delay ** 2))
        assert sp.simplify(gain - expected) == sp.S.Zero

        sfg = SignalFlowGraph_DirectFormII([b0, b1, b2], [1, a1, a2])
        assert sfg == biquad
        assert sp.simplify(sfg.graph_gain() - expected) == sp.S.Zero

    def test_SFG_program(self):
        # b0, b1, b2 = sp.symbols('b:3')
        # _, a1, a2 = sp.symbols('a:3')
        # B = [b0, b1, b2]
        # A = [1, a1, a2]
        B = [sp.S(1) / 4, sp.S(1) / 8, 1]
        A = list(reversed(B))
        sfg = SignalFlowGraph_DirectFormII(B, A)
        with pytest.raises(ValueError):
            sfg.to_program('C++')
        program = sfg.to_program('python')
        with open('SFG.py', 'wt') as f:
            f.write(program)
        