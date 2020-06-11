from itertools import combinations
from functools import reduce

import networkx as nx
import sympy as sp

_delay = sp.Symbol("z") ** (-1)


class SignalFlowGraph(object):
    def __init__(self, sources, sinks, edges):
        self._graph = nx.DiGraph()
        self._sources = sources
        self._sinks = sinks
        self._graph.add_edges_from(edges)

    @property
    def sources(self):
        return self._sources

    @property
    def sinks(self):
        return self._sinks

    @property
    def nodes(self):
        return self._graph.nodes

    @property
    def edges(self):
        return self._graph.edges

    @property
    def cycles(self):
        return list(map(lambda x: tuple(x), nx.simple_cycles(self._graph)))

    @property
    def determinant(self):
        return self._find_cofactor(self.cycles)

    @property
    def Δ(self):
        return self.determinant

    def graph_gain(self, from_node=None, to_node=None):
        if from_node is None:
            from_node = self.sources[0]
        if to_node is None:
            to_node = self.sinks[0]
        paths = nx.all_simple_paths(self._graph, from_node, to_node)
        forward_gain = 0
        for path in paths:
            # Calculate the path gain
            path_gain = 1
            for i in range(len(path) - 1):
                edge_gain = sp.S.One
                try:
                    edge_gain = self._graph.edges[path[i], path[i + 1]]["expr"]
                except KeyError:
                    pass
                path_gain *= edge_gain
            # Calculate the path's cofactor
            nontouching_cycles = []
            for cycle in self.cycles:
                if set(path).isdisjoint(set(cycle)):
                    nontouching_cycles.append(cycle)
            cofactor = self._find_cofactor(nontouching_cycles)
            # Add to graph gain
            forward_gain += path_gain * cofactor
        return forward_gain / self.Δ

    def _cycle_gain(self, cycle):
        gain = sp.S.One
        try:
            gain = self._graph.edges[cycle[-1], cycle[0]]["expr"]
        except KeyError:
            pass
        for i in range(len(cycle) - 1):
            edge_gain = sp.S.One
            try:
                edge_gain = self._graph.edges[cycle[i], cycle[i + 1]]["expr"]
            except KeyError:
                pass
            gain *= edge_gain
        return gain

    def _find_cofactor(self, cycles):
        cofactor = 1
        sign = 1
        for i in range(len(cycles)):
            sign *= -1
            for subcycles in combinations(cycles, i + 1):
                # Check if the combinations are not touchable
                full_len = len(reduce(lambda x, y: x | set(y), subcycles, set()))
                if full_len == reduce(lambda x, y: x + len(y), subcycles, 0):
                    # If the combination doesn't have cycles that are touchable
                    # to each other, then add the gain to cofactor
                    cofactor += reduce(
                        lambda x, y: x * self._cycle_gain(y), subcycles, sign
                    )
        return cofactor

    def __eq__(self, other):
        if not isinstance(other, SignalFlowGraph):
            return False
        import networkx.algorithms.isomorphism as iso
        from operator import eq

        em = iso.generic_edge_match("expr", sp.S.One, eq)
        ok = nx.is_isomorphic(self._graph, other._graph, edge_match=em)
        ok &= self._sources == other._sources
        ok &= self._sinks == other._sinks
        return ok

    def transpose(self):
        rev = self._graph.reverse(False)
        return SignalFlowGraph(self.sinks, self.sources, rev.edges)

    def _expand_delays_and_cut(self, mem_prefix='M'):
        G = self._graph.copy()
        tochange = []
        cuts = []
        for (u, v, expr) in G.edges.data("expr", default=sp.S(None)):
            if expr == _delay:
                tochange.append((u, v))
        m = 1
        for u, v in tochange:
            G.remove_edge(u, v)
            M = mem_prefix + str(m)
            m += 1
            # G.add_edge(u, M, expr=_delay)
            cuts.append((u, M))
            G.add_edge(M, v)
        return G, cuts

    @property
    def is_computable(self):
        for cycle in self.cycles:
            G = self._graph.subgraph(cycle)
            for (u, v, expr) in G.edges.data("expr", default=sp.S(None)):
                if expr == _delay:
                    break
            else:
                return False
        return True                    

    def to_program(self, language='python', func_name='SFG', mem_prefix='M', source=None, sink=None):
        lang = language.lower()
        if lang not in ['python', 'matlab']:
            raise ValueError("Don't know how to make a {0} program".format(language))
        if not self.is_computable:
            raise ValueError("No computable graph.")
        if source is None:
            if len(self.sources) != 1:
                raise ValueError("Please, specify source!")
            source = self.sources[0]
        else:
            if not source in self.sources:
                raise ValueError("Invalid source!")
        if sink is None:
            if len(self.sinks) != 1:
                raise ValueError("Please, specify sink!")
            sink = self.sinks[0]
        else:
            if not sink in self.sinks:
                raise ValueError("Invalid sink!")
        G, cuts = self._expand_delays_and_cut(mem_prefix)
        tor = nx.topological_sort(G)
        if lang == 'matlab':
            program = "function [y] = {0}(x, mem)\n".format(func_name)
            program += "\ty = zeros(size(x));\n"
            for k in range(len(cuts)):
                program += "\t{0}{1} = mem({1});\n".format(mem_prefix, k + 1)
            program += "\tfor n = 1:length(x)\n"
            program += "\t\t{0} = x(n);\n".format(source)
            for node in tor:
                sentence = None
                for k, p in enumerate(G.predecessors(node)):
                    if sentence is None:
                        sentence = "\t\t{0} = ".format(node)
                    plus = '' if k == 0 else ' + ' 
                    expr = (G[p][node]).get('expr', sp.S.One)
                    exprstr = str(expr).strip() + '*' if expr != sp.S.One else ''
                    sentence += plus + "{0}{1}".format(exprstr, p)
                if sentence is not None:
                    program += sentence.rstrip() + ";\n"
            program += "\t\ty(n) = {0};\n".format(sink)
            for q, m in cuts:
                program += "\t\t{0} = {1};\n".format(m, q)
            program += "\tend\nend\n"
        elif lang == 'python':
            program = "def {0}(x, mem):\n".format(func_name)
            program += "\ty = []\n"
            for k in range(len(cuts)):
                program += "\t{0}{1} = mem[{1}]\n".format(mem_prefix, k + 1)
            program += "\tfor n in range(len(x)):\n"
            program += "\t\t{0} = x[n]\n".format(source)
            for node in tor:
                sentence = None
                for k, p in enumerate(G.predecessors(node)):
                    if sentence is None:
                        sentence = "\t\t{0} = ".format(node)
                    plus = '' if k == 0 else ' + ' 
                    expr = (G[p][node]).get('expr', sp.S.One)
                    exprstr = str(expr).strip() + '*' if expr != sp.S.One else ''
                    sentence += plus + "{0}{1}".format(exprstr, p)
                if sentence is not None:
                    program += sentence.rstrip() + "\n"
            program += "\t\ty[n] = {0}\n".format(sink)
            for q, m in cuts:
                program += "\t\t{0} = {1}\n".format(m, q)
            program += "\treturn y\n"
        return program

class SignalFlowGraph_DirectFormII(SignalFlowGraph):
    def __init__(self, B, A):
        B = [sp.S(b) / sp.S(A[0]) for b in B]
        A = [sp.S(a) / sp.S(A[0]) for a in A]
        sources = ["X"]
        sinks = ["Y"]
        edges = [("X", "A0"), ("A0", "W0"), ("W0", "B0", {"expr": B[0]}), ("B0", "Y")]
        for k in range(1, max(len(A), len(B))):
            edges.append(("W{0}".format(k - 1), "W{0}".format(k), {"expr": _delay}))
            if k < len(A):
                edges.append(("A{0}".format(k), "A{0}".format(k - 1)))
                edges.append(("W{0}".format(k), "A{0}".format(k), {"expr": A[k]}))
            if k < len(B):
                edges.append(("B{0}".format(k), "B{0}".format(k - 1)))
                edges.append(("W{0}".format(k), "B{0}".format(k), {"expr": B[k]}))
        super().__init__(sources, sinks, edges)
