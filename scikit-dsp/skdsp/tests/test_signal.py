import numpy as np
import pytest
import sympy as sp

from skdsp.signal.signal import Signal


class Test_Signal(object):
    """ Test class for Signal """

    def test_Signal_constructor_with_constant(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)

        s = Signal(2.0, t)
        assert s != None

        s = Signal(2*sp.I, n)
        assert s != None

        s = Signal(2 * n, n)
        assert s != None

        with pytest.raises(ValueError):
            s = Signal(2)

        with pytest.raises(ValueError):
            s = Signal(2.0, "t")

        with pytest.raises(ValueError):
            s = Signal(2, sp.cos(t))

        with pytest.raises(ValueError):
            s = Signal(2*sp.I, t, codomain=sp.S.Integers)

    def test_Signal_constructor_with_expression(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", integer=True)

        s = Signal(sp.cos(x * n), n, codomain=sp.S.Reals)
        assert s != None

        s = Signal(sp.exp(sp.I * k * x * t), n, codomain=sp.S.Complexes)
        assert s != None

        s = Signal("2**t", n, codomain=sp.S.Complexes)
        assert s != None

        with pytest.raises(ValueError):
            s = Signal("2**t", t, sp.S.Reals)

    def test_Signal_constructor_with_undef(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        k = sp.Symbol("k", integer=True)

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s != None

        s = Signal(sp.Function("x")(t))
        assert s != None

        s = Signal(sp.Function("s")(n, t), n)
        assert s != None

        s = Signal(sp.Function("s")(n), t)
        assert s != None
        # pylint: enable-msg=not-callable

    def test_Signal_amplitude(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", integer=True)

        s = Signal(2.0, t)
        assert s.amplitude == 2
        assert sp.simplify(s.magnitude() - 2) == 0
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.S(2.0), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal(2 * sp.I, n)
        assert s.amplitude == 2 * sp.I
        assert sp.simplify(s.magnitude() - 2) == 0
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.S(2), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal(2 * n, n)
        assert s.amplitude == sp.sympify(2 * n)
        assert sp.simplify(s.magnitude() - 2 * sp.Abs(n)) == 0
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.S(2 * sp.Abs(n)), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal("2**t", x, sp.S.Reals)
        assert s.amplitude == sp.sympify("2**t")
        assert sp.simplify(s.magnitude() - sp.Abs(sp.S("2**t")) == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.Abs(sp.S("2**t")), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal(sp.cos(x * n), n, sp.S.Reals)
        assert s.amplitude == sp.cos(x * n)
        s1 = s.magnitude()
        s2 = sp.Abs(sp.cos(x * n))
        assert sp.simplify(s1 - s2 == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.Abs(sp.cos(x * n)), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal(sp.exp(sp.I * k * x * t), n, codomain=sp.S.Complexes)
        assert s.amplitude == sp.exp(sp.I * k * x * t)
        s1 = s.magnitude()
        s2 = 1
        assert sp.simplify(s1 - s2 == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(1, sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.amplitude == sp.Function("X")(k)
        s1 = s.magnitude()
        s2 = sp.Abs(sp.Function("X")(k))
        assert sp.simplify(s1 - s2 == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.Abs(sp.Function("X")(k)), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal(sp.Function("x")(t))
        assert s.amplitude == sp.Function("x")(t)
        s1 = s.magnitude()
        s2 = sp.Abs(sp.Function("x")(t))
        assert sp.simplify(s1 - s2 == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.Abs(sp.Function("x")(t)), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal(sp.Function("s")(n, t), n)
        assert s.amplitude == sp.Function("s")(n, t)
        s1 = s.magnitude()
        s2 = sp.Abs(sp.Function("s")(n, t))
        assert sp.simplify(s1 - s2 == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.Abs(sp.Function("s")(n, t)), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal(sp.Function("s")(n), t)
        assert s.amplitude == sp.Function("s")(n)
        s1 = s.magnitude()
        s2 = sp.Abs(sp.Function("s")(n))
        assert sp.simplify(s1 - s2 == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.Abs(sp.Function("s")(n)), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)
        # pylint: enable-msg=not-callable

        s = Signal(2 * n * t * x, t)
        assert s.amplitude == sp.sympify(2 * n * t * x)
        s1 = s.magnitude()
        s2 = sp.Abs(2 * n * t * x)
        assert sp.simplify(s1 - s2 == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.Abs(2 * n * t * x), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

        s = Signal(5 * n * t, n)
        assert s.amplitude == sp.sympify(5 * n * t, n)
        s1 = s.magnitude()
        s2 = sp.Abs(5 * n * t)
        assert sp.simplify(s1 - s2 == 0)
        s1 = s.magnitude(dB=True)
        s2 = sp.S(20) * sp.log(sp.Abs(5 * n * t), sp.S(10))
        assert sp.simplify(s1 - s2 == 0)

    def test_Signal_iv(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", integer=True)

        s = Signal(2.0, t)
        assert s.iv == t

        s = Signal(2*sp.I, n)
        assert s.iv == n

        s = Signal(2 * n, n)
        assert s.iv == n

        s = Signal("2**t", x, codomain=sp.S.Reals)
        assert s.iv == x

        s = Signal(sp.cos(x * n), n, codomain=sp.S.Reals)
        assert s.iv == n

        s = Signal(sp.exp(sp.I * k * x * t), n, codomain=sp.S.Complexes)
        assert s.iv == n

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.iv == k

        s = Signal(sp.Function("x")(t))
        assert s.iv == t

        s = Signal(sp.Function("s")(n, t), n)
        assert s.iv == n

        s = Signal(sp.Function("s")(n), t)
        assert s.iv == t
        # pylint: enable-msg=not-callable

        s = Signal(2 * n * t * x, t)
        assert s.iv == t

        s = Signal(5 * n * t, n)
        assert s.iv == n

    def test_Signal_domain(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", integer=True)

        s = Signal(2.0, t)
        assert s.domain == sp.S.Reals
        assert s.is_continuous == True
        assert s.is_discrete == False

        s = Signal(2*sp.I, n)
        assert s.domain == sp.S.Integers
        assert s.is_continuous == False
        assert s.is_discrete == True

        s = Signal(2 * n, n)
        assert s.domain == sp.S.Integers
        assert s.is_continuous == False
        assert s.is_discrete == True

        s = Signal("2**t", x, codomain=sp.S.Reals)
        assert s.domain == sp.S.Reals
        assert s.is_continuous == True
        assert s.is_discrete == False

        s = Signal(sp.cos(x * n), n, codomain=sp.S.Reals)
        assert s.domain == sp.S.Integers
        assert s.is_continuous == False
        assert s.is_discrete == True

        s = Signal(sp.exp(sp.I * k * x * t), n, codomain=sp.S.Complexes)
        assert s.domain == sp.S.Integers
        assert s.is_continuous == False
        assert s.is_discrete == True

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.domain == sp.S.Integers
        assert s.is_continuous == False
        assert s.is_discrete == True

        s = Signal(sp.Function("x")(t))
        assert s.domain == sp.S.Reals
        assert s.is_continuous == True
        assert s.is_discrete == False

        s = Signal(sp.Function("s")(n, t), n)
        assert s.domain == sp.S.Integers
        assert s.is_continuous == False
        assert s.is_discrete == True

        s = Signal(sp.Function("s")(n), t)
        assert s.domain == sp.S.Reals
        assert s.is_continuous == True
        assert s.is_discrete == False
        # pylint: enable-msg=not-callable

        s = Signal(2 * n * t * x, t)
        assert s.domain == sp.S.Reals
        assert s.is_continuous == True
        assert s.is_discrete == False

        s = Signal(5 * n * t, n)
        assert s.domain == sp.S.Integers
        assert s.is_continuous == False
        assert s.is_discrete == True

    def test_Signal_codomain(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", integer=True)

        s = Signal(2.0, t)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False

        s = Signal(2*sp.I, n)
        assert s.codomain == sp.S.Complexes
        assert s.is_real == False
        assert s.is_complex == True

        s = Signal(2 * n, n)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False

        s = Signal("2**t", x, codomain=sp.S.Reals)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False

        s = Signal(sp.cos(x * n), n, codomain=sp.S.Reals)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False

        s = Signal(sp.exp(sp.I * k * x * t), n, codomain=sp.S.Complexes)
        assert s.codomain == sp.S.Complexes
        assert s.is_real == False
        assert s.is_complex == True

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.codomain == sp.S.Complexes
        assert s.is_real == False
        assert s.is_complex == True

        s = Signal(sp.Function("x")(t), codomain=sp.S.Reals)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False

        s = Signal(sp.Function("s")(n, t), n, codomain=sp.S.Reals)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False

        s = Signal(sp.Function("s")(n), t, codomain=sp.S.Reals)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False
        # pylint: enable-msg=not-callable

        s = Signal(2 * n * t * x, t, codomain=sp.S.Reals)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False

        s = Signal(5 * n * t, n, codomain=sp.S.Reals)
        assert s.codomain == sp.S.Reals
        assert s.is_real == True
        assert s.is_complex == False

    def test_Signal_free_symbols(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", integer=True)

        s = Signal(2.0, t)
        assert s.free_symbols == []

        s = Signal(2*sp.I, n)
        assert s.free_symbols == []

        s = Signal(2 * n, n)
        assert s.free_symbols == []

        s = Signal("2**t", x, codomain=sp.S.Reals)
        assert len(s.free_symbols) == 1

        s = Signal(sp.cos(x * n), n, codomain=sp.S.Reals)
        assert s.free_symbols == [x]

        s = Signal(sp.exp(sp.I * k * x * t), n, codomain=sp.S.Complexes)
        assert s.free_symbols == [k, t, x]

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.free_symbols == []

        s = Signal(sp.Function("x")(t))
        assert s.free_symbols == []

        s = Signal(sp.Function("s")(n, t), n)
        assert s.free_symbols == [t]

        s = Signal(sp.Function("s")(n), t)
        assert s.free_symbols == [n]
        # pylint: enable-msg=not-callable

        s = Signal(2 * n * t * x, t)
        assert s.free_symbols == [n, x]

        s = Signal(5 * n * t, n)
        assert s.free_symbols == [t]

    def test_Signal_eval_scalar(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", integer=True)

        s = Signal(2.0, t)
        assert s.eval(1.5) == 2.0
        assert s(2.5) == 2.0
        assert s[0] == 2.0
        with pytest.raises(ValueError):
            s(sp.I)

        s = Signal(2*sp.I, n)
        assert s.eval(2) == 2*sp.I
        assert s(3) == 2*sp.I
        assert s[4] == 2*sp.I
        with pytest.raises(ValueError):
            s(1.5)

        s = Signal(2 * n, n)
        assert s.eval(2) == 4
        assert s(3) == 6
        assert s[4] == 8
        with pytest.raises(ValueError):
            s(1.5)

        s = Signal("2**t", x, codomain=sp.S.Reals)
        assert s.eval(1.5) == sp.sympify("2**t")
        assert s(2.5) == sp.sympify("2**t")
        assert s.eval(1.5, {s.free_symbols.pop(): 4}) == 16
        assert s(2.5, 3) == 8
        assert s[4] == sp.sympify("2**t")
        with pytest.raises(ValueError):
            s(sp.I)

        s = Signal(sp.cos(x * n), n, codomain=sp.S.Reals)
        assert s.eval(2) == sp.cos(2 * x)
        assert s(3) == sp.cos(3 * x)
        assert s.eval(2, {x: 4}) == sp.cos(8)
        assert s(3, 3) == sp.cos(9)
        assert s[4] == sp.cos(4 * x)
        with pytest.raises(ValueError):
            s(1.5)

        s = Signal(sp.exp(sp.I * k * x * t), n, codomain=sp.S.Complexes)
        assert s.eval(2) == sp.exp(sp.I * k * x * t)
        assert s(3) == sp.exp(sp.I * k * x * t)
        assert s.eval(2, {x: 4}) == sp.exp(4*sp.I * k * t)
        assert s(3, 3) == sp.exp(3*sp.I * x * t)
        assert s.eval(2, {x: 4, k: 3}) == sp.exp(12*sp.I * t)
        assert s(3, 4, 3) == sp.exp(12*sp.I * x)
        assert s[4] == sp.exp(sp.I * k * x * t)
        with pytest.raises(ValueError):
            s(1.5)

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.eval(2) == sp.sympify("X(2)")
        assert s(-1) == sp.sympify("X(-1)")
        assert s[0] == sp.sympify("X(0)")
        with pytest.raises(ValueError):
            s(1.5)

        s = Signal(sp.Function("x")(t))
        assert s.eval(1.5) == sp.sympify("x(1.5)")
        assert s(-2.5) == sp.sympify("x(-2.5)")
        assert s[-1] == sp.sympify("x(-1)")
        with pytest.raises(ValueError):
            s(sp.I)

        s = Signal(sp.Function("s")(n, t), n)
        assert s.eval(2) == s(2, t)
        assert s.eval(2, {t: 3}) == s(2, 3)
        assert s.eval(-1, {t: 4}) == s(-1, 4)
        assert s[-1] == s(-1, t)
        with pytest.raises(ValueError):
            s(1.5)

        s = Signal(sp.Function("s")(n), t)
        assert s(-2.5) == sp.Function("s")(n)
        assert s.eval(1.5, {t: 3}) == sp.Function("s")(n)
        assert s(-2.5, 3) == sp.Function("s")(3)
        assert s[0] == sp.Function("s")(n)
        with pytest.raises(ValueError):
            s(sp.I)
        # pylint: enable-msg=not-callable

        s = Signal(2 * n * t * x, t)
        assert s.eval(1.5) == 3.0 * n * x
        assert s(2.5) == 5.0 * n * x
        assert s.eval(1.5, {x: 4}) == 12.0 * n
        assert s(2.5, 3) == 15.0 * x
        assert s.eval(1.5, {x: 4, n: 3}) == 36.0
        assert s(2.5, 4, 3) == 60
        assert s[4] == 8 * n * x
        with pytest.raises(ValueError):
            s(sp.I)

        s = Signal(5 * n * t, n)
        assert s.eval(2) == 10 * t
        assert s(3) == 15 * t
        assert s.eval(2, {t: 2.5}) == 25.0
        assert s(3, 3.1) == 46.5
        assert s.eval(2, {t: -1.5}) == -15.0
        assert s(3, 4) == 60
        assert s[4] == 20 * t
        with pytest.raises(ValueError):
            s(1.5)

    def test_Signal_eval_range(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        x = sp.Symbol("x", real=True)
        k = sp.Symbol("k", integer=True)

        s = Signal(2.0, t)
        assert s(range(0, 3)) == [2.0] * 3
        assert s.eval(np.linspace(1, 2, 3)) == [2.0] * 3
        assert s[1:3] == [2.0] * 2
        assert s([1, 2]) == [2.0] * 2
        with pytest.raises(ValueError):
            s([[1, 2], [3, 4]])

        s = Signal(2*sp.I, n)
        assert s(range(0, 3)) == [2*sp.I] * 3
        assert s.eval(np.linspace(1, 3, 3)) == [2*sp.I] * 3
        assert s[1:3] == [2 * sp.I] * 2
        assert s([1, 2]) == [2 * sp.I] * 2
        with pytest.raises(ValueError):
            s.eval(np.linspace(1, 3, 4))

        s = Signal(2 * n, n)
        assert s.eval(range(0, 3)) == [0, 2, 4]
        assert s.eval(np.linspace(1, 3, 3)) == [2, 4, 6]
        assert s[0:3] == [0, 2, 4]
        assert s([0, 1, 2]) == [0, 2, 4]
        with pytest.raises(ValueError):
            s.eval(np.linspace(1, 3, 4))

        s = Signal("2**t", x, codomain=sp.S.Reals)
        assert s.eval(range(0, 3)) == [sp.sympify("2**t")] * 3
        assert s.eval(range(0, 3), {s.free_symbols.pop(): 2}) == [4] * 3
        assert s[0:3, 3] == [8] * 3
        assert s([0, 1, 2], 3) == [8] * 3
        with pytest.raises(ValueError):
            s([[1, 2], [3, 4]])

        s = Signal(sp.cos(x * n), n, codomain=sp.S.Reals)
        assert s.eval(range(-1, 2)) == [sp.cos(-x), 1, sp.cos(x)]
        assert s.eval(range(-1, 2), {x: 4}) == [sp.cos(4), 1, sp.cos(4)]
        assert s[0:5:2] == [1, sp.cos(2 * x), sp.cos(4 * x)]
        assert s([0, 2, 4]) == [1, sp.cos(2 * x), sp.cos(4 * x)]
        with pytest.raises(ValueError):
            s.eval(np.linspace(1, 2, 4))

        s = Signal(sp.exp(sp.I * k * x * t), n, codomain=sp.S.Complexes)
        assert s.eval(range(-1, 2)) == [sp.exp(sp.I * k * x * t)] * 3
        assert s.eval(range(-1, 2), {x: 4}) == [sp.exp(4*sp.I * k * t)] * 3
        assert s(range(-1, 2), 3) == [sp.exp(3*sp.I * x * t)] * 3
        assert s.eval(range(-1, 2), {x: 4, k: 3}) == [sp.exp(12*sp.I * t)] * 3
        assert s(range(-1, 2), 4, 3) == [sp.exp(12*sp.I * x)] * 3
        assert s[0:5:2] == [sp.exp(sp.I * k * x * t)] * 3
        assert s[1, 2, 3, 4] == sp.exp(24*sp.I)
        with pytest.raises(ValueError):
            s.eval(np.linspace(1, 2, 4))

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.eval(range(-1, 2)) == sp.sympify(["X(-1)", "X(0)", "X(1)"])
        assert s(range(-1, 2)) == sp.sympify(["X(-1)", "X(0)", "X(1)"])
        assert s[1:3] == sp.sympify(["X(1)", "X(2)"])
        with pytest.raises(ValueError):
            s.eval(np.linspace(1, 2, 4))

        s = Signal(sp.Function("x")(t))
        assert s.eval(range(-1, 2)) == sp.sympify(["x(-1)", "x(0)", "x(1)"])
        assert s(np.linspace(1, 2, 3)) == sp.sympify(["x(1)", "x(1.5)", "x(2)"])
        assert s[1:3] == sp.sympify(["x(1)", "x(2)"])

        s = Signal(sp.Function("s")(n, t), n)
        assert s([-1, 2, 0]) == [s(-1, t), s(2, t), s(0, t)]
        assert s.eval((-1, 0, 1), {t: 3}) == [s(-1, 3), s(0, 3), s(1, 3)]
        assert s.eval(range(-1, 2), {t: 4}) == [s(-1, 4), s(0, 4), s(1, 4)]
        assert s[-1:2] == [s(-1, t), s(0, t), s(1, t)]
        assert s[-1:2, 1] == [s(-1, 1), s(0, 1), s(1, 1)]
        with pytest.raises(ValueError):
            s((1, 1.5))

        s = Signal(sp.Function("s")(n), t)
        assert s(range(-1, 2)) == [sp.Function("s")(n)] * 3
        assert s.eval((1.5, 2.5), {t: 3}) == [sp.Function("s")(n)] * 2
        assert s([-1, 0, 1], 3) == [sp.Function("s")(3)] * 3
        assert s[-1:2] == [sp.Function("s")(n)] * 3
        assert s[-1:2, 3] == [sp.Function("s")(3)] * 3
        # pylint: enable-msg=not-callable

        s = Signal(2 * n * t * x, t)
        assert s(range(-1, 2)) == [-2 * n * x, 0, 2 * n * x]
        assert s.eval((1.5, -3), {x: 4}) == [12.0 * n, -24 * n]
        assert s(np.arange(1.0, 2.0, 0.5), 3) == [6 * x, 9.0 * x]
        assert s.eval((1.0, 2.0), {x: 4, n: 3}) == [24, 48]
        assert s((1, 2, 3), 4, 3) == [24, 48, 72]
        assert s[-1:4:2] == [-2 * n * x, 2 * n * x, 6 * n * x]
        assert s[-1:2, 1, 2, 3] == [-4, 0, 4]

        s = Signal(5 * n * t, n)
        assert s.eval(range(-1, 2)) == [-5 * t, 0, 5 * t]
        assert s((-1, 0, 1)) == [-5 * t, 0, 5 * t]
        assert s.eval(range(-1, 2), {t: 2}) == [-10.0, 0, 10.0]
        assert s(range(-1, 2), 30) == [-150, 0, 150]
        assert s[-1:3, 4] == [-20, 0, 20, 40]

    def test_Signal_periodic(self):
        N = sp.Symbol("N", integer=True)
        n = sp.Symbol("n", integer=True)
        x = sp.Symbol("x", real=True)
        t = sp.Symbol("t", real=True)
        omega = sp.Symbol("omega", real=True)

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("r", period=N)(x))
        assert s.is_periodic == True
        assert s.period == N

        s = Signal(sp.Function("w", period=N)(t))
        assert s.is_periodic == True
        assert s.period == N

        s = Signal(sp.Function("X", period=2 * sp.S.Pi)(omega))
        assert s.is_periodic == True
        assert s.period == 2 * sp.S.Pi
        # pylint: enable-msg=not-callable

        s = Signal(x ** 2)
        assert s.is_periodic == False
        assert s.period == None

        s = Signal(sp.cos(x))
        assert s.is_periodic == True
        assert s.period == 2 * sp.S.Pi

        s = Signal(sp.cos(x - 1))
        assert s.is_periodic == True
        assert s.period == 2 * sp.S.Pi

        s = Signal(sp.cos(2 * sp.S.Pi * 100 * t - sp.S.Pi / 4))
        assert s.is_periodic == True
        assert s.period == sp.Rational(1, 100)

        s = Signal(sp.cos(2 * sp.S.Pi * 100 * t - sp.S.Pi / 2))
        assert s.is_periodic == True
        assert s.period == sp.Rational(1, 100)

        s = Signal(sp.cos(x - sp.S.Pi))
        assert s.is_periodic == True
        assert s.period == 2 * sp.S.Pi

        s = Signal(sp.exp(sp.I * sp.S.Pi * n / 4), codomain=sp.S.Complexes)
        assert s.is_periodic == True
        assert s.period == 8

        # sympy can't do this because of gotchas
        s = Signal(sp.exp(1j * sp.S.Pi * n * 3 / 4), codomain=sp.S.Complexes)
        assert s.is_periodic == False

        # but this works
        s = Signal(sp.exp(sp.I * sp.S.Pi * n * 3 / 4), codomain=sp.S.Complexes)
        assert s.is_periodic == True
        assert s.period == 8

        # and this
        s = Signal(sp.exp(sp.I * sp.S.Pi * n / 4), codomain=sp.S.Complexes)
        assert s.is_periodic == True
        assert s.period == 8

        # better use I and Rationals
        s = Signal(
            sp.exp(sp.I * sp.S.Pi * n * sp.Rational(3, 5)), codomain=sp.S.Complexes
        )
        assert s.is_periodic == True
        assert s.period == 10

        s = Signal(
            sp.exp(sp.I * sp.S.Pi * (n - 3) * sp.Rational(3, 5)),
            codomain=sp.S.Complexes,
        )
        assert s.is_periodic == True
        assert s.period == 10

        s = Signal(sp.cos(sp.S.Pi * n / 10))
        assert s.is_periodic == True
        assert s.period == 20

        s = Signal(sp.cos(sp.S.Pi * (n + 10) / 10))
        assert s.is_periodic == True
        assert s.period == 20

        s = Signal(sp.cos(3 * sp.S.Pi * n / 5))
        assert s.is_periodic == True
        assert s.period == 10

        s = Signal(sp.cos(3 * n / 5))
        assert s.is_periodic == False
        assert s.period == None

        s = Signal(3, n)
        assert s.is_periodic == True
        assert s.period == 1

        s = Signal(3, t)
        assert s.is_periodic == False
        assert s.period == None

    def test_Signal_real_imag_conj(self):
        k = sp.Symbol("k", integer=True)
        n = sp.Symbol("n", integer=True)
        z = sp.Symbol("z", complex=True)
        t = sp.Symbol("t", real=True)

        s = Signal(3, n)
        assert s.real_part == 3
        assert s.imag_part == 0
        assert s.real == Signal(3, n)
        assert s.imag == Signal(0, n)
        assert s.conjugate == s

        s = Signal(3, t)
        assert s.real_part == 3
        assert s.imag_part == 0
        assert s.real == Signal(3, t)
        assert s.imag == Signal(0, t)
        assert s.conjugate == s

        s = Signal(sp.exp(sp.I * sp.S.Pi * n / 4), codomain=sp.S.Complexes)
        assert s.real_part == sp.cos(sp.S.Pi * n / 4)
        assert s.imag_part == sp.sin(sp.S.Pi * n / 4)
        assert s.real == Signal(sp.cos(sp.S.Pi * n / 4), n)
        assert s.imag == Signal(sp.sin(sp.S.Pi * n / 4), n)
        assert s.conjugate == Signal(
            sp.exp(-sp.I * sp.S.Pi * n / 4), codomain=sp.S.Complexes
        )

        s = Signal((1 / 3) ** n)
        assert s.real_part == (1 / 3) ** n
        assert s.imag_part == 0
        assert s.real == Signal((1 / 3) ** n)
        assert s.imag == Signal(0, n)
        assert s.conjugate == s

        s = Signal(z ** n, n)
        assert s.real_part == sp.re(z ** n)
        assert s.imag_part == sp.im(z ** n)
        assert s.real == Signal(sp.re(z ** n), n)
        assert s.imag == Signal(sp.im(z ** n), n)
        assert s.conjugate == Signal(sp.conjugate(z) ** n, n)

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.real_part == sp.re(s.amplitude)
        assert s.imag_part == sp.im(s.amplitude)
        assert s.real == Signal(sp.re(s.amplitude), k)
        assert s.imag == Signal(sp.im(s.amplitude), k)
        assert s.conjugate == Signal(sp.conjugate(sp.Function("X")(k)))
        # pylint: enable-msg=not-callable
 
    def test_Signal_delay_shift(self):
        k = sp.Symbol("k", integer=True)
        n = sp.Symbol("n", integer=True)
        z = sp.Symbol("z", complex=True)
        t = sp.Symbol("t", real=True)

        s = Signal(3, n)
        assert s.delay(3) == Signal(3, n)
        assert s.shift(-3) == Signal(3, n)
        assert s >> 3  == Signal(3, n)
        assert s << 3 == Signal(3, n)
        with pytest.raises(ValueError):
            s.delay(1.5)
        with pytest.raises(ValueError):
            s.shift(1.5)
        s = Signal(3, n)
        s >>= 3
        assert s == Signal(3, n)
        s <<= 3
        assert s == Signal(3, n)

        s = Signal(3, t)
        assert s.delay(3) == Signal(3, t)
        assert s.delay(1.5) == Signal(3, t)
        assert s >> 1.5 == Signal(3, t)
        assert s.shift(-3) == Signal(3, t)
        assert s.shift(-1.5) == Signal(3, t)
        assert s << 1.5 == Signal(3, t)
        s = Signal(3, t)
        s >>= 3
        assert s == Signal(3, t)
        s <<= 3
        assert s == Signal(3, t)

        s = Signal(
            sp.exp(sp.I * sp.S.Pi * n * sp.Rational(1, 4)), codomain=sp.S.Complexes
        )
        assert s.delay(3) == Signal(
            sp.exp(sp.I * sp.S.Pi * (n - 3) * sp.Rational(1, 4)),
            codomain=sp.S.Complexes,
        )
        assert s >> 3 == Signal(
            sp.exp(sp.I * sp.S.Pi * (n - 3) * sp.Rational(1, 4)),
            codomain=sp.S.Complexes,
        )
        assert s.shift(-3) == Signal(
            sp.exp(sp.I * sp.S.Pi * (n + 3) * sp.Rational(1, 4)),
            codomain=sp.S.Complexes,
        )
        assert s << 3 == Signal(
            sp.exp(sp.I * sp.S.Pi * (n + 3) * sp.Rational(1, 4)),
            codomain=sp.S.Complexes,
        )
        s = Signal(
            sp.exp(sp.I * sp.S.Pi * n * sp.Rational(1, 4)), codomain=sp.S.Complexes
        )
        s >>= 3
        assert s == Signal(sp.exp(sp.I * sp.S.Pi * (n - 3) * sp.Rational(1, 4)), codomain=sp.S.Complexes)
        s <<= 3
        assert s == Signal(sp.exp(sp.I * sp.S.Pi * n * sp.Rational(1, 4)), codomain=sp.S.Complexes)

        s = Signal((1 / 3) ** n)
        assert s.delay(3) == Signal((1 / 3) ** (n - 3))
        assert s >> 3 == Signal((1 / 3) ** (n - 3))
        assert s.shift(-3) == Signal((1 / 3) ** (n + 3))
        assert s << 3 == Signal((1 / 3) ** (n + 3))
        s = Signal((1 / 3)**n)
        s >>= 3
        assert s == Signal((1 / 3)**(n - 3))
        s <<= 3
        assert s == Signal((1 / 3)**n)

        # mejor
        s = Signal(sp.Rational(1, 3) ** n)
        assert s.delay(3) == Signal(sp.Rational(1, 3) ** (n - 3))
        assert s >> 3 == Signal(sp.Rational(1, 3) ** (n - 3))
        assert s.shift(-3) == Signal(sp.Rational(1, 3) ** (n + 3))
        assert s << 3 == Signal(sp.Rational(1, 3) ** (n + 3))
        s = Signal(sp.Rational(1, 3) ** n)
        s >>= 3
        assert s == Signal(sp.Rational(1, 3) ** (n - 3))
        s <<= 3
        assert s == Signal(sp.Rational(1, 3) ** n)

        s = Signal(z ** n, n)
        assert s.delay(3) == Signal(z ** (n - 3), n)
        assert s >> 3 == Signal(z ** (n - 3), n)
        assert s.shift(-3) == Signal(z ** (n + 3), n)
        assert s << 3 == Signal(z ** (n + 3), n)
        s = Signal(z ** n, n)
        s >>= 3
        assert s == Signal(z ** (n - 3), n)
        s <<= 3
        assert s == Signal(z ** n, n)

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.delay(3) == Signal(sp.Function("X")(k - 3), k)
        assert s >> 3 == Signal(sp.Function("X")(k - 3), k)
        assert s.shift(-3) == Signal(sp.Function("X")(k + 3), k)
        assert s << 3 == Signal(sp.Function("X")(k + 3), k)
        s = Signal(sp.Function("X")(k))
        s >>= 3
        assert s == Signal(sp.Function("X")(k - 3), k)
        s <<= 3
        assert s == Signal(sp.Function("X")(k), k)

        s = Signal(sp.Function("x")(t))
        assert s.delay(1.5) == Signal(sp.Function("x")(t - 1.5), t)
        assert s >> 1.5 == Signal(sp.Function("x")(t - 1.5), t)
        assert s.shift(-1.5) == Signal(sp.Function("x")(t + 1.5), t)
        assert s << 1.5 == Signal(sp.Function("x")(t + 1.5), t)
        s = Signal(sp.Function("x")(t))
        s >>= 1.5
        assert s == Signal(sp.Function("x")(t - 1.5), t)
        s <<= 1.5
        assert s == Signal(sp.Function("x")(t), t)
        # pylint: enable-msg=not-callable

    def test_Signal_flip(self):
        k = sp.Symbol("k", integer=True)
        n = sp.Symbol("n", integer=True)
        z = sp.Symbol("z", complex=True)
        t = sp.Symbol("t", real=True)

        s = Signal(3, n)
        assert s.flip() == Signal(3, n)

        s = Signal(3, t)
        assert s.flip() == Signal(3, t)

        s = Signal(sp.exp(sp.I * sp.S.Pi * n / 4), codomain=sp.S.Complexes)
        assert s.flip() == Signal(
            sp.exp(-sp.I * sp.S.Pi * n / 4), codomain=sp.S.Complexes
        )

        s = Signal((1 / 3) ** n)
        assert s.flip() == Signal((1 / 3) ** (-n))

        s = Signal(z ** n, n)
        assert s.flip() == Signal(z ** (-n), n)

        # pylint: disable-msg=not-callable
        s = Signal(sp.Function("X")(k))
        assert s.flip() == Signal(sp.Function("X")(-k), k)

        s = Signal(sp.Function("x")(t))
        assert s.flip() == Signal(sp.Function("x")(-t), t)
        # pylint: enable-msg=not-callable

    def test_Signal_support_duration(self):
        N = sp.Symbol("N", integer=True)
        n = sp.Symbol("n", integer=True)
        x = sp.Symbol("x", real=True)
        t = sp.Symbol("t", real=True)
        T = sp.Symbol("T", real=True)
        omega = sp.Symbol("omega", real=True)

        # pylint: disable-msg=not-callable
        with pytest.raises(ValueError):
            s = Signal(sp.Function("r", duration=T, period=N)(n))

        s = Signal(sp.Function("r", duration=T)(x))
        assert s.support == sp.Interval(0, T)
        assert s.duration == T

        s = Signal(sp.Function("r", duration=N)(n))
        assert s.support == sp.Range(0, N)
        assert s.duration == N

        s = Signal(sp.Function("w", period=T)(t))
        assert s.support == sp.S.Reals
        assert s.duration == None

        s = Signal(sp.Function("X", duration=2.5)(omega))
        assert s.support == sp.Interval(0, 2.5)
        assert s.duration == 2.5
        # pylint: enable-msg=not-callable

        s = Signal(x ** 2)
        assert s.support == sp.S.Reals
        assert s.duration == None

        s = Signal(sp.cos(x))
        assert s.support == sp.S.Reals
        assert s.duration == None

        s = Signal(sp.cos(x - 1))
        assert s.support == sp.S.Reals
        assert s.duration == None

        s = Signal(sp.cos(2 * sp.S.Pi * 100 * t - sp.S.Pi / 4))
        assert s.support == sp.S.Reals
        assert s.duration == None

        s = Signal(sp.cos(2 * sp.S.Pi * 100 * t - sp.S.Pi / 2))
        assert s.support == sp.S.Reals
        assert s.duration == None

        s = Signal(sp.cos(x - sp.S.Pi))
        assert s.support == sp.S.Reals
        assert s.duration == None

        s = Signal(sp.exp(sp.I * sp.S.Pi * n / 4), codomain=sp.S.Complexes)
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(sp.exp(sp.I * sp.S.Pi * n * 3 / 4), codomain=sp.S.Complexes)
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(
            sp.exp(sp.I * sp.S.Pi * n * sp.Rational(3, 5)), codomain=sp.S.Complexes
        )
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(
            sp.exp(sp.I * sp.S.Pi * (n - 3) * sp.Rational(3, 5)),
            codomain=sp.S.Complexes,
        )
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(sp.cos(sp.S.Pi * n / 10))
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(sp.cos(sp.S.Pi * (n + 10) / 10))
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(sp.cos(3 * sp.S.Pi * n / 5))
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(sp.cos(3 * n / 5))
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(3, n)
        assert s.support == sp.S.Integers
        assert s.duration == None

        s = Signal(3, t)
        assert s.support == sp.S.Reals
        assert s.duration == None

    def test_Signal_equality(self):
        n = sp.Symbol("n", integer=True)
        t = sp.Symbol("t", real=True)
        u = sp.Symbol("u", real=True)

        s1 = Signal(sp.cos(sp.S.Pi * n * sp.Rational(1, 4)))
        s2 = s1.copy()
        assert s1 == s2

        s1 = Signal(sp.cos(sp.S.Pi * n * sp.Rational(1, 4)))
        s2 = Signal(sp.cos(sp.S.Pi * t * sp.Rational(1, 4)))
        s3 = s1.clone(s1.__class__, s1.amplitude, iv=t)
        assert s2 != s3  # different domains
        s3 = s1.clone(s1.__class__, s1.amplitude, iv=t, domain=sp.S.Reals)
        assert s2 != s3  # different free_symbol
        s3 = s1.clone(s1.__class__, s1.amplitude.subs({n: t}), iv=t, domain=sp.S.Reals)
        assert s2 == s3

        s1 = Signal(sp.cos(sp.S.Pi * 100.5 * t))
        s2 = s1.copy()
        assert s1 == s2

        s1 = Signal(sp.cos(sp.S.Pi * 100.5 * t))
        s2 = Signal(sp.cos(sp.S.Pi * 100.5 * u))
        assert s1.amplitude != s2.amplitude  # sympy expr comparison
        assert s1 == s2

        s1 = Signal(sp.cos(sp.S.Pi * 100.5 * t))
        s2 = Signal(sp.cos(sp.S.Pi * 100.5 * n))
        assert s1 != s2

        # pylint: disable-msg=not-callable
        s1 = Signal(sp.Function("x")(n))
        s2 = Signal(sp.Function("x")(n))
        assert s1 == s2

        s1 = Signal(sp.Function("x")(n))
        s2 = Signal(sp.Function("y")(n))
        assert s1 != s2
        # pylint: enable-msg=not-callable
