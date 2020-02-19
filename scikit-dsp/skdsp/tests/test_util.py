import sympy as sp
from skdsp.signal.util import complex2polar


class Test_Complex2Polar(object):
    def test_rectangular(self):
        z = 3 + 3*sp.I
        (rr, oo) = complex2polar(z)
        assert rr == sp.sqrt(18)
        assert oo == sp.S.Pi / 4

        z = -2 + 0.2*sp.I
        (rr, oo) = complex2polar(z)
        assert rr == sp.sqrt(sp.re(z)**2 + sp.im(z)**2)
        assert oo == sp.atan2(sp.im(z), sp.re(z))

        z = sp.S(3.2)
        (rr, oo) = complex2polar(z)
        assert rr == sp.S(3.2)
        assert oo == sp.S.Zero

        z = sp.S(-3.2)
        (rr, oo) = complex2polar(z)
        assert rr == sp.S(3.2)
        assert oo == sp.S.Pi

        z = 3.2*sp.I
        (rr, oo) = complex2polar(z)
        assert rr == sp.S(3.2)
        assert oo == sp.S.Pi / 2

        z = -3.2*sp.I
        (rr, oo) = complex2polar(z)
        assert rr == sp.S(3.2)
        assert oo == -sp.S.Pi / 2

        z = sp.S.Zero
        (rr, oo) = complex2polar(z)
        assert rr == sp.S.Zero
        assert oo is None

    def test_polar(self):
        o = sp.S.Pi / 10
        r = sp.Float(2.5)

        z = -r * sp.exp(sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == o + sp.S.Pi

        z = sp.exp(sp.I * o) * (-r)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == o + sp.S.Pi

        z = r * sp.exp(sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == o

        z = sp.exp(sp.I * o) * r
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == o

        z = sp.exp(sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == 1
        assert oo == o

        z = -r * sp.exp(-sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == -o + sp.S.Pi

        z = sp.exp(-sp.I * o) * (-r)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == -o + sp.S.Pi

        z = r * sp.exp(-sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == -o

        z = sp.exp(-sp.I * o) * r
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == -o

        z = sp.exp(-sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == 1
        assert oo == -o

        o = sp.S.Pi / 4
        r = 2

        z = -r * sp.exp(sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == o + sp.S.Pi

        z = sp.exp(sp.I * o) * (-r)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == o + sp.S.Pi

        z = r * sp.exp(sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == o

        z = sp.exp(sp.I * o) * r
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == o

        z = sp.exp(sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == 1
        assert oo == o

        z = -r * sp.exp(-sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == -o + sp.S.Pi

        z = sp.exp(-sp.I * o) * (-r)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == -o + sp.S.Pi

        z = r * sp.exp(-sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == -o

        z = sp.exp(-sp.I * o) * r
        (rr, oo) = complex2polar(z)
        assert rr == r
        assert oo == -o

        z = sp.exp(-sp.I * o)
        (rr, oo) = complex2polar(z)
        assert rr == 1
        assert oo == -o
