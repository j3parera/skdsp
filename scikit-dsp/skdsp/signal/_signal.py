    # --- math wrappers -------------------------------------------------------
    # math functions must not arrive here, they must be previously catched
    def __add__(self, other):
        """ Signal addition:
        :math:`z[n] = x[n] + y[n]`,
        :math:`z(t) = x(t) + y(t)`.
        """
        raise NotImplementedError("({0}).__add__".format(self))

    __radd__ = __add__
    __iadd__ = __add__

    def __sub__(self, other):
        """ Signal substraction:
        :math:`z[n] = x[n] - y[n]`,
        :math:`z(t) = x(t) - y(t)`.
        """
        raise NotImplementedError("({0}).__sub__".format(self))

    __rsub__ = __sub__
    __isub__ = __sub__

    def __neg__(self):
        """ Signal sign inversion:
        :math:`y[n] = -x[n]`,
        :math:`y(t) = -x(t)`.
        """
        raise NotImplementedError("({0}).__neg__".format(self))

    def __mul__(self, other):
        """ Signal multiplication:
        :math:`z[n] = x[n] x y[n]`,
        :math:`z(t) = x(t) x y(t)`.
        """
        raise NotImplementedError("({0}).__mul__".format(self))

    __rmul__ = __mul__
    __imul__ = __mul__

    def __pow__(self, other):
        """ Signal exponentiation:
        :math:`z[n] = x[n]^{y[n]}`,
        :math:`z(t) = x(t)^{y(t)}`.
        """
        raise NotImplementedError("({0}).__pow__".format(self))

    def __truediv__(self, other):
        """ Signal division:
        :math:`z[n] = x[n] / y[n]`,
        :math:`z(t) = x(t) / y(t)`.
        """
        raise NotImplementedError("({0}).__truediv__".format(self))

    __rtruediv__ = __truediv__
    __itruediv__ = __truediv__

    def scale(self, v, mul=True):
        """
        Scales the the independent variable; i.e.
        :math:`y[n] = x[v*n]`,
        :math:`y(t) = x(v*t)`.

        Args:
            v: The amount of scaling.
            mul (bool): If True, the scale multiplies, else
                divides.

        Returns:
            A signal copy with the independent variable scaled.

        """
        s = self._clone()
        s._xexpr = ScaleOperator.apply(s._xvar, s._xexpr, v, mul)
        return s


class _FunctionSignal(_Signal):
    self._ylambda = None

    def eval(self, x):
        # Hay que ver si hay 'Pow'
        to_real = False
        pows = []
        for arg in sp.preorder_traversal(self._yexpr):
            if isinstance(arg, sp.Pow):
                pows.append(arg)
        for p in pows:
            base = p.args[0]
            if isinstance(base, (Number, sp.Number)):
                if base <= 0:
                    # base negativa, los exponentes deben ser complejos
                    # por si acaso no son enteros
                    x = x.astype(np.complex_)
                    self.dtype = np.complex_
                    to_real = True
                    # break # ??
        try:
            if self._ylambda is None:
                self._ylambda = sp.lambdify(self._xvar, self._yexpr, "numpy")
            y = self._ylambda(x)
            if not hasattr(y, "__len__"):
                # workaround para issue #5642 de sympy. Cuando yexpr es una
                # constante, se devuelve un escalar aunque la entrada sea un
                # array
                y = np.full(x.shape, y, self.dtype)
            if not to_real:
                y = y.astype(self.dtype)
        except (NameError, ValueError):
            # sympy no ha podido hacer una función lambda
            # (o hay algún problema de cálculo, p.e 2^(-1) enteros)
            # así que se procesan los valores uno a uno
            y = np.zeros_like(x, self.dtype)
            for k, x0 in enumerate(x):
                try:
                    y[k] = self._yexpr.xreplace({self._xvar: x0})
                except TypeError:
                    y[k] = np.nan
        if to_real:
            y = np.real_if_close(y)
        return y


