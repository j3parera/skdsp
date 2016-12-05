@path/home/pepo/git/skdsp/scikit-dsp/docs/

Cosas
=====

-  *Simbólico*. Parece que va ser muy difícil, sino imposible, hacer un paquete simbólico. Ni Sympy ni Matlab consideran la posibilidad, al menos en este momento, de funciones con soporte discreto; de hecho tienen incluso problemas para gestionar los soportes no enteros de la recta real. Se puede intentar un esquema mixto simbólico-numérico, más numérico que simbólico, pero que tenga en cuenta, aunque sea de manera lateral, a éste último.

   -  Esta solución va funcionando bastante bien

   -  ¿Merece la pena mantener una expresión simbólica paramétrica de una señal o sistema (además de la expresión simbólica para cálculo), p.ej en una sinusoide :math:`A\cos[\omega_{0}n+\phi]`, ¿mantener esta expresión y la correspondiente con los valores substituidos?

-  ¿Sería interesante que cualquier parámetro pudiera ser una expresión? Por ejemplo :math:`\delta[n-\alpha],\alpha=f[n]`

-  http://python-textbok.readthedocs.org/en/latest/index.html

-  Empieza a haber una estructura más o menos clara, con algunos problemas que van surgiendo (con sympy o con mi ignorancia), algunos evitables y otros no. Por ejemplo, sympy no sabe calcular si una función es o no periódica (mathematica sí, desde v10) y si se intenta resolver mediante: ``solve(cos(x)-cos(x+T), T)`` (o ``solveset``), aparecen 2 resultados, uno bueno y otro raro; con ``sin(x)`` directamente la caga. Se pueden buscar workarounds, por ejemplo, indicar que ``period = None``, significa que no se sabe, ``period = Inf``, que no es periódica y ``period = cte`` que sí es periódica de periodo ``cte``.

-  Cuidado, el ``upcast`` a pelo no funciona como se esperaba, la función ``_demote_xxx`` no hace lo que se pretende.

Clase Señal
===========

Atributos
---------

-  data: numpy.array para los datos de la señal (si definida por datos) o una cache (cuando se define por una función)

-  range: python range para los datos de la señal

-  real, imag

Métodos
-------

-  Para algunos métodos hay funciones cuyo primer parámetro es la señal

-  Métodos numpy.ndarray interesantes:

   -  copy copia sin cast

   -  min, max, ptp (peak-to-peak), clip, conj, round, sum, cumsum, prod, cumprod, std, var, mean

-  Operadores como funciones:

   -  operadores unitarios: ``__pos__``, ``__abs__``

      -  ``__neg__``

   -  operadores binarios: ``__(i)pow__``

      -  ``__add__``, ``__iadd__``, ``__radd__``

      -  ``__sub__``, ``__isub__``, ``__rsub__``

      -  ``__mul__``, ``__imul__``, ``__rmul__``

      -  ``__lshift__``, ``__ilshift__``

      -  ``__rshift__``, ``__irshift__``

      -  ``__truediv__``, ``__itruediv__``

      -  ``__floordiv__``, ``__ifloordiv__``

   -  indexado: ``__len__, __setitem__``

   -  indexado con ``__getitem__`` funciona de dos formas:

      -  ``s[x0]``, con ``x0`` escalar, devuelve el valor escalar de la señal en ``x0``

      -  ``s[start:stop:step]`` (o ``s[x0]`` con ``x0 slice``), devuelve un array con los valores de la señal en el rango ``[start:stop:step]`` (es decir, en el intervalo ``[start, stop)`` con saltos de ``step``) sin interpretar valores negativos como índices desde el final de la señal. El valor predeterminado para ``step`` es 1; no existen valores predeterminados para ``start`` y ``stop``. ``step`` puede tomar valores negativos.

   -  representación:

      -  ``__str__``

      -  ``__repr__``

Funciones
---------

-  Además de las correspondientes a los métodos

-  median, correlate, average (con pesos),

Otras cosas
-----------

-  Las señales son vectores de un espacio de Hilbert

-  Una señal es (o tiene):

   -  una función en el dominio del tiempo x(t) o x(n) (continua/discreta)

   -  una función en el dominio de la frecuencia X(Omega) o X(omega)

   -  una función en el dominio complejo X(s) o X(z)

-  Como función

   -  existe una expresión para evaluar valores (en cualquier dominio) con o sin parámetros u(n), delta(n), cos(omega0\*n+phi), ...

   -  o los valores están dados por datos (no debería valer para señales continuas)

   -  soporte -oo a oo, en principio

   -  la longitud puede estar limitada (valores en los que es no-nula, sin ceros en medio) = duración ... LENGTH(x)

   -  Se debería poder (si procede, según existencia, convergencia, etc) pasar una función a otra:

      -  x(n) a X(omega) a x(n)

      -  x(n) a X(z) a x(n)

      -  X(z) a X(omega) a X(z)

   -  Podrían existir expansiones en serie de bases natural, ...

   -  Las señales tienen propiedades: (true/false)

      -  real, par, impar, ...

      -  periódica ...

      -  ``isCOLA(R)``

-  Podría usarse ``int()``, ``float()``, ``complex()`` para cuantificar, y convertir a real/compleja

Operadores
----------

-  Cada operador tiene una definición en un dominio discreto o continuo.

-  Un operador en el dominio del tiempo se corresponde con otro operador en el dominio transformado (y viceversa). Son los teoremas que podrían recogerse en una tabla.

-  Los operadores pueden escribirse como funciones o como métodos; en cualquier caso el resultado es una señal distinta:

   Por ejemplo, si ``x`` es una señal se puede escribir:

   -  ``y = x.flip().shift(2)``, que produce ``s1 = x.flip(); y = s1.shift(2)``, con :math:`s1=x[-n]` y :math:`y=s1[n-2]=x[-n+2]` que podría escribirse como ``y = sg.shift(sg.flip(x), 2)`` como funciones

   -  ``y = x.shift(2).flip()``, que produce ``s1 = x.shift(2); y = s1.flip()``, con :math:`s1=x[n-2]` y :math:`y=s1[-n]=x[-n-2]` que podría escribirse como ``y = sg.flip(sg.shift(x, 2))`` como funciones

   Nótese que como métodos se calculan de izquierda a derecha y como funciones en la forma habitual impuesta por los paréntesis.

   Al choricillo ``x.a().b().c()...`` parece que se le llama *fluent interface*.

-  La expresión como métodos permite escribir cosas como:

   -  ``x = 3*Delta().shift(-2) + 2*(Delta() << 1) + (2+2j)*Delta().shift(2) + 3*(Delta() >> 4)``, para formar la señal que responde a la expresión :math:`x[n]=3\delta[n+2]+2\delta[n+1]+(2+2{\rm {j}})\delta[n-2]+3\delta[n-4]`.

   -  | Como ``Delta`` y ``Step`` se usan mucho desplazadas, se puede pasar el retardo como parámetro al constructor:
      | ``x = 3*Delta(-2) + 2*Delta(-1) + (2+2j)*Delta(2) + 3*Delta(4)``, es :math:`x[n]=3\delta[n+2]+2\delta[n+1]+(2+2{\rm {j}})\delta[n-2]+3\delta[n-4]`.

   -  Nótese que el desplazamiento :math:`x[n-k]` se indica con ``x.shift(k)`` o ``x >> k``

-  Como funciones: el primer operando (si el operador es unitario) o los dos primeros (si es binario) son las señales, el resto parámetros.

Unitarios ’reduce’: x(n) → número
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  norma 1, 2, oo, genérica (p)

   -  energía

   -  potencia

   -  max, min

   -  peak-to-peak: max(x)-min(x), numpy.ptp

Unitarios ’map’: x(n) → y(n)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operadores que cambian la variable independiente:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  pueden interpretarse como :math:`x(\pm|\alpha|t+t_{0})`

-  si :math:`|\alpha|<1` es una expansión; si :math:`|\alpha|>1` es una compresión

-  si :math:`\alpha<0` es, además, una reflexión

-  si las señales son discretas, :math:`\alpha` debe ser entero en la compresión y :math:`1/\alpha` entero en la expansión; en ambos casos mayores que 0

-  :math:`{\rm {flip}}(x[n])=x[-n]`, :math:`{\rm {flip}}(x(t))=x(-t)`

-  :math:`{\rm {compress}}(x[n],\alpha)=x[\alpha n]`; si :math:`\alpha` no entero, excepción, :math:`{\rm {compress}}(x(t),\alpha)=x(\alpha t)`

-  :math:`{\rm {expand}}(x[n],\beta)=x[n/\beta]`; si :math:`\beta` no entero, excepción, :math:`{\rm {expand}}(x(t),\beta)=x(t/\beta)`

-  :math:`{\rm {shift}}(x[n],k)=x[n-k]`; si :math:`k` no entero, excepción, :math:`{\rm {shift}}(x(t),t_{0})=x(t-t_{0})`

Otros
^^^^^

-  producto por escalar: :math:`ax[n]`

-  desplazamiento circular: :math:`{\rm {cshift}}(x[n],n_{0},N)=x((n-n0)_{N})`

-  conjugación: :math:`{\rm {conj}(x[n])=x^{*}[n]}`, :math:`{\rm {conj}}(x(t))=x^{*}(t)`

-  potencia instantánea

-  otras funciones no lineales:

   -  valor absoluto: :math:`|x(n)|`

   -  cuadrado: :math:`x^{2}(n)`

   -  raíz enésima: :math:`x(n)^{(1/n)}`

-  derivada: dx(t)/dt o diferencia: x(n)-x(n-1)

-  integral: int\_oo^t x(t) o acumulación: sum\_oo^n x(n)

Transformadas
^^^^^^^^^^^^^

-  Serie de Fourier continua: FS(x, omega, T)

-  Serie de Fourier discreta: DFS(x, k, N)

-  Transformada de Fourier en tiempo discreto: DTFT(x, omega), IDTFT(X, n)

-  Transformada de Fourier discreta: DFT(x, k, N), IDFT(X, n, N)

-  Transformada de Fourier rápida: FFT(x, k, N), IFFT(X, n, N)

Binarios ’map’: x(n) op y(n) → z(n)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  suma: :math:`x(n)+y(n)`

-  resta: :math:`x(n)-y(n)`

-  multiplicación: :math:`x(n)\times y(n)`

-  división: :math:`x(n)/y(n)`

-  convolución: :math:`x(n)*y(n)`: CONV(x, y) = DOT(x, SHIFT(FLIP(h), n))

-  convolución circular: x(n) \*N y(n): CCONV(x, y, N) = DOT(x, CSHIFT(FLIP(h), n, N)

Binarios ’reduce’:
~~~~~~~~~~~~~~~~~~

-  producto escalar: <x(n), y(n)>: DOT(x, y)

Creación de Señales
===================

Basadas en Funciones
--------------------

-  delta(), step, sin, cos, exp ..., quizá con nombres típicos, aunque sobrecarguen

-  np.fromfunction: NO, porque sólo acepta coordenadas del array

-  piecewise: puede ser, p.e. para evaluar u[n] = np.piecewise(n, [n < 0, n >= 0], [lambda n: 0.0, lambda n: 1.0]), hay que guardar la lista de condiciones y de funciones

Basadas en Datos
----------------

Señales
=======

Diente de sierra
----------------

Copiado de matlab para que valga para hacer ondas dientes de sierra y triangulares

2 parámetros, N (número de muestras por periodo), width (<= N, anchura)

La señal empieza en -1 y sube hasta 1- en width-1, (1 exacto en width), después baja desde 1 (en width) a -1+ (-1 en N)

Ecuaciones:

.. math::

   saw[n,N,w]=\begin{cases}
   -1+\frac{{2m}}{w} & 0\le m<w\\
   1-2\frac{(m-w)}{(N-w)} & w\le m<N
   \end{cases}\quad m=((n))_{N}


