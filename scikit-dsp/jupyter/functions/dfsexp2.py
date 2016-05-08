def dfsexp2(N=16, k=1):
    #s = 0
    #for k0 in range(0, k):
    #    s += ds.Exponential(sp.exp(sp.I*sp.S.Pi*2*k0/N))
    #s = s/N
    s = S[k-1]
    n = np.arange(-50, 50)
    X = s.dfs(N, force=True)
    X.real[np.isclose(X.real, 0)] = 0
    X.imag[np.isclose(X.imag, 0)] = 0
    plt.figure(figsize=(20, 3))
    plt.subplot(121)
    plt.stem(n, np.real(s[n]), markerfmt='bo', linefmt='b-', basefmt='k-')
    plt.stem(n, np.imag(s[n]), markerfmt='ro', linefmt='r-', basefmt='k-')
    plt.axis([n[0]-0.5, n[-1]+0.5, -1.1, 1.1])
    plt.grid(True)
    plt.subplot(122)
    k = np.arange(0, N)
    plt.stem(k, np.abs(X))
    plt.axis([-0.5, 15.5, -0.1, 1.1])
    plt.axhline(0, color='k')
    plt.axvline(0, color='k')
    plt.grid(True)