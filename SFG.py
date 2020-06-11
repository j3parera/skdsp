def SFG(x, mem):
	y = []
	M1 = mem[1]
	M2 = mem[2]
	for n in range(len(x)):
		X = x[n]
		W2 = M2
		B2 = W2
		A2 = 1/4*W2
		W1 = M1
		B1 = 1/8*W1 + B2
		A1 = 1/8*W1 + A2
		A0 = X + A1
		W0 = A0
		B0 = 1/4*W0 + B1
		Y = B0
		y[n] = Y
		M1 = W0
		M2 = W1
	return y
