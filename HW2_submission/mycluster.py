import numpy as np

def cluster(T, K, num_iters = 1000, epsilon = 1e-12):
	"""

	:param bow:
		bag-of-word matrix of (num_doc, V), where V is the vocabulary size
	:param K:
		number of topics
	:return:
		idx of size (num_doc), idx should be 1, 2, 3 or 4
	"""
	

	n_d = T.shape[0]
	n_w = T.shape[1]
	n_c = K

	# initialize
	pi = np.random.rand(n_c)
	pi /= np.sum(pi)
	mu = np.random.rand(n_w, n_c)

	for _ in range(num_iters):
		# E step
		gamma = np.zeros((n_d, n_c))

		for i in range(n_d):
			d = 0
			for c in range(n_c):
				d += pi[c] * np.product(np.power(mu[:, c], T[i, :]))
			
			for c in range(n_c):
				gamma[i][c] = pi[c] * np.product(np.power(mu[:, c], T[i, :])) / d

		# M step
		X = np.matmul(gamma.T, T)

		for c in range(n_c):
			for j in range(n_w):
				mu[j][c] = X[c][j] / np.sum(X[c, :])

		pi = np.sum(gamma, axis = 0)

	idx = np.argmax(gamma, axis=1)
	idx += 1
	# raise NotImplementedError
	return idx





