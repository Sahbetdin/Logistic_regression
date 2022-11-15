
from random import sample, random, seed 

class Splitting:
	def __init__(self, n, test_ratio, random_state=None, shuffle=False):

		self.test_ratio = test_ratio
		self.random_state = random_state
		self.shuffle = shuffle
		self.n = n
		# assert len(y) == X.shape[0], "Splitting: X and y should have equal sample lengths"
		self.split_indices()
		# return self.split_dataset(X, y)

	def split_indices(self): 
		#split to train and test datasets
		assert 0.1 <= self.test_ratio <= 0.3, "Adjust 0.1 <= test_ratio <= 0.3"
		self.n_test = int(self.n * self.test_ratio)
		self.n_train = self.n - self.n_test
		seed(self.random_state)
		if self.shuffle and self.random_state is not None:
			p = sample(range(self.n))
			self.idx_train = p[: self.n_train+1]
			self.idx_test = p[self.n_train+1 : ]
		elif self.shuffle and self.random_state is None:
			raise Exception("Split: Please set random_state to some number")
		else:
			p = range(self.n)
			self.idx_train =  p[:self.n_train+1]    #train indices
			self.idx_test = p[self.n_train:]
			# print("n: ",self.idx_train[0],self.idx_train[1])
			
	def split_dataset(self, X, y):
		X_train = X[self.idx_train]
		y_train = y[self.idx_train]
		X_test = X[self.idx_test]
		y_test = y[self.idx_test]
		return X_train, X_test, y_train, y_test
