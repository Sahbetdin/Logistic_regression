import numpy as np
from used_functions import *

class MyStandardScaler(object):
	def __init__(self):
		self._mean = None
		self._std = None

	def fit(self, X):
		self._mean = np.zeros(X.shape[1])
		self._std = np.zeros(X.shape[1])
		for j in range(X.shape[1]):
			self._mean[j] = mean_(X[:,j])
			self._std[j] = std_(X[:,j], self._mean[j])

	def transform(self, X):			
		return (X - self._mean) / self._std
	
	def fillna_with_zero(self, X):
		for j in range(X.shape[1]):
			for i in range(X.shape[0]):
				if np.isnan(X[i,j]):
					# X[i,j] = self._mean[j]
					X[i,j] = 0
		return X

