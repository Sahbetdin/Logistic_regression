import matplotlib.pyplot as plt
import numpy as np
import pickle

class LogRegr:
	def __init__(self, eta=0.1, max_iter=50, random_state=42, early_stopping=False):
		self.eta = eta
		self.max_iter = max_iter
		self._w = None
		self.accuracy_tr = np.zeros(max_iter)
		self.accuracy_te = np.zeros(max_iter)
		self.cost_tr = np.zeros(max_iter)
		self.cost_te = np.zeros(max_iter)
		self.random_state = random_state
		self.early_stopping = early_stopping
		self.early_st = max_iter

	@staticmethod
	def sigmoid(z):
		assert type(z) == np.ndarray, "sigmoid: some \
type problems with z."
		z =  np.where(np.abs(z) < 10, z, np.where(z > 10, 10, -10))
		return 1.0 / (1.0 + np.exp(-z))

	def init_weights(self, size):
		np.random.seed(self.random_state)
		self._w = np.random.random(size)

	def get_costs(self, pred, y_true):
		assert pred.shape[0] == y_true.shape[0], "get_costs: \
lengths should be equal"
		fir = np.log(pred[range(y_true.shape[0]), y_true])
		sec = np.zeros(y_true.shape[0])
		for k in range(y_true.shape[0]):
			for i in range(pred.shape[1]):
				if i == y_true[k]:
					continue
				if pred[k,i] == 1:
					sec[k] += 20 #clipped for inf
				else:
					sec[k] += np.log(1 - pred[k,i])
		return -(fir + sec).mean()

	def update_weights(self, X, pr, y):
		pr[range(y.shape[0]), y] = pr[range(y.shape[0]), y] - 1
		self._w -= self.eta * pr.T.dot(X)
	
	@staticmethod
	def get_accuracy(pred, y_true):
		assert pred.shape[0] == y_true.shape[0], "get_accuracy: \
lengths should be equal"
		predicted_label = np.argmax(pred, axis=1)
		return (predicted_label == y_true).sum()/pred.shape[0]

	def train(self, X_tr, y_tr, X_te, y_te, y_max):
		"""
		for given X(features), y(targets) find optimal self._w
		with gradient descent
		y consists of values like [0,3,2,1,0,...], in this case has 4 classes
		"""
		self.init_weights((y_max, X_tr.shape[1]))
		for it in range(self.max_iter):
			pr_tr = self.predict(X_tr)
			pr_te = self.predict(X_te)
			self.cost_tr[it] = self.get_costs(pr_tr, y_tr)
			self.cost_te[it] = self.get_costs(pr_te, y_te)
			# print(it, self.cost_tr[it], self.cost_te[it])
			self.accuracy_tr[it] = self.get_accuracy(pr_tr, y_tr)
			self.accuracy_te[it] = self.get_accuracy(pr_te, y_te)
			self.update_weights(X_tr, pr_tr, y_tr)
			if self.early_stopping and it > 5:
				cost_pct_change = (self.cost_tr[it-1]-self.cost_tr[it])/self.cost_tr[it-1]
				if np.abs(cost_pct_change) < 0.001:
					print(f'Quited on {it}-th step with early stopping.')
					self.early_st = it
					return

	def predict(self, X):
		z = X.dot(self._w.T)
		y_pred_prob = self.sigmoid(z)
		return y_pred_prob
	
	def save_model(self, file_name):
		with open(file_name, 'wb') as file:
			pickle.dump(self._w, file)

	def load_model(self, file_name):
		with open(file_name, 'rb') as file:
			self._w = pickle.load(file)

	def plot_costs_accuracy(self):
		fig, (ax1, ax2) = plt.subplots(1, 2)
		ax1.plot(self.cost_tr[:self.early_st], color='b',label='Train costs')
		ax1.plot(self.cost_te[:self.early_st], color='r',label='Test costs')
		ax1.set_title('Costs during train stage')
		ax1.set_xlabel('n_epochs, #')
		ax1.set_ylabel('Costs, arb.units')

		ax2.plot(self.accuracy_tr[:self.early_st], color='b',label='Train accuracy')
		ax2.plot(self.accuracy_te[:self.early_st], color='r',label='Test accuracy')
		ax2.set_title('Accuracy during train stage')
		ax2.set_xlabel('n_epochs, #')
		ax2.set_ylabel('Accuracy, %')
		plt.legend()
		plt.tight_layout()
		plt.show()

