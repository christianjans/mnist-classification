import numpy as np


class CJNeuralNetwork:

	# initialize the neural network
	def __init__(self, net_map, learning_rate):
		self.net_map = np.matrix(net_map)
		self.learning_rate = learning_rate

		self.weights = []
		self.biases = []

		# initialize the weights and biases with random numbers
		for i in range(max(np.shape(net_map)) - 1):
			this_layer_nodes = self.net_map[0, i]
			next_layer_nodes = self.net_map[0, (i + 1)]

			self.weights.append(np.matrix(np.random.uniform(low = -1, high = 1, size = (next_layer_nodes, this_layer_nodes))))
			self.biases.append(np.matrix(np.random.uniform(low = -1, high = 1, size = (next_layer_nodes, 1))))

	# feed forward the input vector
	def feed_forward(self, inputs):
		self.layer_vals = []
		self.layer_zs = []

		input_mat = np.matrix(inputs)

		self.layer_vals.append(input_mat.reshape(max(np.shape(input_mat)), 1))
		self.layer_zs.append(self.layer_vals[0])
		last_index = len(self.weights) - 1

		for i in range(last_index):
			self.layer_vals.append(self.__sigmoid(np.add(np.matmul(self.weights[i], self.layer_vals[i]), self.biases[i])))
			self.layer_zs.append(np.add(np.matmul(self.weights[i], self.layer_vals[i]), self.biases[i]))

		self.layer_vals.append(self.__softmax(np.add(np.matmul(self.weights[last_index], self.layer_vals[last_index]), self.biases[last_index])))
		self.layer_zs.append(np.add(np.matmul(self.weights[last_index], self.layer_vals[last_index]), self.biases[last_index]))

		return self.layer_vals[last_index + 1]

	# backpropagate the target vector and train the neural net using gradient descent
	def backpropagate(self, targets):
		targets_mat = np.matrix(targets)

		initial_targets = targets_mat.reshape((max(np.shape(targets_mat)), 1))
		last_layer = len(self.weights)

		initial_error = np.subtract(self.layer_vals[last_layer], initial_targets)
		delta = np.multiply(initial_error, self.__softmax_deriv(self.layer_zs[last_layer]))
		weight_change = np.matmul(delta, np.transpose(self.layer_vals[last_layer - 1]))

		for i in range(last_layer - 1, -1, -1):
			self.weights[i] = np.subtract(self.weights[i], self.learning_rate * weight_change)
			self.biases[i] = np.subtract(self.biases[i], self.learning_rate * delta)

			delta = np.multiply(np.matmul(np.transpose(self.weights[i]), delta), self.__sigmoid_deriv(self.layer_zs[i]))
			weight_change = np.matmul(delta, np.transpose(self.layer_vals[i - 1]))


	# sigmoid function
	def __sigmoid(self, val):
		return 1 / (1 + np.exp(-val))

	# sigmoid derivative function
	def __sigmoid_deriv(self, val):
		sig = self.__sigmoid(val)
		return np.multiply(sig, 1 - sig)

	# relu function
	def __relu(self, arr):
		arr[np.where(arr < 0)] = 0
		return arr

	# relu derivative function
	def __relu_deriv(self, arr):
		arr[np.where(arr <= 0)] = 0
		arr[np.where(arr != 0)] = 1
		return arr

	# softmax function
	def __softmax(self, val):
		column_exp_sum = np.sum(np.exp(val))
		return np.exp(val) / column_exp_sum

	# softmax derivative function
	def __softmax_deriv(self, val):
		soft = self.__softmax(val)
		return np.multiply(soft, 1 - soft)












		