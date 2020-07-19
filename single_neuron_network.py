import numpy as np

def cost(y, t):
	return np.power(y-t, 2)

def sigmoid(y):
	return 1.0 / (1 + np.exp(-1 * y))

def sigmoid_deriv(y):
	return sigmoid(y) * (1.0 - sigmoid(y))

def err_deriv(y, t):
	return 2 * (y-t)

def main():
	print('********Gradient Descent********')

	w = np.random.rand()
	b = np.random.rand()
	x = 0.3
	target = 0.5
	learning_rate = 0.01


	for i in range(100000):

		#Feed Forwad
		y = w * x + b
		z = sigmoid(y)
		c = cost(z, target)
		print(z)

		#Back propagation
		dw = err_deriv(z, target) * sigmoid_deriv(z) * x
		db = err_deriv(z, target) * sigmoid_deriv(z)
		w = w - dw*learning_rate
		b = b - db*learning_rate



	


if __name__ =='__main__':
	main()
