from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(X):
	return 1/(1+np.exp(-X))

def relu(X):
	return np.maximum(X,0)

def xor_net(x1,x2,weights,activation):
	'''
	Return the prediction of the network given an activation function
	'''
	x = np.array([1, x1, x2])
	if activation == 'sigmoid':
		a1 = sigmoid(np.dot(weights[0,:],x))
		a2 = sigmoid(np.dot(weights[1,:],x))
		a = np.array([1, a1, a2])
		return sigmoid(np.dot(weights[2,:],a))
	if activation == 'tanh':
		a1 = np.tanh(np.dot(weights[0,:],x))
		a2 = np.tanh(np.dot(weights[1,:],x))
		a = np.array([1, a1, a2])
		return np.tanh(np.dot(weights[2,:],a))
	if activation == 'relu':
		a1 = relu(np.dot(weights[0,:],x))
		a2 = relu(np.dot(weights[1,:],x))
		a = np.array([1, a1, a2])
		return relu(np.dot(weights[2,:],a))

def mse(weights,activation):
	'''
	Compute the mean squared error of given weights
	'''
	x = np.array([[0,0],[0,1],[1,0],[1,1]])
	y = np.array([0,1,1,0])
	mse = 0
	for i in range(4):
		xor = xor_net(x[i,0],x[i,1],weights,activation)
		mse += (xor-y[i])**2
	return mse/4.

def grdmse(weights,activation):
	'''
	Compute the gradient of all the weights
	'''
	err = 0.001
	grads = np.zeros((3,3))
	weights_err = np.copy(weights)
	const_mse = mse(weights,activation)
	for i in range(3):
		for j in range(3):
			weights_err[i,j] += err
			grads[i,j] = (mse(weights_err,activation)-const_mse)/err
			weights_err[i,j] -= err
	return grads

def train_xor_net(train_x, train_y, learning_rate, activation='sigmoid'):
	'''
	Train the neural network on XOR data with a given learning rate and activation function
	''' 
	np.random.seed(42)
	w = np.random.randn(3,3)*0.5

	iterations = 0
	n_missclassified = 4
	mses = []
	n_correct = []
	#iterate until there no cases missclassified or until 10000 iterations are reached                                     
	while n_missclassified > 0:
		n_missclassified = 0
		w -= learning_rate*grdmse(w, activation)
		error = mse(w, activation)
		mses.append(error)
		for i in range(4):
			y_h = xor_net(train_x[i,0],train_x[i,1],w, activation)
			n_missclassified += abs(round(y_h)-train_y[i])
		n_correct.append(4-n_missclassified)
		iterations += 1
		if iterations > 10000:
			break 
	
	print 'Mean squared error after {0} iterations: {1}'.format(iterations,error)

	return iterations, mses, n_correct

def main():	
	train_x = np.array([[0,0],[0,1],[1,0],[1,1]])
	train_y = np.array([0,1,1,0])
	
	rates = np.linspace(0.01,1,50)

	#test converge for several activation functions and learning rates
	for activation in ['sigmoid','tanh','relu']:
		print 'Testing convergence for {} activation'.format(activation)
		iters = []
		for learning_rate in rates:
			iterations, mses, n_correct = train_xor_net(train_x,train_y,learning_rate,activation)
			iters.append(iterations)
		plt.plot(rates,iters,label=activation)
	plt.xlabel('Learning Rate')
	plt.ylabel('Iterations to convergence')	
	plt.legend()
	plt.savefig('convergence_test.png', dpi=300)
	plt.show()
	
if __name__ == '__main__':
	main()
