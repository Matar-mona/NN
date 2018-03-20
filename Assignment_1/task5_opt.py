import matplotlib.pyplot as plt
import numpy as np
from astroML.decorators import pickle_results

def sigmoid(X):
	return 1/(1+np.exp(-X))
	
def relu(X):
	return np.maximum(X,0)

def train_mnist_net(train_x, train_y, learning_rate=0.01):
	'''
	Train a 2-layer neural network on the MNIST data set
	'''
	w1 = np.random.randn(30,257)*0.5
	w1[:,0] = 1
	w2 = np.random.randn(10,31)*0.5
	w2[:,0] = 1
	learning_rate = 10

	Y_train = np.zeros((10,len(train_y)))
	for i in range(len(train_y)):
		Y_train[int(train_y[i]),i] = 1 

	def mnist_net(X, weights1, weights2):
		'''
		Return the prediction of the network
		'''
		a1 = sigmoid(np.dot(weights1[:,1:],X.T)+weights1[:,0].reshape(-1,1))
		a2 = sigmoid(np.dot(weights2[:,1:],a1)+weights2[:,0].reshape(-1,1))
		return a2, a2.argmax(axis=0)

	def mse(weights1,weights2):
		'''
		Compute the mean squared error of the weights
		'''
		mse = 0
		A2, y_hat = mnist_net(train_x,weights1,weights2)
		mse = np.sum((A2-Y_train)**2)
		return mse/float(len(train_y))

	
	def grdmse(weights1, weights2):
		'''
		Compute the gradient of the weights
		'''
		err = 0.001
		grads1 = np.zeros((30,257))
		grads2 = np.zeros((10,31))
		weights_err1 = np.copy(weights1)
		weights_err2 = np.copy(weights2)
		const_mse = mse(weights1,weights2)
		for i in range(10):
			for j in range(31):
				weights_err2[i,j] += err
				grads2[i,j] = (mse(weights1,weights_err2)-const_mse)/err
				weights_err2[i,j] -= err
		for i in range(30):
			for j in range(257):
				weights_err1[i,j] += err
				grads1[i,j] = (mse(weights_err1,weights2)-const_mse)/err
				weights_err1[i,j] -= err
		return grads1, grads2
	
	missclass = 10
	iterations = 0
	mses = []
	#iterate until there are no cases missclassified or 1000 iterations are reached                                 
	while missclass > 0:
		grads1, grads2 = grdmse(w1,w2)
		w1 -= learning_rate*grads1
		w2 -= learning_rate*grads2
		error = mse(w1,w2)
		mses.append(error)
		A2, y_h = mnist_net(train_x,w1,w2)
		missclass = np.sum(1*np.not_equal(y_h,train_y))
		iterations += 1
		print 'Mean squared error at iteration {0}: {1}'.format(iterations,error)
		if iterations > 1000:
			break 	

	print 'Mean squared error after {0} iterations: {1}'.format(iterations,error)
	
	plt.plot(np.arange(iterations),mses, color='k')
	plt.ylabel('Mean squared error')
	plt.xlabel('Iteration')
	plt.savefig('opt_task5.png', dpi=300)
	plt.close()

	return w1, w2
	
def main():
	train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
	train_y = np.genfromtxt('data/train_out.csv', delimiter=',')

	test_x = np.genfromtxt('data/test_in.csv', delimiter=',')
	test_y = np.genfromtxt('data/test_out.csv', delimiter=',')

	w1, w2 = train_mnist_net(train_x, train_y, learning_rate=5)

	#predict values for the test set
	a1 = sigmoid(np.dot(w1[:,1:],test_x.T)+w1[:,0].reshape(-1,1))
	a2 = sigmoid(np.dot(w2[:,1:],a1)+w2[:,0].reshape(-1,1))
	y_h = a2.argmax(axis=0)
	missclassified = 1*np.not_equal(y_h,test_y)
	acctest = 1-np.sum(missclassified)/float(len(missclassified))
	print 'Accuracy on test set: {0}%'.format(acctest*100)
	
if __name__ == '__main__':
    main()
