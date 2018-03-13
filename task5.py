from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(X):
	return 1/(1+np.exp(-X))

def relu(X):
	return np.maximum(X,0)

def xor_net(x1,x2,weights):
	x = np.array([1, x1, x2])
	a1 = np.tanh(np.dot(weights[0,:],x))
	a2 = np.tanh(np.dot(weights[1,:],x))
	a = np.array([1, a1, a2])
	return np.tanh(np.dot(weights[2,:],a))

def mse(weights):
	x = np.array([[0,0],[0,1],[1,0],[1,1]])
	y = np.array([0,1,1,0])
	mse = 0
	for i in range(4):
		xor = xor_net(x[i,0],x[i,1],weights)
		mse += (xor-y[i])**2
	return mse/4.

def grdmse(weights):
	err = 0.001
	grads = np.zeros((3,3))
	weights_err = np.copy(weights)
	const_mse = mse(weights)
	for i in range(3):
		for j in range(3):
			weights_err[i,j] += err
			grads[i,j] = (mse(weights_err)-const_mse)/err
			weights_err[i,j] -= err
	return grads

def main():	
	w = np.random.randn(3,3)*0.5
	learning_rate = 0.1
	test_x = np.array([[0,0],[0,1],[1,0],[1,1]])
	test_y = np.array([0,1,1,0])

	iterations = 0
	n_missclassified = 4
	mses = []
	n_correct = []                                     
	while n_missclassified > 0:
		n_missclassified = 0
		w -= learning_rate*grdmse(w)
		error = mse(w)
		mses.append(error)
		for i in range(4):
			y_h = xor_net(test_x[i,0],test_x[i,1],w)
			n_missclassified += abs(round(y_h)-test_y[i])
		n_correct.append(4-n_missclassified)
		iterations += 1
		if iterations > 10000:
			break 
	
	print 'Mean squared error after {0} iterations: {1}'.format(iterations,error)
	
	fig = plt.figure()

	axL = fig.add_subplot(1,1,1)
	axL.set_ylabel('Correct predictions of XOR')
	axL.set_xlabel('Iteration')
	axL.plot(np.arange(iterations),n_correct, 'k')
	axL.set_xlim(0,iterations)
	axL.yaxis.set_major_locator(MaxNLocator(integer=True))

	axR = fig.add_subplot(1,1,1, sharex=axL, frameon=False)
	axR.yaxis.tick_right()
	axR.yaxis.set_label_position('right')
	axR.set_ylabel('Mean squared error')
	axR.plot(np.arange(iterations),mses, 'k--')
	
#	plt.savefig('task5.png', dpi=300)
	plt.show()
	
if __name__ == '__main__':
	main()
