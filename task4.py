import matplotlib.pyplot as plt
import numpy as np

def main():
	train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
	train_y = np.genfromtxt('data/train_out.csv', delimiter=',')

	test_x = np.genfromtxt('data/test_in.csv', delimiter=',')
	test_y = np.genfromtxt('data/test_out.csv', delimiter=',')
	
	w = np.random.randn(10,257)*0.01
	w[:,0] = 1
	learning_rate = 0.01

	missclassified = 10
	iterations = 0
	accuracy = 0
	acc = []
	while missclassified > 0:
		a = np.dot(w[:,1:],train_x.T)+w[:,0].reshape(-1,1)
		y_h = a.argmax(axis=0)
		missclassified = 1*np.not_equal(y_h,train_y)
		for i in range(len(missclassified)):
			w[int(train_y[i]),1:] += learning_rate*missclassified[i]*train_x[i,:]
		accuracy = 1-np.sum(missclassified)/float(len(missclassified))
		acc.append(accuracy)
		missclass = np.sum(missclassified)
		iterations += 1
		if iterations > 10000:
			break
	
	acctrain = 1-np.sum(missclassified)/float(len(missclassified))
	print 'Accuracy on training set after {1} iterations: {0}%'.format(acctrain*100,iterations)
	
	plt.plot(np.arange(iterations), acc, color='k')
	plt.ylabel('Accuracy')
	plt.xlabel('Iteration')
	plt.savefig('task4.png', dpi=300)
	plt.show()
	
	a = np.dot(w[:,1:],test_x.T)+w[:,0].reshape(-1,1)
	y_h = a.argmax(axis=0)
	missclassified = 1*np.not_equal(y_h,test_y)
	acctest = 1-np.sum(missclassified)/float(len(missclassified))
	print 'Accuracy on test set: {0}%'.format(acctest*100)
	
if __name__ == '__main__':
	main()
