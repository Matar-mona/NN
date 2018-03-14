import matplotlib.pyplot as plt
import numpy as np

def train_multiclass_perceptron(train_x, train_y, learning_rate=0.01):
	w = np.random.randn(10,257)*0.01
	w[:,0] = 1

	missclass = 10
	iterations = 0
	accuracy = 0
	acc = []
	while missclass > 0:
		a = np.dot(w[:,1:],train_x.T)+w[:,0].reshape(-1,1)
		y_h = a.argmax(axis=0)
		missclassified = 1*np.not_equal(y_h,train_y)
		for i in range(len(missclassified)):
			w[int(train_y[i]),1:] += learning_rate*missclassified[i]*train_x[i,:]
			w[int(y_h[i]),1:] -= learning_rate*missclassified[i]*train_x[i,:]
		accuracy = 1-np.sum(missclassified)/float(len(missclassified))
		acc.append(accuracy)
		missclass = np.sum(missclassified)
		iterations += 1
		if iterations > 10000:
			break
	
	plt.plot(np.arange(iterations), acc, color='k')
	plt.ylabel('Accuracy')
	plt.xlabel('Iteration')
	plt.savefig('task4.png', dpi=300)
	plt.show()

	return w

def main():
	train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
	train_y = np.genfromtxt('data/train_out.csv', delimiter=',')

	test_x = np.genfromtxt('data/test_in.csv', delimiter=',')
	test_y = np.genfromtxt('data/test_out.csv', delimiter=',')
	
	w = train_multiclass_perceptron(train_x, train_y)
	
	a = np.dot(w[:,1:],test_x.T)+w[:,0].reshape(-1,1)
	y_h = a.argmax(axis=0)
	missclassified = 1*np.not_equal(y_h,test_y)
	acctest = 1-np.sum(missclassified)/float(len(missclassified))
	print 'Accuracy on test set: {0}%'.format(acctest*100)
	
if __name__ == '__main__':
	main()
