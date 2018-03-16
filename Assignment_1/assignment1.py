from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cPickle
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def pickle_to_file(data, fname):
	fh = open(fname, 'w') 
	cPickle.dump(data, fh) 
	fh.close() 

def pickle_from_file(fname): 
    fh = open(fname, 'r') 
    data = cPickle.load(fh) 
    fh.close() 
    return data

def custom_euclidean(x_means, train_x):
    dist_points = np.zeros((10,len(train_x)))

    for j in range(len(train_x)):
        for i in range(10):
            dist_points[i,j] = np.linalg.norm(train_x[j,:] - x_means[:,i])
    return dist_points
    
def calc_ratio(number):
    number_ratio = np.zeros(len(number))
    ones_number = 1*(number==1)
    for i in range(len(number)):
      temp = ones_number[i,:].reshape(16,16)
      number_ratio[i] = np.sum(temp[:,:8])/float(np.sum(temp[:,8:]))
    return number_ratio
    
def mean_ones(number):
	ones_number = 1*(number==1)
	return ones_number.sum(axis=1)
    
def sigmoid(X):
	return 1/(1+np.exp(-X))
	
def relu(X):
	return np.maximum(X,0)

def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '{:.0f}%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_xlim(0,9)
    ax.set_ylim(9,0)
    
def task1_2(train_x, train_y, test_x, test_y):
    x_means = np.zeros((10,256))
    radius = np.zeros((10))

    for i in range(10):
    	print 'Calculating means for ', i
    	X = train_x[train_y == i,:]
    	x_means[i,:] = X.mean(axis=0)

    	radius[i] = np.amax(np.linalg.norm(X - x_means[i,:]))
    	print 'Radius: ', radius[i]
    	print 'Number of points in digit radius: ', len(X[:,0])
    	
    print x_means.shape
    print train_x.shape

    dist_clouds = np.zeros((10,10)) 

    for i in range(10):
        for j in range(10):
            dist_clouds[i,j] = np.linalg.norm(x_means[:,i]-x_means[:,j])
    print 'Distance between clouds:'
    print dist_clouds

    dist_points = np.zeros((len(test_x),10))

    fig, axs = plt.subplots(2,3, figsize=(14,9.2), sharex=True, sharey=True)
    ax = axs.ravel()

    for j, metric in enumerate(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']):
      for i in range(len(test_x)):
    	   dist_points[i,:] = metrics.pairwise.pairwise_distances(test_x[i,:].reshape(1,-1),
    	                                                          x_means, metric=metric)[0]
      closest = np.argmin(dist_points, axis=1)
      confusion = metrics.confusion_matrix(test_y, closest)
      plot_confusion_matrix(confusion, classes=['0','1','2','3','4','5','6','7','8','9'], 
                            ax=ax[j], title=metric, normalize=True)
      print 'Accuracy of {0} : {1}%'.format(metric,np.trace(confusion)*100./confusion.sum())
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    plt.show()
    
def task3(train_x, train_y, test_x, test_y):
    fives_train = train_x[train_y == 5,:]
    threes_train = train_x[train_y == 1,:]
        
    five_ratio_train = calc_ratio(fives_train)  
    three_ratio_train = calc_ratio(threes_train)
    
    n_five, bins_five, patches = plt.hist(five_ratio_train, bins=20, alpha=0.5, normed=True)
    n_three, bins_three, patches = plt.hist(three_ratio_train[three_ratio_train < 10], bins=20, 														  alpha=0.5, normed=True)
    plt.show()
    
    fives = test_x[test_y == 5,:]
    threes = test_x[test_y == 1,:]
    
    print 'Number of 5s: ', len(fives)
    print 'Number of 3s: ', len(threes)
    
    prior_three = float(len(threes_train))/(len(fives_train)+len(threes_train))
    prior_five = float(len(fives_train))/(len(fives_train)+len(threes_train))
    
    five_ratio = calc_ratio(fives)  
    three_ratio = calc_ratio(threes)
    
    all_n = np.concatenate((five_ratio,three_ratio))
    
    binned_five = np.digitize(all_n, bins_five)
    binned_five[binned_five >= len(n_five)] = len(n_five)-1
    binned_three = np.digitize(all_n, bins_three)
    binned_three[binned_three >= len(n_three)] = len(n_three)-1
    
    classified_five = np.zeros(len(all_n))
    for i in range(len(all_n)):
         if (n_five[binned_five[i]]*prior_five > n_three[binned_three[i]]*prior_three):
            classified_five[i] = 1
      
    accuracy = (np.sum(1*classified_five[:len(fives)]==1)+
    				 np.sum(1*classified_five[len(fives):]==0))/float(len(classified_five))
    print 'Accuracy : {}%'.format(accuracy*100)
    
def task4(train_x, train_y, test_x, test_y):
	w = np.random.randn(10,257)*0.01
	w[:,0] = 1
	learning_rate = 0.01

	iterations = 0
	accuracy = 0
	acc = []
	while accuracy < 0.97:
		a = np.dot(w[:,1:],train_x.T)+w[:,0].reshape(-1,1)
		y_h = a.argmax(axis=0)
		missclassified = 1*np.not_equal(y_h,train_y)
		for i in range(len(missclassified)):
			w[int(train_y[i]),1:] += learning_rate*missclassified[i]*train_x[i,:]
		accuracy = 1-np.sum(missclassified)/float(len(missclassified))
		acc.append(accuracy)
		iterations += 1
		if iterations > 2000:
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
	
def task5(train_x, train_y, test_x, test_y):
	
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
		for i in range(3):
			for j in range(3):
				weights_err[i,j] += err
				grads[i,j] = (mse(weights_err)-mse(weights))/err
				weights_err[i,j] -= err
		return grads
			
	w = np.random.randn(3,3)*0.5
	learning_rate = 0.1
	
	error = 1
	iterations = 0
	mses = []                                     
	while error > 0.01:
		w -= learning_rate*grdmse(w)
		error = mse(w)
		mses.append(error)
		iterations += 1
		if iterations > 10000:
			break 
	
	x = np.array([[0,0],[0,1],[1,0],[1,1]])
	for i in range(4):
		print xor_net(x[i,0],x[i,1],w)
	print 'Mean squared error after {0} iterations: {1}'.format(iterations,error)
	
	plt.plot(np.arange(iterations),mses, color='k')
	plt.ylabel('Mean squared error')
	plt.xlabel('Iteration')
	plt.savefig('task5.png', dpi=300)
	plt.show()

def task5_opt(train_x, train_y, test_x, test_y, Y_train):

	def mnist_net(X, weights1, weights2):
		a1 = sigmoid(np.dot(weights1[:,1:],X.T)+weights1[:,0].reshape(-1,1))
		a2 = sigmoid(np.dot(weights2[:,1:],a1)+weights2[:,0].reshape(-1,1))
		return a2, a2.argmax(axis=0)
	
	def mse(weights1,weights2):
		mse = 0
		A2, y_hat = mnist_net(train_x,weights1,weights2)
		mse = np.sum((A2-Y_train)**2)
		return mse/float(len(train_y))
	
	def grdmse(weights1, weights2):
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
			
	w1 = np.random.randn(30,257)*0.5
	w1[:,0] = 1
	w2 = np.random.randn(10,31)*0.5
	w2[:,0] = 1
	learning_rate = 5
	
	error = 1
	iterations = 0
	mses = []                                     
	while error > 0.01:
		grads1, grads2 = grdmse(w1,w2)
		w1 -= learning_rate*grads1
		w2 -= learning_rate*grads2
		error = mse(w1,w2)
		mses.append(error)
		iterations += 1
		print 'Mean squared error at iteration {0}: {1}'.format(iterations,error)
		if iterations > 1000:
			break 

	pickle_to_file((w1,w2), 'weights.pkl')		

	print 'Mean squared error after {0} iterations: {1}'.format(iterations,error)
	
	plt.plot(np.arange(iterations),mses, color='k')
	plt.ylabel('Mean squared error')
	plt.xlabel('Iteration')
	plt.savefig('opt_task5.png', dpi=300)
	plt.close()

	a1 = sigmoid(np.dot(w1[:,1:],X.T)+w1[:,0].reshape(-1,1))
	a2 = sigmoid(np.dot(w2[:,1:],a1)+w2[:,0].reshape(-1,1))
	y_h = a2.argmax(axis=0)
	missclassified = 1*np.not_equal(y_h,test_y)
	acctest = 1-np.sum(missclassified)/float(len(missclassified))
	print 'Accuracy on test set: {0}%'.format(acctest*100)

def main():
	train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
	train_y = np.genfromtxt('data/train_out.csv', delimiter=',')

	test_x = np.genfromtxt('data/test_in.csv', delimiter=',')
	test_y = np.genfromtxt('data/test_out.csv', delimiter=',')

	y_matrix = np.zeros((10,len(train_y)))
	for i in range(len(train_y)):
		y_matrix[int(train_y[i]),i] = 1 

	#task1_2(train_x, train_y, test_x, test_y)
	#task3(train_x, train_y, test_x, test_y)
	#task4(train_x, train_y, test_x, test_y)
	#task5(train_x, train_y, test_x, test_y)
	task5_opt(train_x, train_y, test_x, test_y, y_matrix)

if __name__ == '__main__':
    main()
