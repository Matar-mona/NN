from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import numpy as np
import matplotlib.pyplot as plt

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
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes)

    fmt = '{:.0f}%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
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
    print dist_clouds

    dist_points = np.zeros((len(test_x),10))

    fig, axs = plt.subplots(2,3, figsize=(20,15), sharex=True, sharey=True)
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
   
def g_step(X):
	X[X >= 0] = 1
	X[X < 0] = -1
	return X 

def main():
	train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
	train_y = np.genfromtxt('data/train_out.csv', delimiter=',')

	test_x = np.genfromtxt('data/test_in.csv', delimiter=',')
	test_y = np.genfromtxt('data/test_out.csv', delimiter=',')

	#task1_2(train_x, train_y, test_x, test_y)
	#task3(train_x, train_y, test_x, test_y)

	np.random.seed(42)
	w = np.random.randn(10,257)*0.01
	w[:,0] = 1
	learning_rate = 0.05
	g = np.zeros((len(train_x),10))
	
	y_true = np.full((10,len(train_y)), fill_value=-1) 
	for i in range(len(train_y)):
		y_true[int(train_y[i]),i]=1

	iterations = 0
	while iterations < 10:
		a = np.dot(w[:,1:],train_x.T)+w[:,0].reshape(-1,1)
		g = g_step(a)
		diff = np.subtract(y_true,g)
		w[:,1:] += learning_rate*np.dot(y_true[:,diff.nonzero()[1][0]].reshape(-1,1),train_x[diff.nonzero()[1][0],:].reshape(1,-1))
		iterations += 1
	
	accuracy = 1-np.sum(1*(diff.sum(axis=0)>0))/float(len(train_x))
	print 'Accuracy after {1} iterations: {0}%'.format(accuracy*100,iterations)

if __name__ == '__main__':
    main()
