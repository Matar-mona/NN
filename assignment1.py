import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
train_y = np.genfromtxt('data/train_out.csv', delimiter=',')

test_x = np.genfromtxt('data/test_in.csv', delimiter=',')
test_y = np.genfromtxt('data/test_out.csv', delimiter=',')

x_means = np.zeros((10,256))
radius = np.zeros((10))
points = 0

for i in range(10):
	print 'Calculating means for ', i
	X = train_x[train_y == i,:]
	x_means[i,:] = X.mean(axis=0)

	radius[i] = np.amax(np.linalg.norm(X - x_means[i,:]))
	print 'Radius: ', radius[i]
	print 'Number of points in digit radius: ', len(X[:,0])
	points += len(X[:,0])
	
print points
print x_means.shape
print train_x.shape

dist_clouds = np.zeros((10,10)) 

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

def euclidean(x_means, train_x):
	for i in range(10):
		for j in range(10):
			dist_clouds[i,j] = np.linalg.norm(x_means[:,i]-x_means[:,j])
	
	print dist_clouds

	dist_points = np.zeros((10,len(train_x)))

	for j in range(len(train_x)):
		for i in range(10):
			dist_points[i,j] = np.linalg.norm(train_x[j,:] - x_means[:,i])
	return dist_points

dist_points = np.zeros((len(train_x),10))

fig, axs = plt.subplots(2,3, figsize=(20,15), sharex=True, sharey=True)
ax = axs.ravel()

for j, metric in enumerate(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']):
	for i in range(len(train_x)):
		dist_points[i,:] = metrics.pairwise.pairwise_distances(train_x[i,:].reshape(1,-1),x_means, metric=metric)[0]
	closest = np.argmin(dist_points, axis=1)

	confusion = metrics.confusion_matrix(train_y, closest)

	plot_confusion_matrix(confusion, classes=['0','1','2','3','4','5','6','7','8','9'], ax=ax[j], title=metric, normalize=True)
	print 'Accuracy of {0} : {1}%'.format(metric,np.trace(confusion)*100./confusion.sum())
plt.tight_layout()
plt.show()
