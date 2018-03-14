import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import metrics
import seaborn as sns; sns.set()


def plot_confusion_matrix(conf_mat, fmt, classes, title):
    '''
    Plot a heatmap of a given confusion matrix
    '''
    sns.heatmap(conf_mat, square= True,annot=True, fmt = fmt, cbar = False,
               xticklabels= classes, yticklabels= classes)  
    plt.suptitle(title)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig(title)
    plt.show()

def main():
    train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
    train_y = np.genfromtxt('data/train_out.csv',delimiter=',')

    test_x = np.genfromtxt('data/test_in.csv',delimiter=',')
    test_y = np.genfromtxt('data/test_out.csv',delimiter=',')

    #create the clouds and compute the centers
    cld = [train_x[train_y == x,:] for x in range (10)]
    cld = np.array((cld))

    #compute the centers
    xx_mean =np.array(( [np.mean(cld[x],axis=0) for x in range(10)]))

    #compute the radii
    rad = np.zeros(10)
    for i in range(10):
        rad[i] =np.amax(np.array(([np.linalg.norm(cld[i][j] - xx_mean[i])  for j in range(len(cld[i]))])))

    #calculate the distance between the clouds
    dist_cm = np.array(([[np.linalg.norm(xx_mean[x,:]- xx_mean[y,:]) for y in range (10) ] for x in range (10)]))

    #print distance matrix
    classes= np.arange(0,10)

    sns.heatmap(dist_cm, square= True,annot=True, cbar = False,
                   xticklabels= classes, yticklabels= classes)
    plt.suptitle('distance matrix')

    plt.savefig('dist.png')
    plt.show()

    # run the classifier on the training set
    # compute the distance between the image and the clouds
    # the shortest distance is the digit

    tt = np.zeros(10)
    num_1 = np.zeros(len(train_x))
    for i in range(len(train_x)):
        tt_1 = [np.linalg.norm(train_x[i] - xx_mean[y]) for y in range(10)]
        num_1[i] = np.argmin(tt_1)

    # confusion matrix
    c_m = metrics.confusion_matrix(train_y, num_1)
    plot_confusion_matrix(c_m, 'd',classes, 'train')


    cm_per = (c_m.astype('float') / c_m.sum(axis=1)*100)
    plot_confusion_matrix(cm_per.T,'.0f',classes,'Train %')

    # now   running the algorith on the test set 
    num_class = np.zeros(len(test_x))
    for i in range (len(test_x)):
        num_class[i] = np.argmin(np.array(([np.linalg.norm(test_x[i]- xx_mean[y]) for y in range (10)])))

    cm_test = metrics.confusion_matrix(test_y, num_class)
    cm_test_per = (cm_test.astype('float') / cm_test.sum(axis=1)*100)

    plot_confusion_matrix(cm_test,'d',classes,'Test')
    plot_confusion_matrix(cm_test_per.T,'.0f',classes,'Test %')

    # try different distance metrics
    fig, axs = plt.subplots(2,3, figsize=(14.75,9.1), sharex=True, sharey=True)
    ax = axs.ravel()
    dist_points = np.zeros((len(test_x),10))

    # try different distance metrics
    for j, metric in enumerate(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']):
        for i in range(len(test_x)):
            dist_points[i,:] = metrics.pairwise.pairwise_distances(test_x[i,:].reshape(1,-1),xx_mean, metric=metric)[0]
        closest = np.argmin(dist_points, axis=1)
        confusion = metrics.confusion_matrix(test_y, closest)
        confusion_per = (confusion.astype('float') / confusion.sum(axis=1)*100)
        sns.heatmap(confusion_per, square=True,annot=True, fmt = '.0f', cbar = False,
                   xticklabels=classes, yticklabels= classes, ax = ax[j], cmap = sns.cm.rocket_r)
        t_metric =  ('Accuracy of {0} : {1}%'.format(metric,np.trace(confusion)*100./confusion.sum()))
        ax[j].set_adjustable('box-forced')
        ax[j].set_title(t_metric)
    plt.tight_layout()
    plt.savefig('metric.png')
    plt.show()

if __name__ == '__main__':
    main()