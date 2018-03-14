import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_style('ticks')

def calc_feature(number):
    '''
    Compute the number of pixels in the lower half of the image for all instances of a certain digit
    '''
    number_feature = np.zeros(len(number))
    ones_number = 1*(number > 0.0)
    for i in range(len(number)):
        temp = ones_number[i,:].reshape(16,16)
        number_feature[i] = np.sum(temp[8:,:])
    return number_feature

def calc_priors(train_x, train_y, class1, class2):
    '''
    Calculate the prior probabilities of the chosen digits
    '''
    class1_train = train_x[train_y == class1,:]
    class2_train = train_x[train_y == class2,:]

    prior_class1 = float(len(class1_train))/(len(class1_train)+len(class2_train))
    prior_class2 = float(len(class2_train))/(len(class1_train)+len(class2_train))
    return prior_class1, prior_class2

def train_bayes(train_x, train_y, class1, class2):
    '''
    Calculate the likelihood of the chosen feature using histograms
    '''
    class1_train = train_x[train_y == class1,:]
    class2_train = train_x[train_y == class2,:]
        
    class1_feature_train = calc_feature(class1_train)  
    class2_feature_train = calc_feature(class2_train)

    minfeature = min(class1_feature_train.min(),class2_feature_train.min())
    maxfeature = max(class1_feature_train.max(),class2_feature_train.max())
    binbounds = np.linspace(minfeature,maxfeature, 15)
    
    #get the bins and corresponding bin values for the chosen classes
    n_class1, bins_class1, patches = plt.hist(class1_feature_train, alpha=0.6, bins=binbounds, normed=True, label=str(class1))
    n_class2, bins_class2, patches = plt.hist(class2_feature_train, alpha=0.6, bins=binbounds, normed=True, label=str(class2))

    #plot the histograms
    plt.legend()
    plt.title('Histogram of pixel counts')
    plt.xlabel('Number of positive pixels in image lower half')
    plt.savefig('{0}_{1}_hist.png'.format(class1,class2), dpi=300)
    plt.show()

    return n_class1, bins_class1, n_class2, bins_class2

def get_test_data(test_x, test_y, class1, class2):
    '''
    Retrieve the data of the chosen digits from the test set
    '''
    class1_test = test_x[test_y == class1,:]
    class2_test = test_x[test_y == class2,:]
    
    print 'Number of instances of class 1 in the test set: ', len(class1_test)
    print 'Number of instances of class 2 in the test set: ', len(class2_test)
    
    class1_feature = calc_feature(class1_test)  
    class2_feature = calc_feature(class2_test)

    return class1_feature, class2_feature

def main():
    train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
    train_y = np.genfromtxt('data/train_out.csv', delimiter=',')

    test_x = np.genfromtxt('data/test_in.csv', delimiter=',')
    test_y = np.genfromtxt('data/test_out.csv', delimiter=',')

    #get the likelihood and priors based on the test data 
    n_class1, bins_class1, n_class2, bins_class2 = train_bayes(train_x, train_y, 5, 7)
    prior1, prior2 = calc_priors(train_x, train_y, 5, 7)
    class1_test, class2_test = get_test_data(test_x, test_y, 5, 7)
    
    all_n = np.concatenate((class1_test,class2_test))
    
    #bin the test data according to the bins defined by the training set
    binned_class1 = np.digitize(all_n, bins_class1)
    binned_class1[binned_class1 >= len(n_class1)] = len(n_class1)-1
    binned_class2 = np.digitize(all_n, bins_class2)
    binned_class2[binned_class2 >= len(n_class2)] = len(n_class2)-1

    #calculate posteriors for each instance of the test data and assign classes accordingly
    classified_class = np.zeros(len(all_n))
    for i in range(len(all_n)):
         if (n_class1[binned_class1[i]-1]*prior1 > n_class2[binned_class2[i]-1]*prior2):
            classified_class[i] = 1
      
    accuracy = (np.sum(1*classified_class[:len(class1_test)]==1)+
    				 np.sum(1*classified_class[len(class1_test):]==0))/float(len(classified_class))
    print 'Accuracy on test set : {}%'.format(accuracy*100)

if __name__ == '__main__':
    main()